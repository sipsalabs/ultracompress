"""lambada_overlay_mixed.py -- Claim 18D: mixed-precision row-overlay.

Intuition: Claim 17 (fp16) stores every restored row exactly. Claim 18A
(fp8) halves the per-row cost and buys ~2x rows at matched bpw, but
pays ~0.5% per-element quant noise. The rows that hurt most when
quantized are the ones with the largest residual -- exactly the top of
the score ranking. So restore the top ~rho_hi rows in fp16 and the
next ~rho_lo rows in fp8.

Budget: (rho_hi, rho_lo) is chosen so the overlay-bpw matches a target
operating point. Per-row cost:
  fp16:   16*I + 32        bits
  fp8:     8*I + 16 + 32   bits
  cb:     gbpw * I         bits  (what the row cost before overlay)

Net overhead per row:
  fp16_net = (16 - gbpw) * I + 32
  fp8_net  = ( 8 - gbpw) * I + 48
So at gbpw ~= 2.78 the per-row bit premium is roughly 2.4x higher for fp16
than fp8 -- which is why mixed is a strictly richer bit-budget than either
alone at the same effective bpw.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign
from lambada_overlay import MODELS


def _fp8_round_trip(x: torch.Tensor):
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 448.0
    xs = (x / scale).to(torch.float8_e4m3fn)
    xq = xs.to(torch.float32) * scale
    return xq


def _reconstruct_v17_with_mixed_overlay(W_fp16, role, bank, s_col, D,
                                         rot, device, rho_hi, rho_lo,
                                         score_mode="weighted"):
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    W_scaled = W * s.unsqueeze(0)
    Wrot = W_scaled @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    chunk = max(25_000, (200_000 * 2048) // max(cb1.shape[0], 1))
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=chunk)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)
    raw = W - Wq
    diff = raw * s.unsqueeze(0) if score_mode == "weighted" else raw
    score = (diff * diff).sum(1)

    K_hi = max(0, int(round(rho_hi * O)))
    K_lo = max(0, int(round(rho_lo * O)))
    K_total = K_hi + K_lo
    n_fp16 = n_fp8 = 0
    if K_total > 0 and K_total <= O:
        # top K_total by score; top K_hi of those -> fp16, next K_lo -> fp8
        top = score.topk(K_total)
        idx_all = top.indices
        # already sorted descending by .topk
        idx_hi = idx_all[:K_hi]
        idx_lo = idx_all[K_hi:K_hi + K_lo]
        if K_hi > 0:
            Wq[idx_hi] = W[idx_hi]           # exact fp16
            n_fp16 = K_hi
        if K_lo > 0:
            Wq[idx_lo] = _fp8_round_trip(W[idx_lo])
            n_fp8 = K_lo

    del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, s, raw, diff, score
    out = Wq.to(dtype=torch.float16, device="cpu")
    del Wq
    return out, n_fp16, n_fp8


def substitute_v17_mixed_overlay(model, state_dict, v17, device, D,
                                  rho_hi, rho_lo, score_mode="weighted"):
    rots = {}
    dims = sorted({v.shape[1] for k, v in state_dict.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims:
        rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v17["banks"]
    s_col = v17["s_col"]
    hf_keys = [k for k in state_dict.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and state_dict[k].ndim == 2
               and state_dict[k].shape[1] % D == 0]
    print(f"  substituting {len(hf_keys)} body Linears "
          f"(v17+mixed-overlay, rho_hi={rho_hi}, rho_lo={rho_lo}, score={score_mode})")
    tot_hi = tot_lo = 0
    tot_params_hi = tot_params_lo = 0
    tot_rows = tot_params = 0
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = state_dict[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        W_new, n16, n8 = _reconstruct_v17_with_mixed_overlay(
            W, role, bank, s, D, rots[W.shape[1]], device, rho_hi, rho_lo, score_mode)
        tot_hi += n16; tot_lo += n8
        tot_rows += W.shape[0]
        tot_params_hi += n16 * W.shape[1]
        tot_params_lo += n8 * W.shape[1]
        tot_params += W.numel()
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()
    total_restored = tot_hi + tot_lo
    total_params_restored = tot_params_hi + tot_params_lo
    print(f"  mixed-overlay: fp16 rows {tot_hi}, fp8 rows {tot_lo}, "
          f"total {total_restored}/{tot_rows} "
          f"({100*total_restored/max(tot_rows,1):.3f}%)")
    gbpw = float(v17.get("global_bpw", 2.78))
    avg_I_hi = (tot_params_hi / tot_hi) if tot_hi > 0 else 0.0
    avg_I_lo = (tot_params_lo / tot_lo) if tot_lo > 0 else 0.0
    excess_hi = tot_hi * ((16 - gbpw) * avg_I_hi + 32) if tot_hi > 0 else 0.0
    excess_lo = tot_lo * (( 8 - gbpw) * avg_I_lo + 16 + 32) if tot_lo > 0 else 0.0
    overlay_bpw = (excess_hi + excess_lo) / max(tot_params, 1)
    print(f"  mixed-overlay bpw overhead: +{overlay_bpw:.4f} "
          f"(base {gbpw:.4f} -> effective {gbpw+overlay_bpw:.4f}) "
          f"[fp16 part +{excess_hi/max(tot_params,1):.4f}, fp8 part +{excess_lo/max(tot_params,1):.4f}]")
    return {
        "n_fp16_rows": tot_hi, "n_fp8_rows": tot_lo,
        "n_total_rows": tot_rows,
        "rows_pct": 100*total_restored/max(tot_rows,1),
        "params_pct": 100*total_params_restored/max(tot_params,1),
        "overlay_bpw": float(overlay_bpw),
        "base_bpw": gbpw,
        "effective_bpw": float(gbpw + overlay_bpw),
    }


def run_one(name, model_id, teacher_path, v17_path, tokens_path,
            n, seq_len, device, rho_hi, rho_lo, score_mode="weighted",
            tier="hifi+mixed-overlay"):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from eval_v16_ppl import measure_ppl
    from eval_claim16_topk import measure_topk

    print(f"\n[{name}] teacher -> {teacher_path}", flush=True)
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    toks = torch.load(tokens_path, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, toks.numel() - seq_len - 1, (n,), generator=g)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache = measure_topk(model, toks, starts, seq_len, device, teacher_topk=None)
    tch_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] teacher  PPL={tch_ppl:.4f}  T1={tch_topk['t1_gt']*100:.2f}%  "
          f"T10={tch_topk['t10_gt']*100:.2f}%   ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = v17.get("D", 8)
    ov = substitute_v17_mixed_overlay(model, sd, v17, device, D, rho_hi, rho_lo, score_mode)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17+mixedov rho_hi={rho_hi} rho_lo={rho_lo} score={score_mode}  "
          f"PPL={v17_ppl:.4f}  T1={v17_topk['t1_gt']*100:.2f}%  "
          f"T10={v17_topk['t10_gt']*100:.2f}%  "
          f"T1_vs_teacher={v17_topk['t1_agree']*100:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    ppl_ratio = v17_ppl / tch_ppl
    t1_ret  = v17_topk['t1_gt']  / tch_topk['t1_gt']  if tch_topk['t1_gt']  > 0 else 0.0
    t10_ret = v17_topk['t10_gt'] / tch_topk['t10_gt'] if tch_topk['t10_gt'] > 0 else 0.0

    rec = {
        "name": name, "model_id": model_id, "fit": v17_path,
        "rho_hi": rho_hi, "rho_lo": rho_lo,
        "rho": rho_hi + rho_lo,
        "score_mode": score_mode, "n": n, "seq_len": seq_len,
        "teacher_ppl": float(tch_ppl),
        "teacher_t1":  float(tch_topk['t1_gt']),
        "teacher_t10": float(tch_topk['t10_gt']),
        "v17_ppl": float(v17_ppl),
        "v17_t1":  float(v17_topk['t1_gt']),
        "v17_t10": float(v17_topk['t10_gt']),
        "v17_t1_vs_teacher": float(v17_topk['t1_agree']),
        "ppl_ratio": float(ppl_ratio),
        "t1_ret":  float(t1_ret),
        "t10_ret": float(t10_ret),
        "tier": tier,
    }
    rec.update(ov)
    del model, sd, toks
    torch.cuda.empty_cache(); gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    # Default grid: two matched-bpw operating points (~2.79 and ~2.83)
    # Target ~2.79 bpw: (0.001 fp16, 0.003 fp8)  -> ~0.013 + 0.016 = 0.029 bpw
    # Target ~2.83 bpw: (0.002 fp16, 0.008 fp8)  -> ~0.026 + 0.042 = 0.068 bpw
    ap.add_argument("--splits", default="0.001:0.003,0.002:0.008",
                    help="comma-separated rho_hi:rho_lo pairs")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="lambada_overlay_mixed_results.json")
    ap.add_argument("--score", default="weighted", choices=["weighted", "unweighted"])
    args = ap.parse_args()

    splits = []
    for s in args.splits.split(","):
        hi, lo = s.split(":")
        splits.append((float(hi), float(lo)))

    models = MODELS
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    if only:
        models = [m for m in models if any(k in m[0].lower() for k in only)]
    tier_label = "hifi+mixed-overlay"

    results = []
    if os.path.exists(args.out):
        try: results = json.load(open(args.out))
        except Exception: results = []
    done = {(r["name"], r["rho_hi"], r["rho_lo"], r.get("score_mode","weighted")) for r in results}

    for rho_hi, rho_lo in splits:
        for name, mid, teacher, fit, tokens in models:
            key = (name, rho_hi, rho_lo, args.score)
            if key in done:
                print(f"[skip] {name} ({rho_hi},{rho_lo}) already done")
                continue
            try:
                rec = run_one(name, mid, teacher, fit, tokens,
                              args.n, args.seq_len, args.device,
                              rho_hi, rho_lo, score_mode=args.score,
                              tier=tier_label)
                results.append(rec)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[mixed-overlay] wrote {args.out} ({len(results)})")
            except Exception as e:
                print(f"[FAIL] {name} ({rho_hi},{rho_lo}): {type(e).__name__}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache(); gc.collect()

    print("\n================ mixed-overlay summary ================")
    print(f"{'model':<18} {'rho_hi':>7} {'rho_lo':>7} {'t1_ret':>7} {'ppl_r':>7} {'eff_bpw':>7}")
    for r in sorted(results, key=lambda x: (x["rho_hi"]+x["rho_lo"], x["name"])):
        print(f"{r['name']:<18} {r['rho_hi']:>7.4f} {r['rho_lo']:>7.4f} "
              f"{r['t1_ret']*100:>6.2f}% {r['ppl_ratio']:>6.3f} "
              f"{r.get('effective_bpw',0):>7.4f}")


if __name__ == "__main__":
    main()
