"""lambada_overlay_int4.py -- Claim 18C: int4 row-overlay.

Extends the fp16 -> fp8 -> int4 precision axis explored in Claim 18A.

Per restored row:
  - fp16 row-overlay:  16*I + 32 bits        (Claim 17)
  - fp8  row-overlay:   8*I + 16 + 32 bits   (Claim 18A)
  - int4 row-overlay:   4*I + 16 + 32 bits   (Claim 18C, this file)

int4 storage: symmetric per-row scale `scale = absmax(row) / 7`, then
`xq = round(x / scale).clamp(-7, 7)` stored as int4, reconstructed as
`xq * scale`. scale is fp16 (16 bits) + uint32 row index (32 bits).

At matched overlay bpw, int4 gives roughly 10x the row density vs fp16
and ~3x vs fp8. The risk is that per-element quantization error grows
enough to out-weigh the row-count advantage -- we measure it here.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign
from lambada_overlay import MODELS


def _int4_round_trip(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 round-trip, per-row absmax-scaled.

    Returns (xq_fp32, per_row_scale_fp32).
    """
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0                                     # [rows, 1]
    q = torch.round(x / scale).clamp_(-7.0, 7.0)             # int4-valued floats
    xq = q * scale
    return xq, scale.squeeze(1)


def _reconstruct_v17_with_int4_overlay(W_fp16, role, bank, s_col, D,
                                        rot, device, rho, score_mode="weighted"):
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
    K = max(1, int(round(rho * O))) if rho > 0 else 0
    n_restored = 0
    if K > 0:
        idx = score.topk(K).indices
        rows_fp32 = W[idx]
        rows_q, _scale = _int4_round_trip(rows_fp32)
        Wq[idx] = rows_q
        n_restored = K

    del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, s, raw, diff, score
    out = Wq.to(dtype=torch.float16, device="cpu")
    del Wq
    return out, n_restored


def substitute_v17_int4_overlay(model, state_dict, v17, device, D, rho,
                                 score_mode="weighted"):
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
          f"(v17+int4-overlay, rho={rho}, score={score_mode})")
    total_restored = 0; total_rows = 0
    total_params_restored = 0; total_params = 0
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = state_dict[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        W_new, nr = _reconstruct_v17_with_int4_overlay(
            W, role, bank, s, D, rots[W.shape[1]], device, rho, score_mode)
        total_restored += nr
        total_rows += W.shape[0]
        total_params_restored += nr * W.shape[1]
        total_params += W.numel()
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()
    print(f"  int4-overlay: restored {total_restored}/{total_rows} rows "
          f"({100*total_restored/max(total_rows,1):.3f}%), "
          f"{total_params_restored}/{total_params} params "
          f"({100*total_params_restored/max(total_params,1):.3f}%)")
    gbpw = float(v17.get("global_bpw", 2.78))
    # per restored row in int4: 4*I + 16 (scale) + 32 (idx) bits
    if total_restored > 0:
        avg_I = total_params_restored / total_restored
        excess_bits = total_restored * ((4 - gbpw) * avg_I + 16 + 32)
    else:
        excess_bits = 0.0
    overlay_bpw = excess_bits / max(total_params, 1)
    print(f"  int4-overlay bpw overhead: +{overlay_bpw:.4f} "
          f"(base {gbpw:.4f} -> effective {gbpw+overlay_bpw:.4f})")
    return {
        "n_restored_rows": total_restored,
        "n_total_rows": total_rows,
        "rows_pct": 100*total_restored/max(total_rows,1),
        "params_pct": 100*total_params_restored/max(total_params,1),
        "overlay_bpw": float(overlay_bpw),
        "base_bpw": gbpw,
        "effective_bpw": float(gbpw + overlay_bpw),
    }


def run_one(name, model_id, teacher_path, v17_path, tokens_path,
            n, seq_len, device, rho, score_mode="weighted",
            tier="hifi+int4-overlay"):
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
    ov = substitute_v17_int4_overlay(model, sd, v17, device, D, rho, score_mode)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17+int4ov rho={rho} score={score_mode}  PPL={v17_ppl:.4f}  "
          f"T1={v17_topk['t1_gt']*100:.2f}%  T10={v17_topk['t10_gt']*100:.2f}%  "
          f"T1_vs_teacher={v17_topk['t1_agree']*100:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    ppl_ratio = v17_ppl / tch_ppl
    t1_ret  = v17_topk['t1_gt']  / tch_topk['t1_gt']  if tch_topk['t1_gt']  > 0 else 0.0
    t10_ret = v17_topk['t10_gt'] / tch_topk['t10_gt'] if tch_topk['t10_gt'] > 0 else 0.0

    rec = {
        "name": name, "model_id": model_id, "fit": v17_path,
        "rho": rho, "score_mode": score_mode, "n": n, "seq_len": seq_len,
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
    ap.add_argument("--rhos", default="0.021,0.054",
                    help="matched bpw to fp16 rho=0.002 (~+0.026) and rho=0.005 (~+0.066)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="lambada_overlay_int4_results.json")
    ap.add_argument("--score", default="weighted", choices=["weighted", "unweighted"])
    ap.add_argument("--base", action="store_true")
    args = ap.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    models = MODELS
    if args.base:
        models = [(n, mid, t, fit.replace("v17hi_fit_", "v17_fit_"), tok)
                  for (n, mid, t, fit, tok) in MODELS]
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    if only:
        models = [m for m in models if any(k in m[0].lower() for k in only)]
    tier_label = "base+int4-overlay" if args.base else "hifi+int4-overlay"

    results = []
    if os.path.exists(args.out):
        try: results = json.load(open(args.out))
        except Exception: results = []
    done = {(r["name"], r["rho"], r.get("score_mode","weighted"),
             r.get("tier", tier_label)) for r in results}

    for rho in rhos:
        for name, mid, teacher, fit, tokens in models:
            key = (name, rho, args.score, tier_label)
            if key in done:
                print(f"[skip] {name} rho={rho} already done")
                continue
            try:
                rec = run_one(name, mid, teacher, fit, tokens,
                              args.n, args.seq_len, args.device, rho,
                              score_mode=args.score, tier=tier_label)
                results.append(rec)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[int4-overlay] wrote {args.out} ({len(results)})")
            except Exception as e:
                print(f"[FAIL] {name} rho={rho}: {type(e).__name__}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache(); gc.collect()

    print("\n================ int4-overlay summary ================")
    print(f"{'model':<18} {'tier':<22} {'rho':>7} {'t1_ret':>7} {'ppl_r':>7} {'eff_bpw':>7}")
    for r in sorted(results, key=lambda x: (x["rho"], x["name"])):
        print(f"{r['name']:<18} {r.get('tier',''):<22} {r['rho']:>7.4f} "
              f"{r['t1_ret']*100:>6.2f}% {r['ppl_ratio']:>6.3f} "
              f"{r.get('effective_bpw',0):>7.4f}")


if __name__ == "__main__":
    main()
