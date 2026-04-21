"""lambada_overlay_adaptive.py -- Claim 18: globally-allocated sparse fp16 row-overlay.

Extension of Claim 17. Claim 17 restores the top ``rho*O`` rows *per tensor* to
fp16. Claim 18 instead computes the row-residual score for every row of every
body Linear, then restores the **global** top ``rho * sum_t O_t`` rows across
the whole body. At the same *total* overlay bpw, tensors with heavier
residual tails (measured by activation-weighted row energy) receive more
restored rows and tensors with flatter residuals receive fewer, producing a
strictly-dominated Pareto point over the uniform Claim-17 allocation.

The decode path is identical; only the row-selection rule changes. Any
Claim-16 fit (base 2.40, hifi 2.78, future tier) works unchanged.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign
from lambada_overlay import MODELS, _reconstruct_v17_with_overlay  # re-use decode


def _decode_and_score(W_fp16, role, bank, s_col, D, rot, device, score_mode="weighted"):
    """Return (Wq_fp32_cpu, score_cpu) without applying any overlay."""
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
    score = (diff * diff).sum(1)  # [O]
    del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, s, raw, diff
    return Wq.detach().to("cpu"), score.detach().to("cpu")


def substitute_v17_overlay_adaptive(model, state_dict, v17, device, D, rho,
                                     score_mode="weighted", clip=(0.25, 4.0)):
    """Two-pass global-topK overlay. Same total rows as uniform rho, but
    allocated per-tensor by residual-energy rank.

    clip: (lo, hi) multipliers on rho to prevent any tensor from receiving
          zero rows or consuming the whole budget. Per-tensor K_t is clipped
          to [lo*rho*O_t, hi*rho*O_t] before the global threshold.
    """
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
          f"(v17+overlay-adaptive, rho={rho}, score={score_mode}, clip={clip})")

    # Pass 1: decode every tensor, collect per-row scores
    decoded = {}       # k -> Wq fp32 cpu
    scores = {}        # k -> score tensor cpu [O]
    O_per_k = {}
    I_per_k = {}
    t0 = time.time()
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = state_dict[k]
        s = s_col.get(k, torch.ones(W.shape[1]))
        Wq, sc = _decode_and_score(W, role, bank, s, D,
                                    rots[W.shape[1]], device, score_mode)
        decoded[k] = Wq
        scores[k] = sc
        O_per_k[k] = W.shape[0]
        I_per_k[k] = W.shape[1]
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    print(f"  pass-1 decode+score: {time.time()-t0:.1f}s")

    # Pass 2: global-topK with per-tensor clip.
    # Strategy: per-tensor budget K_t is found by one pass of priority selection
    # using each tensor's score-sorted queue. Equivalently: pick a global
    # threshold tau s.t. Σ_t clip(count(score_t >= tau), lo*rho*O_t, hi*rho*O_t)
    # equals rho * Σ O_t. We approximate via a direct greedy global sort with
    # per-tensor caps.
    total_O = sum(O_per_k.values())
    target = max(1, int(round(rho * total_O)))
    lo_frac, hi_frac = clip
    caps = {k: int(round(hi_frac * rho * O_per_k[k])) for k in hf_keys}
    mins = {k: int(round(lo_frac * rho * O_per_k[k])) for k in hf_keys}
    # pre-assign minimums
    chosen: dict[str, torch.Tensor] = {}
    remaining = target
    for k in hf_keys:
        m = min(mins[k], caps[k], O_per_k[k])
        if m > 0:
            idx = scores[k].topk(m).indices
            chosen[k] = idx
            remaining -= m
        else:
            chosen[k] = torch.empty(0, dtype=torch.long)
    # global competition for remaining budget, respecting per-tensor caps
    if remaining > 0:
        # build (score, k_idx, row_idx) for rows NOT already chosen
        pool_scores = []
        pool_tensor = []
        pool_row = []
        for ki, k in enumerate(hf_keys):
            already = set(chosen[k].tolist())
            cap_left = caps[k] - len(already)
            if cap_left <= 0:
                continue
            sc = scores[k]
            # mask out already-chosen rows, then take top (cap_left) candidates
            mask = torch.ones(sc.shape[0], dtype=torch.bool)
            if len(already):
                mask[torch.tensor(sorted(already), dtype=torch.long)] = False
            sc_masked = sc.clone()
            sc_masked[~mask] = float("-inf")
            top_k = min(cap_left, O_per_k[k] - len(already))
            if top_k <= 0:
                continue
            top = sc_masked.topk(top_k)
            pool_scores.append(top.values)
            pool_tensor.append(torch.full((top_k,), ki, dtype=torch.long))
            pool_row.append(top.indices)
        if pool_scores:
            all_s = torch.cat(pool_scores)
            all_t = torch.cat(pool_tensor)
            all_r = torch.cat(pool_row)
            take = min(remaining, all_s.shape[0])
            pick = all_s.topk(take).indices
            picked_t = all_t[pick]
            picked_r = all_r[pick]
            # add to chosen per tensor
            for ki, k in enumerate(hf_keys):
                mask_ki = picked_t == ki
                if mask_ki.any():
                    new_rows = picked_r[mask_ki]
                    chosen[k] = torch.cat([chosen[k], new_rows]).unique()

    # Pass 3: apply overlay, write weights
    total_restored = 0
    total_rows = 0
    total_params_restored = 0
    total_params = 0
    per_tensor_K = []
    for n, k in enumerate(hf_keys):
        Wq = decoded[k].to(device)
        W = state_dict[k].to(device=device, dtype=torch.float32)
        idx = chosen[k].to(device)
        if idx.numel() > 0:
            Wq[idx] = W[idx]
        total_restored += int(idx.numel())
        total_rows += O_per_k[k]
        total_params_restored += int(idx.numel()) * I_per_k[k]
        total_params += O_per_k[k] * I_per_k[k]
        per_tensor_K.append((k, int(idx.numel()), O_per_k[k]))
        W_new = Wq.to(dtype=torch.float16, device="cpu")
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del Wq, W, W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()

    torch.cuda.empty_cache(); gc.collect()
    print(f"  overlay-adaptive: restored {total_restored}/{total_rows} rows "
          f"({100*total_restored/max(total_rows,1):.3f}%), "
          f"{total_params_restored}/{total_params} params "
          f"({100*total_params_restored/max(total_params,1):.3f}%)")

    # distribution stats for patent evidence
    fracs = [(k, K, O) for (k, K, O) in per_tensor_K]
    ratios = [ (K / max(O, 1)) / rho for (_, K, O) in fracs]
    ratios_t = torch.tensor(ratios)
    print(f"  per-tensor rho multiplier: "
          f"min={ratios_t.min().item():.3f} "
          f"p25={ratios_t.quantile(0.25).item():.3f} "
          f"med={ratios_t.median().item():.3f} "
          f"p75={ratios_t.quantile(0.75).item():.3f} "
          f"max={ratios_t.max().item():.3f}")

    gbpw = float(v17.get("global_bpw", 2.78))
    if total_restored > 0:
        avg_I = total_params_restored / total_restored
        excess_bits = total_restored * ((16 - gbpw) * avg_I + 32)
    else:
        excess_bits = 0.0
    overlay_bpw = excess_bits / max(total_params, 1)
    print(f"  overlay bpw overhead: +{overlay_bpw:.4f} "
          f"(base {gbpw:.4f} -> effective {gbpw+overlay_bpw:.4f})")
    return {
        "n_restored_rows": total_restored,
        "n_total_rows": total_rows,
        "rows_pct": 100*total_restored/max(total_rows,1),
        "params_pct": 100*total_params_restored/max(total_params,1),
        "overlay_bpw": float(overlay_bpw),
        "base_bpw": gbpw,
        "effective_bpw": float(gbpw + overlay_bpw),
        "rho_mult_min": float(ratios_t.min()),
        "rho_mult_med": float(ratios_t.median()),
        "rho_mult_max": float(ratios_t.max()),
    }


def run_one(name, model_id, teacher_path, v17_path, tokens_path,
            n, seq_len, device, rho, score_mode="weighted",
            tier="hifi+overlay-adaptive"):
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
    ov_stats = substitute_v17_overlay_adaptive(model, sd, v17, device, D, rho, score_mode)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17+ov-adaptive rho={rho} score={score_mode}  PPL={v17_ppl:.4f}  "
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
    rec.update(ov_stats)
    del model, sd, toks
    torch.cuda.empty_cache(); gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rhos", default="0.002,0.005",
                    help="comma-separated rho values")
    ap.add_argument("--only", default="", help="comma-separated model shorts")
    ap.add_argument("--out", default="lambada_overlay_adaptive_results.json")
    ap.add_argument("--score", default="weighted", choices=["weighted", "unweighted"])
    ap.add_argument("--base", action="store_true", help="use 2.40 bpw base fits")
    args = ap.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    models = MODELS
    if args.base:
        models = [(n, mid, t, fit.replace("v17hi_fit_", "v17_fit_"), tok)
                  for (n, mid, t, fit, tok) in MODELS]
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    if only:
        def match(n):
            nl = n.lower()
            return any(k in nl for k in only)
        models = [m for m in models if match(m[0])]

    tier_label = "base+overlay-adaptive" if args.base else "hifi+overlay-adaptive"

    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out, "r"))
        except Exception:
            results = []
    done = {(r["name"], r["rho"], r.get("score_mode","weighted"),
             r.get("tier","hifi+overlay-adaptive")) for r in results}

    for rho in rhos:
        for name, mid, teacher, fit, tokens in models:
            key = (name, rho, args.score, tier_label)
            if key in done:
                print(f"[skip] {name} rho={rho} score={args.score} tier={tier_label}: already done")
                continue
            try:
                rec = run_one(name, mid, teacher, fit, tokens,
                              args.n, args.seq_len, args.device, rho,
                              score_mode=args.score, tier=tier_label)
                results.append(rec)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"[lambada_overlay_adaptive] wrote {args.out} ({len(results)})")
            except Exception as e:
                print(f"[FAIL] {name} rho={rho}: {type(e).__name__}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache(); gc.collect()

    print("\n================ overlay-adaptive summary ================")
    print(f"{'model':<18} {'tier':<24} {'score':<11} {'rho':>7} "
          f"{'t1_ret':>7} {'ppl_r':>7} {'eff_bpw':>7}")
    for r in sorted(results, key=lambda x: (x.get("tier",""), x["rho"],
                                             x.get("score_mode",""), x["name"])):
        print(f"{r['name']:<18} {r.get('tier',''):<24} "
              f"{r.get('score_mode','weighted'):<11} "
              f"{r['rho']:>7.4f} {r['t1_ret']*100:>6.2f}% "
              f"{r['ppl_ratio']:>6.3f} {r.get('effective_bpw',0):>7.4f}")


if __name__ == "__main__":
    main()
