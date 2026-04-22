"""
alpha_sweep.py — Finalize Claim 14 by measuring v17 across
alpha in {0.0, 0.125, 0.25, 0.375, 0.5}.

alpha=0 reduces to v16 (identity scaling). We re-fit anyway so the
comparison isolates exactly the fold-before-rotation mechanism.
Same 500 windows, seq_len=128, seed=42 for all.

Output: alpha_sweep_results.pt with list of (alpha, rel_w_mean,
rel_w_max, global_bpw, overhead_bpw, ppl, ppl_ratio, wall_fit_s,
wall_eval_s).
"""
from __future__ import annotations
import argparse, gc, math, time
import torch

from compress_v17 import v17_compress
from compress_v16 import DEFAULT_ROLE_K
from eval_v17_ppl import substitute_v17
from eval_v16_ppl import measure_ppl, reset_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--v17act", default="v17_activations.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.125, 0.25, 0.375, 0.5])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="alpha_sweep_results.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[sweep] loading teacher")
    teacher_sd = torch.load(args.teacher, map_location="cpu",
                            weights_only=False)
    if "state_dict" in teacher_sd:
        teacher_sd = teacher_sd["state_dict"]

    print(f"[sweep] loading tokens")
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    print(f"[sweep] building Qwen3-1.7B")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    print("\n[sweep] === BASELINE ===")
    ppl_base, _ = measure_ppl(model, all_tokens, starts, args.seq_len, device)
    print(f"[sweep] baseline PPL = {ppl_base:.4f}")

    rows = []
    for alpha in args.alphas:
        print(f"\n[sweep] ========== alpha = {alpha} ==========")
        t0 = time.time()
        r = v17_compress(args.teacher, args.v17act, DEFAULT_ROLE_K, args.D,
                         alpha=alpha, iters=args.iters, beam=args.beam,
                         device=device)
        wall_fit = time.time() - t0
        gc.collect(); torch.cuda.empty_cache()

        # wrap as v17-compatible dict for substitute_v17
        v17 = {"banks": r["banks"], "s_col": r["s_col"],
               "alpha": alpha, "global_bpw": r["global_bpw"],
               "overhead_bpw": r["overhead_bpw"]}
        t1 = time.time()
        substitute_v17(model, teacher_sd, v17, device, args.D)
        ppl, _ = measure_ppl(model, all_tokens, starts, args.seq_len, device)
        wall_eval = time.time() - t1
        ratio = ppl / ppl_base
        reset_teacher(model, teacher_sd)
        torch.cuda.empty_cache(); gc.collect()

        row = {"alpha": alpha,
               "rel_w_mean": r["rel_w_final_mean"],
               "rel_w_max": r["rel_w_final_max"],
               "global_bpw": r["global_bpw"],
               "overhead_bpw": r["overhead_bpw"],
               "total_bpw": r["global_bpw"] + r["overhead_bpw"],
               "ppl": ppl, "ppl_ratio": ratio,
               "wall_fit_s": wall_fit, "wall_eval_s": wall_eval}
        rows.append(row)
        print(f"\n[sweep] alpha={alpha}  rel-W {r['rel_w_final_mean']:.4f} "
              f"/ {r['rel_w_final_max']:.4f}  bpw {row['total_bpw']:.4f}  "
              f"PPL {ppl:.2f}  ratio {ratio:.3f}×")
        del v17, r

    print("\n" + "=" * 72)
    print(f"ALPHA SWEEP  (n={args.n}, seq_len={args.seq_len}, "
          f"baseline PPL {ppl_base:.2f})")
    print("=" * 72)
    print(f"{'alpha':>6s}  {'bpw':>6s}  {'relW_m':>7s}  {'relW_M':>7s}  "
          f"{'PPL':>10s}  {'ratio':>8s}")
    for row in rows:
        print(f"{row['alpha']:>6.3f}  {row['total_bpw']:>6.4f}  "
              f"{row['rel_w_mean']:>7.4f}  {row['rel_w_max']:>7.4f}  "
              f"{row['ppl']:>10.2f}  {row['ppl_ratio']:>8.3f}")

    out = {"baseline_ppl": ppl_base, "rows": rows,
           "n": args.n, "seq_len": args.seq_len}
    torch.save(out, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
