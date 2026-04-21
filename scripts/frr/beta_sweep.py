"""beta_sweep.py — fix alpha=0.125, sweep beta to find Claim 15 optimum."""
from __future__ import annotations
import argparse, gc, time
import torch
from compress_v18 import v18_compress
from compress_v16 import DEFAULT_ROLE_K
from eval_v18_ppl import substitute_v18
from eval_v16_ppl import measure_ppl, reset_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--actio", default="v18_activations_io.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--alpha", type=float, default=0.125)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.03125, 0.0625, 0.125, 0.25])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="beta_sweep_results.pt")
    args = ap.parse_args()
    device = args.device

    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd: teacher_sd = teacher_sd["state_dict"]
    tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    print("[sweep] baseline")
    ppl_b, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
    print(f"  PPL {ppl_b:.4f}")

    rows = []
    for beta in args.betas:
        print(f"\n[sweep] === alpha={args.alpha}  beta={beta} ===")
        t0 = time.time()
        r = v18_compress(args.teacher, args.actio, DEFAULT_ROLE_K, args.D,
                         alpha=args.alpha, beta=beta,
                         iters=args.iters, beam=args.beam, device=device)
        wall_fit = time.time() - t0
        v18 = {"banks": r["banks"], "s_col": r["s_col"], "u_row": r["u_row"],
               "alpha": args.alpha, "beta": beta,
               "global_bpw": r["global_bpw"], "overhead_bpw": r["overhead_bpw"]}
        substitute_v18(model, teacher_sd, v18, device, args.D)
        ppl, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
        ratio = ppl / ppl_b
        reset_teacher(model, teacher_sd)
        torch.cuda.empty_cache(); gc.collect()
        row = {"alpha": args.alpha, "beta": beta,
               "rel_w_mean": r["rel_w_final_mean"],
               "rel_w_max": r["rel_w_final_max"],
               "total_bpw": r["global_bpw"] + r["overhead_bpw"],
               "ppl": ppl, "ppl_ratio": ratio, "wall_fit_s": wall_fit}
        rows.append(row)
        print(f"[sweep] beta={beta}  rel-W {r['rel_w_final_mean']:.4f} "
              f"PPL {ppl:.2f}  ratio {ratio:.3f}x")
        del v18, r

    print("\n" + "="*70)
    print(f"BETA SWEEP (alpha={args.alpha}, baseline {ppl_b:.2f})")
    print("="*70)
    print(f"{'beta':>8s} {'bpw':>7s} {'relW_m':>7s} {'relW_M':>7s} "
          f"{'PPL':>10s} {'ratio':>7s}")
    for row in rows:
        print(f"{row['beta']:>8.4f} {row['total_bpw']:>7.4f} "
              f"{row['rel_w_mean']:>7.4f} {row['rel_w_max']:>7.4f} "
              f"{row['ppl']:>10.2f} {row['ppl_ratio']:>7.3f}")
    torch.save({"baseline_ppl": ppl_b, "rows": rows,
                "alpha": args.alpha, "n": args.n, "seq_len": args.seq_len},
               args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
