"""
per_role_alpha_sweep.py — Claim 16: Per-role alpha damping.

Motivation. Measured INPUT-variance max/mean ratios differ sharply
across role families:
    attention (q/k/v/o):  ~120x
    MLP (gate/up/down) :  ~290x
A single global alpha (Claim 14) is therefore a compromise. This script
sweeps independent alpha_attn, alpha_mlp pairs at fixed bpw (no extra
overhead: the per-role alpha itself adds 7 fp16 scalars = 14 bytes).

Configs evaluated:
    (attn=0.125, mlp=0.125)   - v17 baseline (sanity)
    (attn=0.125, mlp=0.25)    - stronger MLP damping
    (attn=0.125, mlp=0.0625)  - weaker MLP damping
    (attn=0.0625, mlp=0.125)  - weaker attn damping
    (attn=0.25,  mlp=0.125)   - stronger attn damping
"""
from __future__ import annotations
import argparse, gc, time
import torch

from compress_v17 import v17_compress
from compress_v16 import DEFAULT_ROLE_K
from eval_v17_ppl import substitute_v17
from eval_v16_ppl import measure_ppl, reset_teacher


ATTN_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_ROLES  = ("gate_proj", "up_proj", "down_proj")


def make_dict(a_attn: float, a_mlp: float) -> dict[str, float]:
    d = {}
    for r in ATTN_ROLES:
        d[r] = a_attn
    for r in MLP_ROLES:
        d[r] = a_mlp
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--v17act", default="v17_activations.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["0.125,0.125", "0.125,0.25", "0.125,0.0625",
                             "0.0625,0.125", "0.25,0.125"],
                    help="Each entry 'a_attn,a_mlp'")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="per_role_alpha_results.pt")
    args = ap.parse_args()
    device = args.device

    print("[sweep] loading teacher")
    teacher_sd = torch.load(args.teacher, map_location="cpu",
                            weights_only=False)
    if "state_dict" in teacher_sd:
        teacher_sd = teacher_sd["state_dict"]

    print("[sweep] loading tokens")
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    print("[sweep] building Qwen3-1.7B")
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
    for cfg_str in args.configs:
        a_attn, a_mlp = [float(x) for x in cfg_str.split(",")]
        alpha_per_role = make_dict(a_attn, a_mlp)
        print(f"\n[sweep] ======= attn={a_attn}  mlp={a_mlp} =======")
        t0 = time.time()
        r = v17_compress(args.teacher, args.v17act, DEFAULT_ROLE_K, args.D,
                         alpha=a_attn,  # fallback (unused)
                         alpha_per_role=alpha_per_role,
                         iters=args.iters, beam=args.beam,
                         device=device)
        wall_fit = time.time() - t0
        gc.collect(); torch.cuda.empty_cache()

        v17 = {"banks": r["banks"], "s_col": r["s_col"],
               "alpha": a_attn,
               "alpha_per_role": alpha_per_role,
               "global_bpw": r["global_bpw"],
               "overhead_bpw": r["overhead_bpw"]}
        reset_teacher(model, teacher_sd)
        t1 = time.time()
        substitute_v17(model, teacher_sd, v17, device, args.D)
        ppl, _ = measure_ppl(model, all_tokens, starts, args.seq_len, device)
        wall_eval = time.time() - t1
        reset_teacher(model, teacher_sd)

        row = {
            "a_attn": a_attn, "a_mlp": a_mlp,
            "rel_w_mean": r["rel_w_final_mean"],
            "rel_w_max": r["rel_w_final_max"],
            "global_bpw": r["global_bpw"],
            "overhead_bpw": r["overhead_bpw"],
            "total_bpw": r["global_bpw"] + r["overhead_bpw"],
            "ppl": ppl, "ppl_ratio": ppl / ppl_base,
            "wall_fit_s": wall_fit, "wall_eval_s": wall_eval,
            "final_by_role": r["final_by_role"],
        }
        rows.append(row)
        print(f"[sweep] attn={a_attn} mlp={a_mlp}  "
              f"rel-W {row['rel_w_mean']:.4f}/{row['rel_w_max']:.4f}  "
              f"bpw {row['total_bpw']:.4f}  "
              f"PPL {ppl:.2f}  ratio {ppl/ppl_base:.3f}x")

    print("\n" + "=" * 72)
    print(f"PER-ROLE ALPHA SWEEP (baseline {ppl_base:.2f})")
    print("=" * 72)
    print(f"  a_attn  a_mlp    bpw  relW_m  relW_M        PPL   ratio")
    for r in rows:
        print(f"  {r['a_attn']:>6.3f}  {r['a_mlp']:>5.3f}  "
              f"{r['total_bpw']:.4f}  "
              f"{r['rel_w_mean']:.4f}  {r['rel_w_max']:.4f}  "
              f"{r['ppl']:>9.2f}  {r['ppl_ratio']:>6.3f}")

    torch.save({"ppl_base": ppl_base, "rows": rows, "args": vars(args)},
               args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
