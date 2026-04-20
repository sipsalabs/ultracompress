"""Standalone Claim-16 fit on arbitrary model size (8B and beyond).

Only runs v17_compress (no HF model) and saves banks+s_col. Use
eval_v17_8b.py or eval_claim16_topk.py in a separate process for PPL/T1/T10.
"""
from __future__ import annotations
import argparse, time
import torch

from compress_v17 import v17_compress
from compress_v16 import DEFAULT_ROLE_K


ATTN_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_ROLES  = ("gate_proj", "up_proj", "down_proj")


def make_dict(a_attn: float, a_mlp: float) -> dict:
    d = {}
    for r in ATTN_ROLES: d[r] = a_attn
    for r in MLP_ROLES:  d[r] = a_mlp
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher",  required=True)
    ap.add_argument("--v17act",   required=True)
    ap.add_argument("--a_attn",   type=float, default=0.25)
    ap.add_argument("--a_mlp",    type=float, default=0.125)
    ap.add_argument("--iters",    type=int, default=6)
    ap.add_argument("--beam",     type=int, default=8)
    ap.add_argument("--D",        type=int, default=8)
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument("--out",      required=True)
    args = ap.parse_args()

    alpha_per_role = make_dict(args.a_attn, args.a_mlp)
    print(f"[fit] alpha_per_role = {alpha_per_role}")
    t0 = time.time()
    r = v17_compress(args.teacher, args.v17act, DEFAULT_ROLE_K, args.D,
                     alpha=args.a_attn, alpha_per_role=alpha_per_role,
                     iters=args.iters, beam=args.beam, device=args.device)
    wall = time.time() - t0
    out = {
        "a_attn": args.a_attn, "a_mlp": args.a_mlp,
        "banks":       r["banks"],
        "s_col":       r["s_col"],
        "role_K":      r["role_K"],
        "D":           r["D"],
        "rel_w_mean":  r["rel_w_final_mean"],
        "rel_w_max":   r["rel_w_final_max"],
        "global_bpw":  r["global_bpw"],
        "overhead_bpw":r["overhead_bpw"],
        "total_bpw":   r["global_bpw"] + r["overhead_bpw"],
        "final_by_role": r["final_by_role"],
        "wall_sec":    wall,
    }
    torch.save(out, args.out)
    print(f"\n[fit] saved {args.out}")
    print(f"[fit] rel-W mean {out['rel_w_mean']:.4f}  max {out['rel_w_max']:.4f}  "
          f"bpw {out['total_bpw']:.4f}  wall {wall:.0f}s")


if __name__ == "__main__":
    main()
