"""
eval_sweep_ppl.py — PPL vs bpw sweep to establish whether v16 stack recovers
quality at higher bit budgets, and to test activation-aware correction.

Three probes:
  probe A : v16 at (K1=4096, K2=4096) uniform → ~3.0 bpw. Does PPL recover?
  probe B : v16 at (K1=8192, K2=4096) on o_proj, rest default. ~2.5 bpw.
  probe C : v16 at its default (baseline for this sweep, already saved).

This isolates whether the 2100× PPL at 2.396 bpw is a BIT BUDGET problem
or a METRIC problem (rel-W ≠ what PPL cares about).
"""
from __future__ import annotations
import argparse, gc, math, time
import torch
from compress_v16 import v16_compress, DEFAULT_ROLE_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    probes = {
        "v16_3.0bpw_uniform": {r: (4096, 4096) for r in DEFAULT_ROLE_K},   # ~3.00
        "v16_3.0bpw_asym":   {**{r: (4096, 512) for r in DEFAULT_ROLE_K},
                               "o_proj": (8192, 4096)},                    # ~2.72 total
    }
    for tag, role_K in probes.items():
        bpw_est = sum((math.log2(k1) + math.log2(k2)) / args.D
                      for k1, k2 in role_K.values()) / len(role_K)
        print(f"\n==== {tag}  (avg bpw~{bpw_est:.2f}) ====")
        t = time.time()
        r = v16_compress(args.teacher, role_K, args.D,
                         iters=args.iters, beam=args.beam, device=args.device)
        print(f"  final mean {r['rel_w_final_mean']:.4f}  "
              f"max {r['rel_w_final_max']:.4f}  "
              f"global_bpw {r['global_bpw']:.3f}  ({time.time()-t:.0f}s)")
        torch.save(r, f"{tag}.pt")
        del r; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
