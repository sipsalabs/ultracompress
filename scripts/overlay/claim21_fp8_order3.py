"""claim21_fp8_order3.py -- wave 32 sparse order-3 probe.

Wave 31 established that the fp8 stream has a constructive sub-brotli
path at order 2: H(B_i | B_{i-1}, B_{i-2}) sits 0.155 bpB below the
shipping brotli-11 rate on the 4-model cohort. The next local question
is whether brotli's remaining gap is mostly order >= 3 structure or
whether order-2 already captures nearly all useful context.

This script measures the order-3 floor directly:

    H(B_i | B_{i-1}, B_{i-2}, B_{i-3})

Unlike wave 31's dense 256^3 histogram, the order-3 state space is
256^4, so we count only observed 4-grams and 3-gram contexts using
sparse unique-value histograms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from claim21_fp8_order2 import build_fp8_bytes


def H_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


def miller_madow(H_plugin: float, k_observed: int, n: int) -> float:
    """Miller-Madow bias-corrected entropy: H_MM = H_plugin + (K-1)/(2N ln 2)."""
    if n <= 0:
        return H_plugin
    return H_plugin + (k_observed - 1) / (2.0 * n * np.log(2.0))


def sparse_counts(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, counts = np.unique(values, return_counts=True)
    return counts.astype(np.int64), int(counts.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order3] model={args.model}  rho={args.rho}")
    fp8_bytes = build_fp8_bytes(args.model, args.rho, args.device)
    arr = np.frombuffer(fp8_bytes, dtype=np.uint8)
    n = int(arr.size)
    if n < 4:
        raise ValueError("need at least 4 fp8 bytes for order-3 measurement")
    print(f"  n_fp8_bytes={n:,}")

    a0 = arr[:-3].astype(np.uint32)
    a1 = arr[1:-2].astype(np.uint32)
    a2 = arr[2:-1].astype(np.uint32)
    a3 = arr[3:].astype(np.uint32)

    quad = (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
    ctx3 = (a0 << 16) | (a1 << 8) | a2

    quad_counts, n_quad_states = sparse_counts(quad)
    ctx3_counts, n_ctx3_states = sparse_counts(ctx3)

    H4 = H_from_counts(quad_counts)
    H3 = H_from_counts(ctx3_counts)
    H_cond3 = H4 - H3

    H4_mm = miller_madow(H4, n_quad_states, int(quad_counts.sum()))
    H3_mm = miller_madow(H3, n_ctx3_states, int(ctx3_counts.sum()))
    H_cond3_mm = H4_mm - H3_mm

    # Sample-size diagnostic: fraction of 4-grams observed only once.
    quad_singleton_frac = float((quad_counts == 1).sum()) / float(quad_counts.size)

    print(f"  observed 4-gram states         = {n_quad_states:,}")
    print(f"  observed 3-gram contexts       = {n_ctx3_states:,}")
    print(f"  4-gram singleton fraction      = {quad_singleton_frac:.4f}")
    print(f"  H(B_i, ..., B_{{i-3}})         = {H4:.4f} bits / 4 bytes  (MM {H4_mm:.4f})")
    print(f"  H(B_{{i-1}}, ..., B_{{i-3}})   = {H3:.4f} bits / 3 bytes  (MM {H3_mm:.4f})")
    print(f"  order-3 H (plug-in)            = {H_cond3:.4f} bpB")
    print(f"  order-3 H (Miller-Madow)       = {H_cond3_mm:.4f} bpB")

    out = {
        "claim": 21,
        "experiment": "fp8_order3",
        "model": args.model,
        "rho": args.rho,
        "n_fp8_bytes": n,
        "observed_quad_states": n_quad_states,
        "observed_ctx3_states": n_ctx3_states,
        "quad_singleton_frac": quad_singleton_frac,
        "H_joint4_plugin": H4,
        "H_joint3_plugin": H3,
        "H_joint4_mm": H4_mm,
        "H_joint3_mm": H3_mm,
        "order3_H_bpB_plugin": H_cond3,
        "order3_H_bpB_mm": H_cond3_mm,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
