"""claim21_fp8_order2_universal.py -- wave 34.

Wave 33 showed that a zero-initialized adaptive Laplace-1 order-2
coder is 0.50 bpB worse than brotli-11. Wave 34 tests the constructive
alternative flagged in the wave-33 conclusion: carry pre-trained
universal order-2 context tables derived from OTHER models, then code
the held-out model's fp8 stream statically.

This directly exploits wave 26's finding that order-0 fp8 histograms
correlate r > 0.9995 across unrelated models; here we test the
stronger claim at the order-2 joint level.

Protocol (leave-one-out cross-model):
  for each test model T:
    build training counts = sum over all OTHER models M of the
      order-2 triples in M's fp8 stream, plus Laplace-1
    code T's fp8 stream statically using these counts
    rate = sum of -log2(count[ctx][byte] / sum(count[ctx])) / N(T)

Also emits the "oracle" rate using T's own counts as a sanity check
(must match wave-31 H2 + tiny Laplace overhead).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from claim21_fp8_order2 import build_fp8_bytes

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def build_order2_counts(arr: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Return 65536 x 256 float count table with Laplace-alpha prior."""
    counts = np.full((65536, 256), alpha, dtype=np.float64)
    if arr.size >= 3:
        a0 = arr[:-2].astype(np.int64)
        a1 = arr[1:-1].astype(np.int64)
        a2 = arr[2:].astype(np.int64)
        ctx = (a0 << 8) | a1
        flat = ctx * 256 + a2
        uniq, c = np.unique(flat, return_counts=True)
        counts.reshape(-1)[uniq] += c.astype(np.float64)
    return counts


def static_code_rate(test_arr: np.ndarray, counts: np.ndarray) -> float:
    """Static order-2 coding rate for test_arr using counts table."""
    n = int(test_arr.size)
    if n < 3:
        return 0.0
    a0 = test_arr[:-2].astype(np.int64)
    a1 = test_arr[1:-1].astype(np.int64)
    a2 = test_arr[2:].astype(np.int64)
    ctx = (a0 << 8) | a1
    sums = counts.sum(axis=1)
    sel = counts[ctx, a2]
    denom = sums[ctx]
    bits = -np.log2(sel / denom).sum()
    bits += 2 * 8.0
    return float(bits) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2-universal] rho={args.rho}")
    streams = {}
    for m in MODELS:
        print(f"  loading {m} ...")
        arr = np.frombuffer(build_fp8_bytes(m, args.rho, args.device), dtype=np.uint8)
        streams[m] = arr
        print(f"    n_fp8_bytes={arr.size:,}")

    ALPHAS = [1.0, 0.5, 0.1, 0.01]
    rows = []
    for test in MODELS:
        others = [m for m in MODELS if m != test]
        # Build triples counts (no prior) from test and from universe
        test_cnt_raw = build_order2_counts(streams[test], alpha=0.0)
        univ_cnt_raw = np.zeros((65536, 256), dtype=np.float64)
        for m in others:
            univ_cnt_raw += build_order2_counts(streams[m], alpha=0.0)

        test_arr = streams[test]
        result = dict(test_model=test, n_bytes=int(test_arr.size), by_alpha={})
        for alpha in ALPHAS:
            oracle = static_code_rate(test_arr, test_cnt_raw + alpha)
            universal = static_code_rate(test_arr, univ_cnt_raw + alpha)
            result["by_alpha"][f"{alpha}"] = dict(
                oracle_static_bpB=oracle,
                universal_static_bpB=universal,
            )
            print(f"  test={test:<13}  a={alpha:<5}  oracle={oracle:.4f}  universal={universal:.4f}")
        rows.append(result)

    out = dict(
        claim=21,
        experiment="fp8_order2_universal",
        rho=args.rho,
        models=rows,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
