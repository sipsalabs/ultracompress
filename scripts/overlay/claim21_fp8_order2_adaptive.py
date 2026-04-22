"""claim21_fp8_order2_adaptive.py -- wave 33.

Wave 31 established the order-2 Shannon floor H(B_i | B_{i-1}, B_{i-2})
= 6.40 bpB cohort. That's the information-theoretic lower bound. This
wave measures what a realistic ADAPTIVE order-2 arithmetic coder
actually achieves on the same fp8 streams, with Laplace-1 priors that
update online. That converts the -0.155 bpB theoretical gain vs
brotli-11 into a deployable-rate estimate.

Coder model: per 2-byte context (256^2 = 65536 states), maintain a
256-symbol count vector initialized to all ones (Laplace-1). For each
incoming byte, charge -log2((count[ctx][byte] + 0) / sum(count[ctx]))
and then increment count[ctx][byte]. The leading 2 bytes are charged
at lower orders (order 0 and order 1 Laplace-1) for fairness.

The reported rate is total bits / total bytes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from claim21_fp8_order2 import build_fp8_bytes


def run_adaptive_order2(arr: np.ndarray) -> dict:
    n = int(arr.size)
    counts2 = np.ones((65536, 256), dtype=np.int64)
    sums2 = np.full(65536, 256, dtype=np.int64)
    counts1 = np.ones((256, 256), dtype=np.int64)
    sums1 = np.full(256, 256, dtype=np.int64)
    counts0 = np.ones(256, dtype=np.int64)
    sum0 = 256

    total_bits = 0.0
    log2 = math.log2

    b = int(arr[0])
    total_bits += -log2(counts0[b] / sum0)
    counts0[b] += 1
    sum0 += 1

    if n >= 2:
        c = int(arr[0])
        b = int(arr[1])
        total_bits += -log2(counts1[c, b] / sums1[c])
        counts1[c, b] += 1
        sums1[c] += 1

    if n >= 3:
        a0 = arr[:-2].astype(np.uint32)
        a1 = arr[1:-1].astype(np.uint32)
        ctx = ((a0 << 8) | a1).astype(np.int64)
        tgt = arr[2:].astype(np.int64)

        # Pull raw buffers for fastest Python access
        ctx_list = ctx.tolist()
        tgt_list = tgt.tolist()
        m = len(ctx_list)
        # Using nested numpy arrays with item access is fastest simple path
        for i in range(m):
            c = ctx_list[i]
            t = tgt_list[i]
            cnt = counts2[c, t]
            s = sums2[c]
            total_bits += -log2(cnt / s)
            counts2[c, t] = cnt + 1
            sums2[c] = s + 1

    return dict(
        n_bytes=n,
        total_bits=total_bits,
        adaptive_order2_bpB=total_bits / n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2-adaptive] model={args.model}  rho={args.rho}")
    fp8_bytes = build_fp8_bytes(args.model, args.rho, args.device)
    arr = np.frombuffer(fp8_bytes, dtype=np.uint8)
    print(f"  n_fp8_bytes={arr.size:,}")
    res = run_adaptive_order2(arr)
    print(f"  adaptive order-2 rate = {res['adaptive_order2_bpB']:.4f} bpB")

    out = dict(
        claim=21,
        experiment="fp8_order2_adaptive",
        model=args.model,
        rho=args.rho,
        **res,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
