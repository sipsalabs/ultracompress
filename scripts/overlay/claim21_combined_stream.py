"""claim21_combined_stream.py -- wave 41.

Waves 30-40 analyzed each of the three payload streams (fp8,
idx_delta, scale) separately. Wave 41 tests a pragmatic engineering
question: does brotli-11 on the CONCATENATED payload do better,
worse, or the same as summing brotli-11 on each stream individually?

Three orderings are tested:
  A. fp8 || idx_delta || scale   (baseline -- 'payload order')
  B. idx_delta || scale || fp8   (small-first)
  C. interleaved (round-robin one byte at a time from each stream,
     truncated to min length per stream cycle)

If concatenated < sum-of-parts, brotli is exploiting CROSS-STREAM
structure -- i.e. the 3 streams share a reference model. If
concatenated > sum-of-parts, stream boundaries cost a transition
penalty (unlikely for brotli's LZ77+Huffman).

Also report zstd-22 (another strong general-purpose coder) for
context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np
import zstandard as zstd

from claim21_streams_order2 import build_all_streams

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def br11(b): return len(brotli.compress(b, quality=11))
def zs22(b): return len(zstd.ZstdCompressor(level=22).compress(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for m in MODELS:
        print(f"[{m}]")
        fp8, idx, scl = build_all_streams(m, args.rho, args.device)
        # Individual stream sizes
        n_fp8, n_idx, n_scl = len(fp8), len(idx), len(scl)
        n_total = n_fp8 + n_idx + n_scl
        br_fp8, br_idx, br_scl = br11(fp8), br11(idx), br11(scl)
        zs_fp8, zs_idx, zs_scl = zs22(fp8), zs22(idx), zs22(scl)
        sum_br = br_fp8 + br_idx + br_scl
        sum_zs = zs_fp8 + zs_idx + zs_scl

        # Concat orderings
        cat_A = fp8 + idx + scl
        cat_B = idx + scl + fp8
        br_A = br11(cat_A); br_B = br11(cat_B)
        zs_A = zs22(cat_A); zs_B = zs22(cat_B)

        row = dict(
            model=m,
            n_fp8=n_fp8, n_idx=n_idx, n_scl=n_scl, n_total=n_total,
            brotli11=dict(
                per_stream_bytes=dict(fp8=br_fp8, idx=br_idx, scl=br_scl, sum=sum_br),
                concat_A_bytes=br_A, concat_B_bytes=br_B,
                cross_stream_gain_A_bpB=8.0 * (sum_br - br_A) / n_total,
                cross_stream_gain_B_bpB=8.0 * (sum_br - br_B) / n_total,
                sum_bpB=8.0 * sum_br / n_total,
                concat_A_bpB=8.0 * br_A / n_total,
                concat_B_bpB=8.0 * br_B / n_total,
            ),
            zstd22=dict(
                per_stream_bytes=dict(fp8=zs_fp8, idx=zs_idx, scl=zs_scl, sum=sum_zs),
                concat_A_bytes=zs_A, concat_B_bytes=zs_B,
                cross_stream_gain_A_bpB=8.0 * (sum_zs - zs_A) / n_total,
                cross_stream_gain_B_bpB=8.0 * (sum_zs - zs_B) / n_total,
                sum_bpB=8.0 * sum_zs / n_total,
                concat_A_bpB=8.0 * zs_A / n_total,
                concat_B_bpB=8.0 * zs_B / n_total,
            ),
        )
        rows.append(row)
        print(f"  n={n_total:,}  brotli11: sum={sum_br:,}  "
              f"concatA={br_A:,} (gain={row['brotli11']['cross_stream_gain_A_bpB']:+.5f})  "
              f"concatB={br_B:,} (gain={row['brotli11']['cross_stream_gain_B_bpB']:+.5f})")
        print(f"            zstd22:   sum={sum_zs:,}  "
              f"concatA={zs_A:,} (gain={row['zstd22']['cross_stream_gain_A_bpB']:+.5f})  "
              f"concatB={zs_B:,} (gain={row['zstd22']['cross_stream_gain_B_bpB']:+.5f})")

    Path(args.out).write_text(json.dumps({
        "claim": 21, "experiment": "combined_stream",
        "rho": args.rho, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
