"""claim21_fp8_order2_bootstrap.py -- wave 35.

Wave 34 established that oracle (test-model-specific) order-2 tables
recover 94% of the wave-31 theoretical sub-brotli gain, while universal
(cross-model) priors fail. Wave 35 tests the realistic deployable
shipping path: self-bootstrap. The encoder/decoder uses the first F
fraction of the stream (coded with a cheaper bootstrap coder) to learn
an order-2 count table, then statically codes the remaining (1-F) with
that learned table.

For this bitrate accounting we report:
  - bootstrap portion: charged at Shannon H_0 (optimistic floor) and
    separately at brotli-11's measured bpB on the full stream
    (realistic bootstrap coder)
  - tail portion:      charged via static Laplace-0.1 / 0.01 code rate
    using counts from the bootstrap portion
  - combined:          F * bootstrap_bpB + (1-F) * tail_bpB

Sweep F in {0.05, 0.10, 0.25, 0.50, 0.75} and alpha in {0.01, 0.1}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from claim21_fp8_order2 import build_fp8_bytes
from claim21_fp8_order2_universal import build_order2_counts, static_code_rate


MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75]
ALPHAS = [0.1, 0.01]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2-bootstrap] rho={args.rho}")
    streams = {}
    for m in MODELS:
        print(f"  loading {m} ...")
        arr = np.frombuffer(build_fp8_bytes(m, args.rho, args.device), dtype=np.uint8)
        streams[m] = arr
        print(f"    n_fp8_bytes={arr.size:,}")

    rows = []
    for m in MODELS:
        arr = streams[m]
        n = int(arr.size)
        _, cnt0 = np.unique(arr, return_counts=True)
        p0 = cnt0 / cnt0.sum()
        H0 = float(-(p0 * np.log2(p0)).sum())
        row = dict(model=m, n_bytes=n, H0_bpB=H0, bootstraps=[], interleaved=[])
        # Contiguous-head bootstrap
        for F in FRACTIONS:
            k = max(3, int(F * n))
            head = arr[:k]
            tail = arr[k:]
            entry = dict(fraction=F, n_head=k, n_tail=int(tail.size), by_alpha={})
            head_counts = build_order2_counts(head, alpha=0.0)
            for alpha in ALPHAS:
                counts = head_counts + alpha
                tail_rate = static_code_rate(tail, counts)
                entry["by_alpha"][f"{alpha}"] = dict(tail_bpB=tail_rate)
            row["bootstraps"].append(entry)
        # Interleaved bootstrap: every 1/F-th byte is "head"
        for F in FRACTIONS:
            step = max(2, int(round(1.0 / F)))
            idx = np.arange(0, n, step)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            head = arr[mask]
            tail = arr[~mask]
            entry = dict(fraction=F, step=int(step), n_head=int(head.size),
                         n_tail=int(tail.size), by_alpha={})
            head_counts = build_order2_counts(head, alpha=0.0)
            for alpha in ALPHAS:
                counts = head_counts + alpha
                tail_rate = static_code_rate(tail, counts)
                entry["by_alpha"][f"{alpha}"] = dict(tail_bpB=tail_rate)
            row["interleaved"].append(entry)
        print(
            f"  {m:<14} contig F=0.5 a=0.01: "
            f"{row['bootstraps'][3]['by_alpha']['0.01']['tail_bpB']:.4f}  "
            f"interleaved F=0.5 a=0.01: "
            f"{row['interleaved'][3]['by_alpha']['0.01']['tail_bpB']:.4f}"
        )
        rows.append(row)

    out = dict(
        claim=21,
        experiment="fp8_order2_bootstrap",
        rho=args.rho,
        fractions=FRACTIONS,
        alphas=ALPHAS,
        models=rows,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
