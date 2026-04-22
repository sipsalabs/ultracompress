"""claim21_fp8_order2_amortized.py -- wave 37.

Wave 36 established that the one-shot net cost of shipping order-2
priors is +0.63 bpB WORSE than brotli-11, and argued THEORETICALLY
that K>=5 reuses of the same priors would make order-2 coding net-
positive. Wave 37 tests this empirically.

Simulation: the priors are shipped ONCE (we already paid that cost
at wave 36), and then future payloads are coded using those priors.
We approximate a "future payload" by holding out part of the payload
stream from prior-fitting.

For each cohort model at rho=0.010:
  1. Split fp8 bytes into a prior-fit split and a held-out split.
     We test prior_frac in {0.25, 0.50, 0.75}, where prior_frac is
     the fraction used to FIT priors; the remainder is the held-out
     test payload.
  2. Build order-2 counts on the prior-fit split with alpha in
     {0.1, 0.01, 0.001}.
  3. Code the held-out split with those counts (no adaptation).
  4. Compare held-out coding rate to brotli-11 rate on the same
     held-out split (we recompute brotli-11 directly on the bytes).
  5. Compute amortized_gain = br_rate - order2_rate ON THE HELD-OUT
     PAYLOAD ONLY (no side-info charged -- it amortizes over many
     future payloads).

This is the single-reuse (K=1 among K>=1 reuses) performance; by
linearity, K reuses divide the side-info cost by K, so wave 37 tells
us the ASYMPTOTIC (K->inf) amortized rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np

from claim21_fp8_order2 import build_fp8_bytes
from claim21_fp8_order2_universal import build_order2_counts, static_code_rate

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
PRIOR_FRACTIONS = [0.25, 0.50, 0.75]
ALPHAS = [0.1, 0.01, 0.001]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2-amortized] rho={args.rho}")
    rows = []
    for m in MODELS:
        print(f"  [{m}]")
        arr = np.frombuffer(build_fp8_bytes(m, args.rho, args.device), dtype=np.uint8)
        n = int(arr.size)
        print(f"    n={n:,}")
        for pf in PRIOR_FRACTIONS:
            cut = int(n * pf)
            prior_arr = arr[:cut]
            test_arr = arr[cut:]
            n_test = int(test_arr.size)
            # brotli-11 baseline on held-out bytes only
            br_size = len(brotli.compress(test_arr.tobytes(), quality=11))
            br_rate = 8.0 * br_size / n_test

            for a in ALPHAS:
                counts = build_order2_counts(prior_arr, alpha=a)
                o2_rate = static_code_rate(test_arr, counts)
                gain = br_rate - o2_rate
                rows.append(dict(
                    model=m, prior_frac=pf, alpha=a,
                    n_prior=int(prior_arr.size), n_test=n_test,
                    brotli11_heldout_bpB=br_rate,
                    order2_heldout_bpB=o2_rate,
                    amortized_gain_vs_brotli11_bpB=gain,
                ))
                print(f"    prior={pf:.2f} alpha={a:<6}  "
                      f"order2={o2_rate:.4f}  brotli11={br_rate:.4f}  "
                      f"gain={gain:+.4f}")

    out = dict(
        claim=21,
        experiment="fp8_order2_amortized",
        rho=args.rho,
        rows=rows,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
