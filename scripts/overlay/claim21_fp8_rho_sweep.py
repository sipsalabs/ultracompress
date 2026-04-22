"""claim21_fp8_rho_sweep.py -- wave 39.

Waves 30-38 analyzed the fp8 stream only at rho=0.010. Wave 39 sweeps
the rho axis to test whether the Shannon-floor-vs-brotli-11 gap is
stable across operating points.

For each model at rho in {0.005, 0.010, 0.020}, compute H0, H1 MM,
H2 MM, and brotli-11 on the fp8 stream. Report per-(model, rho)
gaps.

Rho=0.005 gives ~5 M bytes/model (weaker sample for H2)
Rho=0.010 is the baseline (~10-16 M, used in waves 30-38)
Rho=0.020 gives ~20-32 M (stronger sample for H2)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np

from claim21_fp8_order2 import build_fp8_bytes, H_from_counts
from claim21_streams_order2 import miller_madow

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
RHOS = [0.005, 0.010, 0.020]


def analyze_fp8(arr):
    n = arr.size
    c0 = np.bincount(arr, minlength=256).astype(np.float64)
    h0 = H_from_counts(c0)
    # H_joint2 via unique pairs
    pair = (arr[:-1].astype(np.int64) << 8) | arr[1:].astype(np.int64)
    u2, c2 = np.unique(pair, return_counts=True)
    h_joint2 = H_from_counts(c2.astype(np.float64))
    h_joint2_mm = miller_madow(h_joint2, len(u2), n - 1)
    h1_mm = h_joint2_mm - h0
    # H_joint3 via unique triples
    a0 = arr[:-2].astype(np.int64); a1 = arr[1:-1].astype(np.int64); a2 = arr[2:].astype(np.int64)
    trip = (a0 << 16) | (a1 << 8) | a2
    u3, c3 = np.unique(trip, return_counts=True)
    h_joint3 = H_from_counts(c3.astype(np.float64))
    h_joint3_mm = miller_madow(h_joint3, len(u3), n - 2)
    h2_mm = h_joint3_mm - h_joint2_mm
    # Brotli-11
    br_size = len(brotli.compress(arr.tobytes(), quality=11))
    br_rate = 8.0 * br_size / n
    return dict(
        n_bytes=int(n),
        H0_bpB=h0,
        H1_MM_bpB=h1_mm,
        H2_MM_bpB=h2_mm,
        brotli11_bpB=br_rate,
        H0_minus_brotli=h0 - br_rate,
        H1_MM_minus_brotli=h1_mm - br_rate,
        H2_MM_minus_brotli=h2_mm - br_rate,
        observed_trigrams=int(len(u3)),
        trigram_singleton_frac=float((c3 == 1).sum()) / int(len(u3)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for m in MODELS:
        for rho in RHOS:
            print(f"[{m} rho={rho}]")
            b = build_fp8_bytes(m, rho, args.device)
            arr = np.frombuffer(b, dtype=np.uint8)
            r = analyze_fp8(arr)
            r["model"] = m
            r["rho"] = rho
            rows.append(r)
            print(f"  n={r['n_bytes']:,} H0={r['H0_bpB']:.4f} "
                  f"H1={r['H1_MM_bpB']:.4f} H2={r['H2_MM_bpB']:.4f} "
                  f"br11={r['brotli11_bpB']:.4f} "
                  f"H2-br={r['H2_MM_minus_brotli']:+.4f} "
                  f"single3gram={r['trigram_singleton_frac']:.3f}")
    Path(args.out).write_text(json.dumps({
        "claim": 21, "experiment": "fp8_rho_sweep",
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
