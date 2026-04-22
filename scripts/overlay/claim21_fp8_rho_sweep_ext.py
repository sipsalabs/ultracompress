"""claim21_fp8_rho_sweep_ext.py -- wave 40.

Wave 39 measured fp8 H2 MM - brotli-11 gaps at rho in {0.005, 0.010,
0.020} and showed the gap contracts 18x as sample size quadruples
(from -0.122 bpB at rho=0.005 to -0.007 bpB at rho=0.020). Wave 40
extends the sweep to rho=0.040 to test whether the cohort gap flips
POSITIVE at larger sample sizes -- the natural bookend to the rho-
decay curve.

Also reports plug-in H2 alongside Miller-Madow H2 so the contribution
of the MM correction itself is visible.
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


def analyze_fp8(arr):
    n = arr.size
    c0 = np.bincount(arr, minlength=256).astype(np.float64)
    h0 = H_from_counts(c0)
    pair = (arr[:-1].astype(np.int64) << 8) | arr[1:].astype(np.int64)
    u2, c2 = np.unique(pair, return_counts=True)
    h_joint2 = H_from_counts(c2.astype(np.float64))
    h_joint2_mm = miller_madow(h_joint2, len(u2), n - 1)
    h1_plugin = h_joint2 - h0
    h1_mm = h_joint2_mm - h0
    a0 = arr[:-2].astype(np.int64); a1 = arr[1:-1].astype(np.int64); a2 = arr[2:].astype(np.int64)
    trip = (a0 << 16) | (a1 << 8) | a2
    u3, c3 = np.unique(trip, return_counts=True)
    h_joint3 = H_from_counts(c3.astype(np.float64))
    h_joint3_mm = miller_madow(h_joint3, len(u3), n - 2)
    h2_plugin = h_joint3 - h_joint2
    h2_mm = h_joint3_mm - h_joint2_mm
    br_size = len(brotli.compress(arr.tobytes(), quality=11))
    br_rate = 8.0 * br_size / n
    return dict(
        n_bytes=int(n),
        H0_bpB=h0,
        H1_plugin_bpB=h1_plugin,
        H1_MM_bpB=h1_mm,
        H2_plugin_bpB=h2_plugin,
        H2_MM_bpB=h2_mm,
        brotli11_bpB=br_rate,
        H2_plugin_minus_brotli=h2_plugin - br_rate,
        H2_MM_minus_brotli=h2_mm - br_rate,
        MM_correction_bpB=h2_mm - h2_plugin,
        observed_trigrams=int(len(u3)),
        trigram_singleton_frac=float((c3 == 1).sum()) / int(len(u3)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for m in MODELS:
        print(f"[{m} rho={args.rho}]")
        b = build_fp8_bytes(m, args.rho, args.device)
        arr = np.frombuffer(b, dtype=np.uint8)
        r = analyze_fp8(arr)
        r["model"] = m; r["rho"] = args.rho
        rows.append(r)
        print(f"  n={r['n_bytes']:,} H2pl={r['H2_plugin_bpB']:.4f} "
              f"H2mm={r['H2_MM_bpB']:.4f} br11={r['brotli11_bpB']:.4f} "
              f"H2pl-br={r['H2_plugin_minus_brotli']:+.4f} "
              f"H2mm-br={r['H2_MM_minus_brotli']:+.4f} "
              f"MMcorr={r['MM_correction_bpB']:+.4f} "
              f"single={r['trigram_singleton_frac']:.3f}")
    Path(args.out).write_text(json.dumps({
        "claim": 21, "experiment": "fp8_rho_sweep_ext",
        "rho": args.rho, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
