"""claim21_fp8_histogram.py -- order-0 byte-histogram diagnostic.

Direct measurement of how far the Claim 21 payload byte distributions
deviate from uniform. This is the *fundamental cause* of the fp8 order-0
savings identified in wave 17 (byte-permutation): if bytes were uniform,
no coder could beat 8.00 bits/byte on fp8. We measure:

 - Shannon entropy H (bits/byte)
 - Total-variation distance from uniform (0..1)
 - Chi-square vs. uniform, normalized per-byte
 - Max histogram deviation ratio (max_count / mean_count)

For each model at rho=0.010. Reuses pack_streams_with_order "sorted".

Emits: results/claim21_fp8_histogram_<model>_rho<rho>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa: E402
from entropy_code_overlay import MODEL_CONFIGS                    # noqa: E402
from claim21_row_order_invariance import (                        # noqa: E402
    collect_rows_per_linear,
    pack_streams_with_order,
)


def analyze(b: bytes) -> dict:
    arr = np.frombuffer(b, dtype=np.uint8)
    n = len(arr)
    h = np.bincount(arr, minlength=256).astype(np.float64)
    p = h / n
    # Shannon entropy, bits/byte
    nz = p[p > 0]
    H = float(-(nz * np.log2(nz)).sum())
    # Uniform distribution
    u = 1.0 / 256.0
    # TV distance
    tv = float(0.5 * np.abs(p - u).sum())
    # Chi-square
    exp = n * u
    chi2 = float(((h - exp) ** 2 / exp).sum())
    # Max deviation ratio
    max_ratio = float(h.max() / (n * u))
    min_ratio = float(h.min() / (n * u))
    # "Savings floor" from order-0: (8 - H) / 8 * 100
    savings_floor_pct = 100.0 * (8.0 - H) / 8.0
    return {
        "n_bytes": int(n),
        "shannon_bits_per_byte": H,
        "bits_below_uniform": float(8.0 - H),
        "order0_savings_floor_pct": savings_floor_pct,
        "tv_distance_from_uniform": tv,
        "chi2_vs_uniform": chi2,
        "max_count_over_mean": max_ratio,
        "min_count_over_mean": min_ratio,
        "histogram": h.astype(int).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/claim21_fp8_histogram_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[fp8-histogram] model={args.model} rho={args.rho}")

    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    banks = v17["banks"]; s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    per_linear = []
    for k in hf_keys:
        role = _role_of(k); bank = banks[role]
        W = sd[k]; I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], device, args.rho)
        per_linear.append((idx, rows, scl))

    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)

    results = {
        "claim": 21,
        "experiment": "fp8_histogram",
        "model": args.model,
        "rho": args.rho,
        "streams": {
            "fp8":       analyze(fp8_b),
            "idx_delta": analyze(idx_b),
            "scale":     analyze(scl_b),
        },
    }

    for sname, s in results["streams"].items():
        print(f"  {sname:9}  n={s['n_bytes']:>12,}  H={s['shannon_bits_per_byte']:.4f} bpB"
              f"  floor={s['order0_savings_floor_pct']:5.2f}%"
              f"  TV={s['tv_distance_from_uniform']:.4f}"
              f"  max/mean={s['max_count_over_mean']:.2f}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
