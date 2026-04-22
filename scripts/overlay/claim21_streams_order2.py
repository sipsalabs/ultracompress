"""claim21_streams_order2.py -- wave 38.

Wave 37 closed the order-2 fp8 context-coding programme with a
definitive negative. Wave 38 pivots to the OTHER two streams in the
Claim-21 payload: idx_delta and scale. Mirrors wave 30 (order-0/1 +
brotli-11 compare) and wave 31 (order-2 Shannon floor) on these
streams to reveal whether either has an exploitable Shannon-vs-
brotli-11 gap similar to fp8's (but perhaps more / less favorable).

For each cohort model at rho=0.010:
  1. Build all 3 streams (fp8, idx_delta, scale) via pack_streams_with_order
  2. For each stream compute:
       - H_0 plug-in from byte counts
       - H_1 plug-in from 2-gram counts
       - H_2 plug-in from 3-gram counts (+ Miller-Madow correction)
       - brotli-11 rate
  3. Report Shannon-vs-brotli gaps at orders 0,1,2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np
import torch

from claim21_fp8_order2 import REPO  # path helper reused
from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from entropy_code_overlay import MODEL_CONFIGS
from claim21_row_order_invariance import (
    collect_rows_per_linear, pack_streams_with_order,
)

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def build_all_streams(model, rho, device):
    teacher_pt, v17_pt = MODEL_CONFIGS[model]
    dev = torch.device(device)
    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8)); banks = v17["banks"]; s_col = v17["s_col"]
    hf_keys = [k for k in sd.keys() if "layers." in k
               and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, dev, seed=42+I) for I in dims}
    per_linear = []
    for k in hf_keys:
        role = _role_of(k); bank = banks[role]; W = sd[k]; I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(W, role, bank, s, D, rots[I], dev, rho)
        per_linear.append((idx, rows, scl))
    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    return fp8_b, idx_b, scl_b


def H_from_counts(counts):
    tot = counts.sum()
    if tot == 0: return 0.0
    p = counts / tot
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def miller_madow(h_plugin, k_observed, n):
    return h_plugin + max(0, k_observed - 1) / (2.0 * n * np.log(2))


def analyze(arr, name):
    n = arr.size
    # H0
    c0 = np.bincount(arr, minlength=256).astype(np.float64)
    h0 = H_from_counts(c0)
    # H1: H(X2|X1) = H(X1,X2) - H(X1)
    if n >= 2:
        pair = (arr[:-1].astype(np.int64) << 8) | arr[1:].astype(np.int64)
        u, c = np.unique(pair, return_counts=True)
        c12 = c.astype(np.float64)
        h_joint2 = H_from_counts(c12)
        h1 = h_joint2 - h0
        h_joint2_mm = miller_madow(h_joint2, len(u), n - 1)
        h1_mm = h_joint2_mm - h0
    else:
        h1 = h1_mm = 0.0
    # H2: H(X3|X1,X2) = H(X1,X2,X3) - H(X1,X2)
    if n >= 3:
        a0 = arr[:-2].astype(np.int64); a1 = arr[1:-1].astype(np.int64); a2 = arr[2:].astype(np.int64)
        trip = (a0 << 16) | (a1 << 8) | a2
        u3, c3 = np.unique(trip, return_counts=True)
        h_joint3 = H_from_counts(c3.astype(np.float64))
        h2 = h_joint3 - h_joint2
        h_joint3_mm = miller_madow(h_joint3, len(u3), n - 2)
        h2_mm = h_joint3_mm - h_joint2_mm
        k3 = len(u3)
        singleton_frac = float((c3 == 1).sum()) / len(u3) if len(u3) else 0.0
    else:
        h2 = h2_mm = 0.0
        k3 = 0
        singleton_frac = 0.0
    # brotli-11
    br_size = len(brotli.compress(arr.tobytes(), quality=11))
    br_rate = 8.0 * br_size / n
    return dict(
        stream=name, n_bytes=int(n),
        H0_bpB=h0,
        H1_plugin_bpB=h1,
        H1_MM_bpB=h1_mm,
        H2_plugin_bpB=h2,
        H2_MM_bpB=h2_mm,
        brotli11_bpB=br_rate,
        H0_minus_brotli=h0 - br_rate,
        H1_MM_minus_brotli=h1_mm - br_rate,
        H2_MM_minus_brotli=h2_mm - br_rate,
        observed_trigrams=k3,
        trigram_singleton_frac=singleton_frac,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for m in MODELS:
        print(f"[{m}]")
        fp8_b, idx_b, scl_b = build_all_streams(m, args.rho, args.device)
        model_rows = []
        for name, b in [("fp8", fp8_b), ("idx_delta", idx_b), ("scale", scl_b)]:
            arr = np.frombuffer(b, dtype=np.uint8)
            r = analyze(arr, name)
            r["model"] = m
            model_rows.append(r)
            print(f"  {name:<10} n={r['n_bytes']:>10,} "
                  f"H0={r['H0_bpB']:.4f} H1={r['H1_MM_bpB']:.4f} "
                  f"H2={r['H2_MM_bpB']:.4f} br11={r['brotli11_bpB']:.4f} "
                  f"(H2-br={r['H2_MM_minus_brotli']:+.4f})")
        results.extend(model_rows)
    Path(args.out).write_text(json.dumps({
        "claim": 21, "experiment": "streams_order2", "rho": args.rho,
        "rows": results,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
