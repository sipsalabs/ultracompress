"""claim21_scale_pair_decomp.py -- wave 26.

Wave 21 decomposed idx_delta int32 bytes and showed bytes 2+3 were
structurally zero (H=0). Wave 26 does the same for the fp16 scale
stream: reshape scale bytes as (N, 2) uint8 pairs (fp16 little-endian
layout -> byte 0 = low byte = mantissa LSBs; byte 1 = sign+exponent+
mantissa MSBs). Measure:
  - Shannon H per byte position (256-bin order-0)
  - full joint H over 65536-bin fp16 value distribution
  - mutual information I(B0; B1) = H(B0) + H(B1) - H(B0,B1)

If I is LARGE => a pair-aware coder can extract it; if small =>
byte positions are independent and a byte-level coder is already
close to optimal.

Uses wave-23 codec-sweep pipeline row-collection.

Per-model output: results/claim21_scale_pair_<model>_rho0.01.json
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa
from entropy_code_overlay import MODEL_CONFIGS                    # noqa
from claim21_row_order_invariance import (                        # noqa
    collect_rows_per_linear, pack_streams_with_order,
)


def build_scale_bytes(model: str, rho: float, device: str) -> bytes:
    teacher_pt, v17_pt = MODEL_CONFIGS[model]
    dev = torch.device(device)
    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8)); banks = v17["banks"]; s_col = v17["s_col"]
    hf_keys = [k for k in sd.keys() if "layers." in k
               and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, dev, seed=42 + I) for I in dims}
    per_linear = []
    for k in hf_keys:
        role = _role_of(k); bank = banks[role]
        W = sd[k]; I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], dev, rho)
        per_linear.append((idx, rows, scl))
    _, _, scale_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    return scale_b


def shannon_H(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot == 0:
        return 0.0
    p = counts / tot
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[scale-pair] model={args.model}  rho={args.rho}")
    b = build_scale_bytes(args.model, args.rho, args.device)
    arr = np.frombuffer(b, dtype=np.uint8)
    assert arr.size % 2 == 0, f"scale stream must be even bytes, got {arr.size}"
    pairs = arr.reshape(-1, 2)   # [N, 2]; little-endian fp16
    N = pairs.shape[0]
    print(f"  n_scales={N:,}  n_bytes={arr.size:,}")

    # Per-byte-position order-0 H
    c0 = np.bincount(pairs[:, 0], minlength=256)
    c1 = np.bincount(pairs[:, 1], minlength=256)
    H0 = shannon_H(c0)
    H1 = shannon_H(c1)

    # Joint H over 65536-bin fp16 value distribution.
    combo = pairs[:, 0].astype(np.uint32) | (pairs[:, 1].astype(np.uint32) << 8)
    cj = np.bincount(combo, minlength=1 << 16)
    H_joint = shannon_H(cj)

    # Bytewise sum = upper bound on joint H if bytes independent
    H_sum = H0 + H1
    MI = H_sum - H_joint  # non-negative

    # fp16 per-pair byte-decomposition: sign(1) | exp(5) | mantissa(10)
    # high byte b1 = sign(1) | exp(5) | mantissa_hi(2)  (bits 15..8)
    # low byte  b0 = mantissa_lo(8)                      (bits 7..0)
    sign = (pairs[:, 1] >> 7) & 0x1
    exp  = (pairs[:, 1] >> 2) & 0x1F
    mant_hi = pairs[:, 1] & 0x3
    c_sign = np.bincount(sign, minlength=2)
    c_exp  = np.bincount(exp, minlength=32)
    c_mhi  = np.bincount(mant_hi, minlength=4)
    H_sign = shannon_H(c_sign)
    H_exp  = shannon_H(c_exp)
    H_mhi  = shannon_H(c_mhi)

    raw_bits_per_scale = 16.0
    # If we packed sign+exp+mant_hi independently (small) and mant_lo (H0)
    # the lower bound of per-scale independent-field H is:
    H_field_sum = H_sign + H_exp + H_mhi + H0  # bits per scale

    print()
    print(f"  byte 0  (mantissa LSB)     H = {H0:.4f} bpB   (256 bins)")
    print(f"  byte 1  (sign/exp/mant_hi) H = {H1:.4f} bpB   (256 bins)")
    print(f"  H(B0)+H(B1)                = {H_sum:.4f} bits/scale "
          f"= {H_sum/2:.4f} bpB")
    print(f"  H(B0, B1) joint fp16       = {H_joint:.4f} bits/scale "
          f"= {H_joint/2:.4f} bpB")
    print(f"  mutual information I(B0;B1)= {MI:.4f} bits/scale")
    print()
    print(f"  field-level: sign H = {H_sign:.4f}")
    print(f"  field-level: exp  H = {H_exp:.4f}  (5-bit field)")
    print(f"  field-level: mhi  H = {H_mhi:.4f}  (2-bit field)")
    print(f"  field-level: mlo  H = {H0:.4f}    (8-bit field)")
    print(f"  field-sum          = {H_field_sum:.4f} bits/scale "
          f"= {H_field_sum/2:.4f} bpB")
    print(f"  raw fp16           = {raw_bits_per_scale:.4f} bits/scale "
          f"= 8.0000 bpB")

    out = {
        "claim": 21, "experiment": "scale_pair_decomp",
        "model": args.model, "rho": args.rho,
        "n_scales": int(N),
        "n_bytes": int(arr.size),
        "byte0_H_bpB": H0,
        "byte1_H_bpB": H1,
        "byte_sum_bits_per_scale": H_sum,
        "byte_sum_bpB": H_sum / 2.0,
        "joint_H_bits_per_scale": H_joint,
        "joint_H_bpB": H_joint / 2.0,
        "mutual_information_bits_per_scale": MI,
        "field_sign_H": H_sign,
        "field_exp_H": H_exp,
        "field_mant_hi_H": H_mhi,
        "field_mant_lo_H": H0,
        "field_sum_bits_per_scale": H_field_sum,
        "field_sum_bpB": H_field_sum / 2.0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
