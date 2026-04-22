"""claim21_fp8_order2.py -- wave 31.

Wave 30 proved brotli-11 ships fp8 BELOW the order-1 floor by 0.10
bpB, so brotli-11 uses order >= 2 context. Wave 31 measures the
order-2 floor itself: H(B_i | B_{i-1}, B_{i-2}) = H(B_i, B_{i-1},
B_{i-2}) - H(B_{i-1}, B_{i-2}). Comparison to brotli-11 tells us
how much of brotli's context use is order-2 vs order >= 3.

256^3 = 16,777,216 joint bins; uint32 bincount + float64 pmf = 64MB.
"""
from __future__ import annotations
import argparse, json, sys
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


def build_fp8_bytes(model: str, rho: float, device: str) -> bytes:
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
    fp8_b, _, _ = pack_streams_with_order(per_linear, "sorted", seed=0)
    return fp8_b


def H_from_counts(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot == 0: return 0.0
    p = counts.astype(np.float64) / float(tot)
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2] model={args.model}  rho={args.rho}")
    b = build_fp8_bytes(args.model, args.rho, args.device)
    arr = np.frombuffer(b, dtype=np.uint8)
    n = arr.size
    print(f"  n_fp8_bytes={n:,}")

    a0 = arr[:-2].astype(np.uint32)
    a1 = arr[1:-1].astype(np.uint32)
    a2 = arr[2:].astype(np.uint32)

    # joint triple counts over 256^3
    triple = (a0 << 16) | (a1 << 8) | a2
    c3 = np.bincount(triple, minlength=1 << 24).astype(np.int64)
    # joint pair counts of (prev2, prev1) = (a0, a1) -- we only use pairs
    # that actually precede some a2, so same length as a0
    pair = (a0 << 8) | a1
    c2 = np.bincount(pair, minlength=1 << 16).astype(np.int64)

    H3 = H_from_counts(c3)
    H2 = H_from_counts(c2)
    H_cond2 = H3 - H2  # bits per current byte given previous two
    print(f"  H(B_i, B_{{i-1}}, B_{{i-2}}) = {H3:.4f} bits / 3 bytes")
    print(f"  H(B_{{i-1}}, B_{{i-2}})      = {H2:.4f} bits / 2 bytes")
    print(f"  order-2 H                    = {H_cond2:.4f} bpB")

    out = dict(
        claim=21, experiment="fp8_order2",
        model=args.model, rho=args.rho, n_fp8_bytes=int(n),
        H_joint3=H3, H_joint2=H2,
        order2_H_bpB=H_cond2,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
