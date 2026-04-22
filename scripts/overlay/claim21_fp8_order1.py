"""claim21_fp8_order1.py -- wave 30.

Waves 27-29 established that further fp8 gains need a context-aware
coder. Wave 30 measures directly the information-theoretic lower
bound of a byte-context-1 coder on fp8: the conditional entropy
    H(B_i | B_{i-1}) = H(B_i, B_{i-1}) - H(B_{i-1})
measured over the entire fp8 byte stream of each model at rho=0.010.

Also reports:
  - order-0 H (should match wave 19 exactly)
  - order-1 H (this wave)
  - order-0 - order-1 = 'context gain headroom'
  - observed brotli-11 bpB (wave 15/23)

If order-1 H is at or below brotli-11 bpB => any residual gap is
coder overhead, NOT information-theoretic missing context.
If order-1 H is clearly above brotli-11 bpB => brotli-11 is already
using longer-range context beyond order-1.

Writes: results/claim21_fp8_order1_<model>_rho0.01.json
"""
from __future__ import annotations
import argparse, json, math, sys
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
    fp8_b, _, _ = pack_streams_with_order(per_linear, "sorted", seed=0)
    return fp8_b


def order0_H_bpB(c256) -> float:
    c = np.asarray(c256, dtype=np.float64); tot = c.sum()
    if tot == 0: return 0.0
    p = c / tot; nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def order1_conditional_H_bpB(arr: np.ndarray) -> tuple[float, float]:
    """Return (H(B_i, B_{i-1}) per 2 bytes, H(B_i | B_{i-1}) in bits/byte).

    Uses the full stream as a single sequence of adjacent byte pairs
    (overlapping).  Conditional H computed from the 256x256 joint.
    """
    assert arr.dtype == np.uint8 and arr.size >= 2
    prev = arr[:-1].astype(np.uint32)
    cur  = arr[1:].astype(np.uint32)
    pair = prev * 256 + cur                # 0..65535
    joint = np.bincount(pair, minlength=1 << 16).astype(np.float64)
    tot = joint.sum()
    joint = joint / tot                     # p(prev, cur)
    # marginal of prev
    p_prev = joint.reshape(256, 256).sum(axis=1)
    # H(cur | prev) = sum_prev p(prev) * H(cur | prev=prev)
    H_cond = 0.0
    for i in range(256):
        if p_prev[i] == 0: continue
        row = joint[i*256:(i+1)*256] / p_prev[i]   # p(cur|prev=i)
        nz = row > 0
        Hi = -(row[nz] * np.log2(row[nz])).sum()
        H_cond += p_prev[i] * Hi
    return float(H_cond), float(p_prev.sum())   # (bits/byte, sanity=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order1] model={args.model}  rho={args.rho}")
    b = build_fp8_bytes(args.model, args.rho, args.device)
    arr = np.frombuffer(b, dtype=np.uint8)
    n = arr.size
    print(f"  n_fp8_bytes={n:,}")

    # order 0
    c = np.bincount(arr, minlength=256)
    H0 = order0_H_bpB(c)
    # order 1
    H1, sanity = order1_conditional_H_bpB(arr)
    print(f"  order-0 H    = {H0:.4f} bpB")
    print(f"  order-1 H    = {H1:.4f} bpB   (conditional on prev byte)")
    print(f"  context gain = {H0-H1:.4f} bpB   (how much info B_{{i-1}} gives about B_i)")

    out = dict(
        claim=21, experiment="fp8_order1",
        model=args.model, rho=args.rho, n_fp8_bytes=int(n),
        order0_H_bpB=H0, order1_H_bpB=H1,
        context_gain_bpB=H0 - H1,
        sanity_marginal_sum=sanity,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
