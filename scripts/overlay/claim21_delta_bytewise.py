"""claim21_delta_bytewise.py -- idx_delta per-byte-position diagnostic.

The idx_delta stream is a sequence of little-endian int32 deltas
(4 bytes each). The wave-18 block-shuffle result showed savings
jump from B=1 to B=4 (a +13.9pp recovery for lzma-6), suggesting
structural dependency on byte position WITHIN the 4-byte int32.

This diagnostic proves it directly: reshape the idx_delta stream
as (N, 4) and emit per-byte-position histograms + Shannon entropy
for each of the 4 byte lanes.

Prediction: byte 0 (LSB) carries most of the entropy; bytes 2-3
(MSBs) are nearly always 0 (deltas are typically small positive
integers). The sum of per-lane entropies is an upper bound on the
achievable per-element rate assuming byte-position independence;
comparing it to the full byte-permutation entropy of the stream
isolates the structural savings from byte-position structure.

Emits: results/claim21_delta_bytewise_<model>_rho<rho>.json
"""
from __future__ import annotations

import argparse
import json
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


def shannon_H(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0: return 0.0
    p = counts.astype(np.float64) / n
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/claim21_delta_bytewise_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    device = torch.device(args.device)
    print(f"[delta-bytewise] model={args.model} rho={args.rho}")

    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]
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

    _, idx_b, _ = pack_streams_with_order(per_linear, "sorted", seed=0)
    assert len(idx_b) % 4 == 0, "idx_delta must be int32-aligned"
    arr = np.frombuffer(idx_b, dtype=np.uint8).reshape(-1, 4)
    n_elem = arr.shape[0]

    # Also decode as int32 to report delta statistics (little-endian)
    deltas = np.frombuffer(idx_b, dtype=np.int32)

    per_byte = []
    H_sum = 0.0
    for pos in range(4):
        h = np.bincount(arr[:, pos], minlength=256).astype(int)
        H = shannon_H(h.astype(np.float64))
        H_sum += H
        nonzero_frac = float((arr[:, pos] != 0).mean())
        per_byte.append({
            "byte_position": pos,
            "shannon_H_bpB": H,
            "order0_savings_floor_pct": 100.0 * (8.0 - H) / 8.0,
            "zero_fraction": 1.0 - nonzero_frac,
            "nonzero_fraction": nonzero_frac,
            "hist": h.tolist(),
        })

    # full-stream H for reference (same as wave 19)
    h_full = np.bincount(np.frombuffer(idx_b, dtype=np.uint8), minlength=256)
    H_full = shannon_H(h_full.astype(np.float64))

    results = {
        "claim": 21, "experiment": "delta_bytewise",
        "model": args.model, "rho": args.rho,
        "idx_delta_bytes": len(idx_b),
        "n_deltas": int(n_elem),
        "delta_stats": {
            "min": int(deltas.min()), "max": int(deltas.max()),
            "mean": float(deltas.mean()), "median": float(np.median(deltas)),
            "p99": float(np.quantile(deltas, 0.99)),
        },
        "full_stream_H_bpB": H_full,
        "full_stream_floor_pct": 100.0 * (8.0 - H_full) / 8.0,
        "per_byte_position": per_byte,
        "sum_per_byte_H_bits_per_int32": H_sum,
        "bound_bits_per_int32_if_bytes_independent": H_sum,
        "bound_bytes_per_int32_if_bytes_independent": H_sum / 8.0,
        "achievable_floor_pct_if_bytes_independent": 100.0 * (32.0 - H_sum) / 32.0,
    }

    print(f"\n  n_deltas={n_elem:,}  idx_delta_bytes={len(idx_b):,}")
    print(f"  deltas: min={deltas.min()} max={deltas.max()} "
          f"mean={deltas.mean():.2f} p99={np.quantile(deltas,0.99):.1f}")
    print(f"  full-stream H = {H_full:.4f} bpB (floor {(8-H_full)/8*100:.2f}%)")
    for pb in per_byte:
        print(f"  byte pos {pb['byte_position']}: H = {pb['shannon_H_bpB']:.4f} bpB "
              f"(floor {pb['order0_savings_floor_pct']:5.2f}%)  "
              f"zero_frac = {pb['zero_fraction']*100:.2f}%")
    print(f"  sum per-byte H = {H_sum:.3f} bits/int32  "
          f"(={H_sum/8:.3f} bytes/int32; floor {100*(32-H_sum)/32:.2f}%)")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
