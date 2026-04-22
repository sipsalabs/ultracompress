"""claim21_varint_emitter.py -- validate the wave-23 idx_delta coder
gap prediction by measuring alternative bit-level encodings.

Wave 23 showed idx_delta's best general-purpose LZ coder (brotli-11)
sits 1.55 bpB ABOVE the order-0 Shannon floor of 2.78 bpB. We
predicted that a simple bit-packed variable-length integer coder
would close most of that gap. This wave MEASURES it directly:

For the same restored-row delta sequence, compute the size of:
  - raw int32 LE                       (the current layout)
  - LEB128 varint (7 bits per byte)    (standard varint)
  - zigzag + LEB128                    (handles signed, used by protobuf)
  - Elias gamma code (bit-packed)      (unary prefix + binary)
  - Rice coding with parameter k       (parameterised by median)
  - Byte-position entropy (theoretical lower bound from wave 21)

Emits per (model, rho=0.010): bits/delta for each scheme.
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


# ------------------------- coders -----------------------------

def leb128_bytes(values: np.ndarray) -> int:
    """Size in bytes of unsigned LEB128 varint encoding of values."""
    total = 0
    for v in values:
        v = int(v)
        if v == 0:
            total += 1
            continue
        # number of 7-bit chunks = ceil(bit_length / 7)
        bl = v.bit_length()
        total += (bl + 6) // 7
    return total


def gamma_bits(values: np.ndarray) -> int:
    """Elias gamma code: (1 + floor(log2(v+1))) * 2 - 1 bits for v>=0
    Using v+1 to handle v=0 as gamma(1).
    """
    total = 0
    for v in values:
        n = int(v) + 1
        k = n.bit_length()  # floor(log2(n))+1
        total += 2 * k - 1
    return total


def rice_bits(values: np.ndarray, k: int) -> int:
    """Rice code with parameter k: quotient (unary) + k-bit remainder.
    bits = floor(v/2^k) + 1 + k
    """
    if k < 0:
        k = 0
    pot = 1 << k
    total = 0
    for v in values:
        q = int(v) // pot
        total += q + 1 + k
    return total


def rice_best_k(values: np.ndarray, k_range=range(0, 12)) -> tuple[int, int]:
    """Find Rice parameter k that minimises total bits. Returns (k, bits)."""
    arr = values.astype(np.int64)
    best_k = 0
    best_bits = None
    for k in k_range:
        pot = 1 << k
        q = arr // pot
        bits = int(q.sum()) + len(arr) * (1 + k)
        if best_bits is None or bits < best_bits:
            best_bits = bits
            best_k = k
    return best_k, int(best_bits)


# ------------------------- payload extraction -----------------------------

def build_idx_deltas(model: str, rho: float, device: str) -> np.ndarray:
    teacher_pt, v17_pt = MODEL_CONFIGS[model]
    dev = torch.device(device)
    sd = torch.load(REPO / teacher_pt, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(REPO / v17_pt, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    banks = v17["banks"]
    s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, dev, seed=42 + I) for I in dims}

    per_linear = []
    for k in hf_keys:
        role = _role_of(k)
        bank = banks[role]
        W = sd[k]; I = W.shape[1]
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], dev, rho)
        per_linear.append((idx, rows, scl))

    _, idx_b, _ = pack_streams_with_order(per_linear, "sorted", seed=0)
    deltas = np.frombuffer(idx_b, dtype=np.int32).astype(np.int64)
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[varint-emitter] model={args.model}  rho={args.rho}")
    deltas = build_idx_deltas(args.model, args.rho, args.device)
    n = len(deltas)
    print(f"  n_deltas={n:,}")
    print(f"  delta_stats: min={deltas.min()} max={deltas.max()} "
          f"mean={float(deltas.mean()):.2f} median={float(np.median(deltas)):.2f}")

    raw_bytes = n * 4
    raw_bits = raw_bytes * 8

    leb_b = leb128_bytes(deltas)
    leb_bits = leb_b * 8

    gm_bits = gamma_bits(deltas)

    # Rice with k derived from median (optimal is often log2(median))
    med = max(1, int(np.median(deltas)))
    k0 = max(0, int(math.log2(med)))
    rice_med_bits = rice_bits(deltas, k0)
    best_k, best_rice_bits = rice_best_k(deltas)

    # Order-0 theoretical floor (on delta distribution itself, not bytes):
    # H = -sum p log2 p over delta values
    vals, counts = np.unique(deltas, return_counts=True)
    p = counts / counts.sum()
    H_delta_bits = float(-np.sum(p * np.log2(p))) * n

    def fmt(total_bits):
        return total_bits / n, total_bits / raw_bits * 4.0  # bits/delta, bpB

    def bpB_of_bits(tb):
        # bpB relative to original 4-byte int32 representation
        return tb / raw_bytes

    rows = [
        ("int32 LE (current)", raw_bits),
        ("LEB128 varint",      leb_bits),
        ("Elias gamma",        gm_bits),
        (f"Rice k={k0} (median)", rice_med_bits),
        (f"Rice k={best_k} (best)", best_rice_bits),
        ("Shannon H (floor)",  H_delta_bits),
    ]

    print()
    print("  scheme                        bits/delta   bytes/delta   bpB(4-byte ref)")
    for name, tb in rows:
        bpd = tb / n
        Bpd = bpd / 8.0
        bpB = tb / raw_bytes
        print(f"  {name:<30}  {bpd:>10.4f}   {Bpd:>10.4f}    {bpB:>10.4f}")

    out = {
        "claim": 21,
        "experiment": "varint_emitter",
        "model": args.model,
        "rho": args.rho,
        "n_deltas": n,
        "delta_stats": {
            "min": int(deltas.min()),
            "max": int(deltas.max()),
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "p99": float(np.percentile(deltas, 99)),
        },
        "raw_int32_bits": raw_bits,
        "leb128_bits": leb_bits,
        "gamma_bits": gm_bits,
        "rice_median_k": k0,
        "rice_median_bits": rice_med_bits,
        "rice_best_k": best_k,
        "rice_best_bits": best_rice_bits,
        "shannon_H_total_bits": H_delta_bits,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
