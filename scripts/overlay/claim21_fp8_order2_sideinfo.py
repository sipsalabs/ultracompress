"""claim21_fp8_order2_sideinfo.py -- wave 36.

Wave 35 concluded that the -0.155 bpB sub-brotli gap from wave 31 is
realizable only by shipping the order-2 context table as side
information. This wave quantifies the side-info cost exactly:
serialize the 65,536 x 256 count table per model, compress it with
several general-purpose coders, and compute NET gain after paying
the serialization cost.

For each model:
  1. Build full-stream order-2 count table (int32, 65536 x 256)
  2. Serialize several ways:
       a. raw int32 bytes  (reference only)
       b. varint (variable-length integer per cell)
       c. zlib-9 on raw int32
       d. brotli-11 on raw int32
       e. brotli-11 on varint
  3. Compute payload savings vs brotli-11 at the oracle alpha=0.01
     static rate (wave 34's achieved rate)
  4. Compute NET gain = savings - side_info_bits / N_bytes, reported
     in bits per payload fp8 byte

Also measures the 'rho amortization floor' -- at rho=0.010 the fp8
stream is 1% of the underlying bf16 tensor bytes, so 1 side-info byte
per 100 fp8 bytes is 1 byte per 10000 bf16 bytes = 0.01% overhead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zlib

try:
    import brotli
except ImportError:
    brotli = None

from claim21_fp8_order2 import build_fp8_bytes
from claim21_fp8_order2_universal import build_order2_counts, static_code_rate


MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]


def varint_encode(values: np.ndarray) -> bytes:
    """Simple unsigned varint (LEB128) over int64 values."""
    out = bytearray()
    for v in values.tolist():
        v = int(v)
        if v < 0:
            raise ValueError("varint requires nonneg")
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[fp8-order2-sideinfo] rho={args.rho}")
    rows = []
    for m in MODELS:
        print(f"  [{m}]")
        arr = np.frombuffer(build_fp8_bytes(m, args.rho, args.device), dtype=np.uint8)
        n = int(arr.size)
        counts = build_order2_counts(arr, alpha=0.0).astype(np.int64)  # raw counts (no prior)
        nnz = int((counts > 0).sum())

        flat = counts.reshape(-1)
        # (a) raw int32
        raw32_bytes = flat.astype(np.int32).tobytes()
        raw32_size = len(raw32_bytes)
        # (b) varint
        vbytes = varint_encode(flat)
        vsize = len(vbytes)
        # (c) zlib-9 on raw int32
        zsize = len(zlib.compress(raw32_bytes, 9))
        # (d) brotli-11 on raw int32
        bsize_raw = len(brotli.compress(raw32_bytes, quality=11)) if brotli else -1
        # (e) brotli-11 on varint
        bsize_varint = len(brotli.compress(vbytes, quality=11)) if brotli else -1

        # Choose the best side-info encoder and compute overhead
        encodings = {
            "raw_int32": raw32_size,
            "varint": vsize,
            "zlib9_int32": zsize,
            "brotli11_int32": bsize_raw,
            "brotli11_varint": bsize_varint,
        }
        best_name, best_bytes = min(
            ((k, v) for k, v in encodings.items() if v >= 0),
            key=lambda kv: kv[1],
        )

        # Payload oracle rate at alpha=0.01 and brotli-11 baseline
        oracle_counts = counts.astype(np.float64) + 0.01
        oracle_rate = static_code_rate(arr, oracle_counts)
        # brotli-11 baseline from codec_sweep file
        cs = json.loads(Path(f"results/claim21_codec_sweep_{m}_rho0.01.json").read_text())
        br_rate = float(cs["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])

        # Per-byte side-info overhead (bits per fp8 payload byte)
        side_info_bits_per_byte = 8.0 * best_bytes / n

        net_rate = oracle_rate + side_info_bits_per_byte
        gain_vs_brotli = br_rate - net_rate  # positive = better than brotli-11

        row = dict(
            model=m,
            n_bytes=n,
            observed_nonzero_cells=nnz,
            encodings=encodings,
            best_encoder=best_name,
            best_side_info_bytes=best_bytes,
            oracle_alpha001_bpB=oracle_rate,
            brotli11_bpB=br_rate,
            side_info_bits_per_payload_byte=side_info_bits_per_byte,
            net_rate_bpB=net_rate,
            net_gain_vs_brotli11_bpB=gain_vs_brotli,
        )
        rows.append(row)
        print(f"    n={n:,} nnz={nnz:,}")
        print(f"    encodings: raw32={raw32_size:,}  varint={vsize:,}  "
              f"zlib9={zsize:,}  br11-raw={bsize_raw:,}  br11-var={bsize_varint:,}")
        print(f"    best: {best_name} = {best_bytes:,} bytes = {side_info_bits_per_byte:.4f} bpB overhead")
        print(f"    oracle a=0.01 rate = {oracle_rate:.4f}  brotli-11 = {br_rate:.4f}  "
              f"net = {net_rate:.4f}  gain = {gain_vs_brotli:+.4f} bpB")

    out = dict(
        claim=21,
        experiment="fp8_order2_sideinfo",
        rho=args.rho,
        models=rows,
    )
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
