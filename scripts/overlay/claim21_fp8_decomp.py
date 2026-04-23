"""claim21_fp8_decomp.py -- wave 43.

Hunt for a lossless fp8 preconditioning that BEATS raw brotli-11.

The fp8 (e4m3) byte stream is the dominant payload (~99.9% of total
bytes). All five entropy-coder families explored in waves 30-37 lost
to brotli-11. Wave 43 takes a different angle: keep brotli-11 as the
backend, but PRECONDITION the fp8 byte stream via cheap, perfectly
invertible byte-level decompositions before handing it to brotli. If
any decomposition produces a smaller brotli-11 output (sum of
sub-streams) than the raw stream, we lower the headline number for
free, with O(n) preprocessing cost (no throughput penalty).

Decompositions tested (all bijective):

  raw          : reference -- the byte stream as-is
  hi_lo_nibble : split each byte into high nibble (b>>4) and low
                 nibble (b&0xF), pack nibbles back into 4-bit-per-byte
                 streams (2 sub-streams of length n/2 each)
  bitplane8    : 8 sub-streams, sub-stream k holds bit k of every
                 byte, packed 8 bits per byte (8 sub-streams of n/8)
  e4m3_split   : fp8 e4m3 logical split: sub-stream S = sign bit (n/8
                 bytes), E = 4 exponent bits (n/2 bytes packed 2-per),
                 M = 3 mantissa bits (3n/8 bytes packed at 1 nibble +
                 spillover -- here we pack as 1 byte per sample for
                 simplicity, total 3*n bytes pre-pack -> we re-pack 4
                 mantissa values per 3 bytes via dense bit packing)
  byte_pair    : interleaved pair-byte split: (b[0::2], b[1::2]),
                 testing whether even/odd byte position carries
                 different statistics
  delta_byte   : signed byte delta: out[0]=b[0]; out[i]=b[i]-b[i-1]
                 (mod 256). Bijective. Tests whether residual stream
                 brotli-compresses better

For each decomposition: compute total brotli-11 bytes summed across
all sub-streams. The gain vs raw is reported in bpB (cohort total
bits / cohort total bytes). Negative gain = decomposition WINS.

Lossless-ness sanity: each decomposition is paired with an inverse
function and we verify byte-equality on the original buffer before
reporting the rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import brotli
import numpy as np

from claim21_streams_order2 import build_all_streams


# ---------------- decompositions (each returns list[bytes]; each pair
#                  with an inverse fn returning bytes) ----------------

def d_raw(b: bytes):
    return [b]

def i_raw(parts):
    return parts[0]


def d_hi_lo_nibble(b: bytes):
    a = np.frombuffer(b, dtype=np.uint8)
    n = a.size
    hi = (a >> 4).astype(np.uint8)
    lo = (a & 0x0F).astype(np.uint8)
    # pack 2 nibbles per byte
    if n % 2 == 1:
        hi = np.concatenate([hi, np.array([0], dtype=np.uint8)])
        lo = np.concatenate([lo, np.array([0], dtype=np.uint8)])
    hi_packed = ((hi[0::2] << 4) | hi[1::2]).astype(np.uint8)
    lo_packed = ((lo[0::2] << 4) | lo[1::2]).astype(np.uint8)
    return [bytes(hi_packed), bytes(lo_packed), n.to_bytes(8, "little")]

def i_hi_lo_nibble(parts):
    hi_packed = np.frombuffer(parts[0], dtype=np.uint8)
    lo_packed = np.frombuffer(parts[1], dtype=np.uint8)
    n = int.from_bytes(parts[2], "little")
    hi = np.empty(hi_packed.size * 2, dtype=np.uint8)
    hi[0::2] = hi_packed >> 4
    hi[1::2] = hi_packed & 0x0F
    lo = np.empty(lo_packed.size * 2, dtype=np.uint8)
    lo[0::2] = lo_packed >> 4
    lo[1::2] = lo_packed & 0x0F
    out = ((hi[:n] << 4) | lo[:n]).astype(np.uint8)
    return bytes(out)


def d_bitplane8(b: bytes):
    a = np.frombuffer(b, dtype=np.uint8)
    n = a.size
    out = []
    for k in range(8):
        bits = ((a >> k) & 1).astype(np.uint8)
        # pack 8 bits/byte
        pad = (-bits.size) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        packed = np.packbits(bits, bitorder="little")
        out.append(bytes(packed))
    out.append(n.to_bytes(8, "little"))
    return out

def i_bitplane8(parts):
    n = int.from_bytes(parts[8], "little")
    a = np.zeros(n, dtype=np.uint8)
    for k in range(8):
        packed = np.frombuffer(parts[k], dtype=np.uint8)
        bits = np.unpackbits(packed, bitorder="little")[:n]
        a |= (bits.astype(np.uint8) << k)
    return bytes(a)


def d_e4m3_split(b: bytes):
    """fp8 e4m3 logical split: 1-bit sign, 4-bit exponent, 3-bit mantissa.

    e4m3 layout (IEEE-style): bit 7 = sign, bits 6-3 = exponent,
    bits 2-0 = mantissa.
    """
    a = np.frombuffer(b, dtype=np.uint8)
    n = a.size
    sign = (a >> 7) & 1               # 1 bit
    expn = (a >> 3) & 0x0F            # 4 bits
    mant = a & 0x07                   # 3 bits
    # pack sign 8/byte, exp 2/byte (high nibble + low nibble),
    # mant -- pack 8 mantissa values into 3 bytes (24 bits = 8*3)
    pad = (-n) % 8
    if pad:
        sign = np.concatenate([sign, np.zeros(pad, dtype=np.uint8)])
        mant = np.concatenate([mant, np.zeros(pad, dtype=np.uint8)])
    sign_packed = np.packbits(sign.astype(np.uint8), bitorder="little")
    if n % 2 == 1:
        expn_p = np.concatenate([expn, np.array([0], dtype=np.uint8)])
    else:
        expn_p = expn
    expn_packed = ((expn_p[0::2] << 4) | expn_p[1::2]).astype(np.uint8)
    # mant: take 8 vals -> pack into 24 bits = 3 bytes, repeat
    m = mant.astype(np.uint32)
    g = m.size // 8
    blocks = m[: g * 8].reshape(g, 8)
    # pack: bits 21-23 = sample0, 18-20 = sample1, ..., 0-2 = sample7
    word = np.zeros(g, dtype=np.uint32)
    for i in range(8):
        word |= (blocks[:, i] & 0x07) << (21 - 3 * i)
    mant_bytes = np.empty(g * 3, dtype=np.uint8)
    mant_bytes[0::3] = (word >> 16) & 0xFF
    mant_bytes[1::3] = (word >> 8) & 0xFF
    mant_bytes[2::3] = word & 0xFF
    return [bytes(sign_packed), bytes(expn_packed), bytes(mant_bytes),
            n.to_bytes(8, "little")]

def i_e4m3_split(parts):
    n = int.from_bytes(parts[3], "little")
    sign = np.unpackbits(np.frombuffer(parts[0], dtype=np.uint8),
                         bitorder="little")[:n].astype(np.uint8)
    epacked = np.frombuffer(parts[1], dtype=np.uint8)
    expn = np.empty(epacked.size * 2, dtype=np.uint8)
    expn[0::2] = epacked >> 4
    expn[1::2] = epacked & 0x0F
    expn = expn[:n]
    mb = np.frombuffer(parts[2], dtype=np.uint8)
    g = mb.size // 3
    word = (mb[0::3].astype(np.uint32) << 16) | \
           (mb[1::3].astype(np.uint32) << 8) | \
           mb[2::3].astype(np.uint32)
    mant = np.zeros(g * 8, dtype=np.uint8)
    for i in range(8):
        mant[i::8] = ((word >> (21 - 3 * i)) & 0x07).astype(np.uint8)
    mant = mant[:n]
    out = ((sign & 1) << 7) | ((expn & 0x0F) << 3) | (mant & 0x07)
    return bytes(out.astype(np.uint8))


def d_byte_pair(b: bytes):
    a = np.frombuffer(b, dtype=np.uint8)
    return [bytes(a[0::2]), bytes(a[1::2]), len(b).to_bytes(8, "little")]

def i_byte_pair(parts):
    n = int.from_bytes(parts[2], "little")
    even = np.frombuffer(parts[0], dtype=np.uint8)
    odd = np.frombuffer(parts[1], dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = even
    out[1::2] = odd
    return bytes(out)


def d_delta_byte(b: bytes):
    a = np.frombuffer(b, dtype=np.uint8).astype(np.int16)
    d = np.empty_like(a)
    d[0] = a[0]
    d[1:] = (a[1:] - a[:-1]) & 0xFF
    return [bytes(d.astype(np.uint8))]

def i_delta_byte(parts):
    d = np.frombuffer(parts[0], dtype=np.uint8).astype(np.int16)
    a = np.empty_like(d)
    a[0] = d[0]
    for i in range(1, d.size):
        a[i] = (a[i - 1] + d[i]) & 0xFF
    return bytes(a.astype(np.uint8))


DECOMPS = {
    "raw":          (d_raw,          i_raw),
    "hi_lo_nibble": (d_hi_lo_nibble, i_hi_lo_nibble),
    "bitplane8":    (d_bitplane8,    i_bitplane8),
    "e4m3_split":   (d_e4m3_split,   i_e4m3_split),
    "byte_pair":    (d_byte_pair,    i_byte_pair),
    "delta_byte":   (d_delta_byte,   i_delta_byte),
}


def evaluate(b: bytes, decomp_name: str):
    enc, dec = DECOMPS[decomp_name]
    parts = enc(b)
    rec = dec(parts)
    assert rec == b, f"{decomp_name}: roundtrip mismatch"
    total_brotli = sum(len(brotli.compress(p, quality=11)) for p in parts)
    return {
        "decomp": decomp_name,
        "n_input_bytes": len(b),
        "n_subparts": len(parts),
        "subpart_sizes": [len(p) for p in parts],
        "total_brotli11_bytes": total_brotli,
        "bpB": 8.0 * total_brotli / len(b),
        "roundtrip_ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"== claim21 wave 43 fp8_decomp [{args.model}] rho={args.rho} ==")
    fp8, idx, scl = build_all_streams(args.model, args.rho, args.device)
    print(f"  fp8={len(fp8):,}  idx_delta={len(idx):,}  scale={len(scl):,}")
    fp8_sha = hashlib.sha256(fp8).hexdigest()

    results = []
    for name in DECOMPS.keys():
        t1 = time.time()
        rec = evaluate(fp8, name)
        rec["wall_seconds"] = time.time() - t1
        results.append(rec)
        print(f"  [{name:<14}] {rec['total_brotli11_bytes']:>10,} B  "
              f"{rec['bpB']:.5f} bpB  ({rec['wall_seconds']:.0f}s)  "
              f"OK roundtrip")

    raw_bpB = next(r["bpB"] for r in results if r["decomp"] == "raw")
    for r in results:
        r["delta_vs_raw_bpB"] = r["bpB"] - raw_bpB

    out = {
        "claim": 21,
        "wave": 43,
        "experiment": "fp8_decomp",
        "model": args.model,
        "rho": args.rho,
        "fp8_bytes": len(fp8),
        "fp8_sha256": fp8_sha,
        "raw_brotli11_bpB": raw_bpB,
        "results": results,
        "best_decomp": min(results, key=lambda r: r["bpB"])["decomp"],
        "best_bpB": min(r["bpB"] for r in results),
        "best_gain_vs_raw_bpB": min(r["bpB"] for r in results) - raw_bpB,
        "wall_seconds_total": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  best={out['best_decomp']}  gain={out['best_gain_vs_raw_bpB']:+.5f} bpB")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
