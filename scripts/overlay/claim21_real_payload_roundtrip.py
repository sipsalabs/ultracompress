"""claim21_real_payload_roundtrip.py -- empirical lossless verification on the
REAL Claim-21 payload bytes (not random buffers).

For one (model, rho), builds the three actual overlay-payload streams
(fp8 K x I value matrix, int32-delta row-index stream, fp16 row-scales)
exactly the way Claim 21 emits them, then for every supported codec:

  compressed = codec.compress(stream)
  restored   = codec.decompress(compressed)
  assert SHA-256(restored) == SHA-256(stream)

This strengthens the standards-sufficiency argument (every tested codec
is a published lossless standard: zstd RFC 8478, zlib RFC 1950/1951,
bzip2 spec, LZMA/xz spec, brotli RFC 7932, LZ4 frame spec) into a
measurement on the actual bytes that Claim 21 produces.

Emits:
  results/claim21_real_payload_roundtrip_<model>_rho<rho>.json
  results/claim21_real_payload_roundtrip.txt
"""
from __future__ import annotations

import argparse
import bz2 as _bz2
import hashlib
import json
import lzma as _lzma
import sys
import time
import zlib as _zlib
from pathlib import Path

import torch
import zstandard as _zstd

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

try:
    import brotli as _brotli
    HAVE_BROTLI = True
except ImportError:
    HAVE_BROTLI = False

try:
    import lz4.frame as _lz4
    HAVE_LZ4 = True
except ImportError:
    HAVE_LZ4 = False


def _zstd_enc(level):
    def f(b):
        return _zstd.ZstdCompressor(level=level).compress(b)
    return f


def _zstd_dec(b):
    return _zstd.ZstdDecompressor().decompress(b)


CODECS = [
    ("zstd-3",    _zstd_enc(3),                                          _zstd_dec),
    ("zstd-9",    _zstd_enc(9),                                          _zstd_dec),
    ("zstd-15",   _zstd_enc(15),                                         _zstd_dec),
    ("zstd-22",   _zstd_enc(22),                                         _zstd_dec),
    ("zlib-9",    lambda b: _zlib.compress(b, 9),                        _zlib.decompress),
    ("bz2-9",     lambda b: _bz2.compress(b, 9),                         _bz2.decompress),
    ("lzma-6",    lambda b: _lzma.compress(b, preset=6),                 _lzma.decompress),
]
if HAVE_BROTLI:
    CODECS.append(("brotli-11", lambda b: _brotli.compress(b, quality=11), _brotli.decompress))
if HAVE_LZ4:
    CODECS.append(("lz4-hc",    lambda b: _lz4.compress(b, compression_level=16), _lz4.decompress))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.010)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/claim21_real_payload_roundtrip_{args.model}_rho{args.rho}.json"

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    teacher_path = REPO / teacher_pt
    v17_path = REPO / v17_pt

    device = torch.device(args.device)
    print(f"[real-roundtrip] model={args.model} rho={args.rho} device={device}")

    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    banks = v17["banks"]
    s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    per_linear = []
    t0 = time.time()
    for i, k in enumerate(hf_keys):
        W = sd[k]
        role = _role_of(k)
        bank = banks[role]
        O, I = W.shape
        s = s_col.get(k, torch.ones(I))
        idx, rows, scl = collect_rows_per_linear(
            W, role, bank, s, D, rots[I], device, args.rho,
        )
        per_linear.append((idx, rows, scl))
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(hf_keys)}]  t={int(time.time()-t0)}s")

    fp8_b, idx_b, scl_b = pack_streams_with_order(per_linear, "sorted", seed=0)
    streams = {"fp8": fp8_b, "idx_delta": idx_b, "scale": scl_b}
    shas = {k: sha256(v) for k, v in streams.items()}
    sizes = {k: len(v) for k, v in streams.items()}
    print(f"  real payload: fp8={sizes['fp8']:,}  idx_delta={sizes['idx_delta']:,}  scale={sizes['scale']:,}")
    for k, s in shas.items():
        print(f"    sha256 {k:<10} = {s[:16]}...")

    results = {
        "model": args.model,
        "rho": args.rho,
        "raw_bytes": sizes,
        "raw_sha256": shas,
        "codecs": {},
    }
    n_pass = 0
    n_total = 0
    for name, enc, dec in CODECS:
        results["codecs"][name] = {}
        for sk, sb in streams.items():
            t1 = time.time()
            c = enc(sb)
            t2 = time.time()
            d = dec(c)
            t3 = time.time()
            ok = (sha256(d) == shas[sk]) and (d == sb)
            n_total += 1
            if ok:
                n_pass += 1
            results["codecs"][name][sk] = {
                "compressed_bytes": len(c),
                "enc_s": t2 - t1,
                "dec_s": t3 - t2,
                "sha256_match": ok,
            }
            print(f"    {name:<11} {sk:<10}  {sb.__len__():>10,} -> {len(c):>10,}  "
                  f"enc={t2-t1:6.2f}s  dec={t3-t2:5.2f}s  sha256_ok={ok}")
        del c, d

    results["pass"] = n_pass
    results["total"] = n_total
    results["pass_rate"] = n_pass / n_total if n_total else 0.0

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {out_path}")
    print(f"[summary] {n_pass}/{n_total} SHA-256 roundtrips PASSED  "
          f"({100.0 * n_pass / max(n_total,1):.4f}%)")


if __name__ == "__main__":
    main()
