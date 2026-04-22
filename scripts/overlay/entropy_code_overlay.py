"""entropy_code_overlay.py -- Claim 21 evidence.

Measure the lossless compressibility of the fp8 overlay payload (fp8 values +
fp16 scales + row indices) that Claims 18 / 18A / 18D currently store raw.

For each restored body-Linear, we collect the exact bytes that the overlay
would serialize and measure:

  1. Shannon entropy  H(fp8 bytes)           -- info-theoretic lower bound
  2. zstd level 22    (fp8 bytes)            -- practical coder
  3. Delta + zstd     (row indices)          -- indices are sorted ascending
  4. fp16 scales left raw (low volume, near-uniform in log space)

Reports: effective bpw before vs after entropy coding, aggregated across the
whole body. A ~25-35% reduction in the fp8 value portion is expected because
residual magnitudes cluster near zero; E4M3 inherits that skew directly.

Run from repo root:
    python scripts/overlay/entropy_code_overlay.py --model qwen3_1.7b --rho 0.003
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation  # noqa: E402
from compress_v15 import beam_assign                              # noqa: E402


MODEL_CONFIGS = {
    "qwen3_0.6b":   ("qwen3_0.6b_cache.pt",   "v17hi_fit_qwen3_0.6b.pt"),
    "qwen3_1.7b":   ("qwen3_1.7b_cache.pt",   "v17hi_fit_qwen3_1.7b.pt"),
    "qwen3_8b":     ("qwen3_8b_cache.pt",     "v17hi_fit_8b.pt"),
    "smollm2_1.7b": ("smollm2_1.7b_cache.pt", "v17hi_fit_smollm2.pt"),
    "tinyllama":    ("tinyllama_1.1b_cache.pt", "v17hi_fit_tinyllama.pt"),
    "olmo2_1b":     ("olmo2_1b_cache.pt",     "v17hi_fit_olmo2.pt"),
    "mistral_7b":   ("mistral_7b_v0.3_cache.pt", "v17hi_fit_mistral.pt"),
}


def shannon_entropy_bytes(buf: bytes) -> float:
    """Shannon entropy of an i.i.d. byte source, bits/symbol."""
    if not buf:
        return 0.0
    arr = np.frombuffer(buf, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def zstd_compress(buf: bytes, level: int = 22) -> bytes:
    """Compress with zstd if available, else zlib level 9 as fallback."""
    try:
        import zstandard as zstd
        return zstd.ZstdCompressor(level=level).compress(buf)
    except ImportError:
        import zlib
        return zlib.compress(buf, 9)


def _fp8_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3 encode. Returns (fp8_bytes [O,I] uint8, scale [O] fp16)."""
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 448.0                                    # [O, 1]
    xs = (x / scale).to(torch.float8_e4m3fn)                  # fp8
    xb = xs.view(torch.uint8)                                  # bit-identical uint8
    return xb, scale.squeeze(1).to(torch.float16)


def _fp8_decode(xb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    xs = xb.view(torch.float8_e4m3fn)
    return xs.to(torch.float32) * scale.unsqueeze(1).to(torch.float32)


def select_overlay_rows(W_fp16, role, bank, s_col, D, rot, device, rho,
                        score_mode="weighted"):
    """Replicates the selection half of `_reconstruct_v17_with_fp8_overlay`.

    Returns (idx [K], rows_fp8_bytes [K,I] cpu uint8, row_scales [K] cpu fp16).
    """
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    W_scaled = W * s.unsqueeze(0)
    Wrot = W_scaled @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    chunk = max(25_000, (200_000 * 2048) // max(cb1.shape[0], 1))
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=chunk)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)
    raw = W - Wq
    diff = raw * s.unsqueeze(0) if score_mode == "weighted" else raw
    score = (diff * diff).sum(1)
    K = max(1, int(round(rho * O))) if rho > 0 else 0
    if K == 0:
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, I, dtype=torch.uint8),
                torch.empty(0, dtype=torch.float16))
    idx = score.topk(K).indices
    idx_sorted, _ = idx.sort()
    rows_fp32 = W[idx_sorted]
    rows_fp8, scales = _fp8_encode(rows_fp32)
    return idx_sorted.cpu(), rows_fp8.cpu(), scales.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--rho",   type=float, default=0.003)
    ap.add_argument("--score", default="weighted", choices=["weighted", "raw"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out",   default="results/claim21_entropy_coding.json")
    ap.add_argument("--codec_sweep", action="store_true",
                    help="Also measure zstd levels {3,9,15,22}, zlib, bz2, lzma on each stream.")
    args = ap.parse_args()

    teacher_pt, v17_pt = MODEL_CONFIGS[args.model]
    teacher_path = REPO / teacher_pt
    v17_path = REPO / v17_pt
    assert teacher_path.exists(), f"missing teacher cache: {teacher_path}"
    assert v17_path.exists(),     f"missing v17 fit: {v17_path}"

    device = torch.device(args.device)
    print(f"[entropy-code] model={args.model} rho={args.rho} device={device}")
    print(f"  teacher = {teacher_path.name}")
    print(f"  v17     = {v17_path.name}")

    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = int(v17.get("D", 8))
    gbpw = float(v17.get("global_bpw", 2.78))
    banks = v17["banks"]
    s_col = v17["s_col"]

    hf_keys = [k for k in sd.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and sd[k].ndim == 2
               and sd[k].shape[1] % D == 0]
    print(f"  body linears: {len(hf_keys)}")

    dims = sorted({sd[k].shape[1] for k in hf_keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    # Aggregators
    n_restored_rows = 0
    n_total_rows = 0
    n_total_params = 0
    n_restored_params = 0

    fp8_bytes_concat = bytearray()
    idx_deltas_concat = bytearray()
    scale_bytes_concat = bytearray()

    t0 = time.time()
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = sd[k]
        O, I = W.shape
        s = s_col.get(k, torch.ones(I))

        idx, rows_fp8, scales = select_overlay_rows(
            W, role, bank, s, D, rots[I], device, args.rho, args.score)

        n_total_rows += O
        n_total_params += W.numel()
        if idx.numel() > 0:
            K = int(idx.numel())
            n_restored_rows += K
            n_restored_params += K * I

            # fp8 bytes: K * I uint8
            fp8_bytes_concat += bytes(rows_fp8.numpy().tobytes())

            # row indices: delta-encode ascending, pack as uint16 (O < 65536)
            assert O < 65536, f"row dim {O} >= 65536, need uint32 delta"
            idx_np = idx.numpy().astype(np.int32)
            deltas = np.concatenate([[idx_np[0]], np.diff(idx_np)]).astype(np.uint16)
            idx_deltas_concat += bytes(deltas.tobytes())

            # fp16 scales: K * 2 bytes
            scale_bytes_concat += bytes(scales.numpy().tobytes())

        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [{n+1}/{len(hf_keys)}] rows={n_restored_rows} "
                  f"t={time.time()-t0:.0f}s")

    print(f"[entropy-code] selection done in {time.time()-t0:.0f}s")
    print(f"  restored rows: {n_restored_rows}/{n_total_rows} "
          f"({100*n_restored_rows/max(n_total_rows,1):.3f}%)")

    # ---------- Entropy measurements ----------
    fp8_buf   = bytes(fp8_bytes_concat)
    idx_buf   = bytes(idx_deltas_concat)
    scale_buf = bytes(scale_bytes_concat)

    H_fp8     = shannon_entropy_bytes(fp8_buf)
    H_idx     = shannon_entropy_bytes(idx_buf)
    H_scale   = shannon_entropy_bytes(scale_buf)

    z_fp8   = zstd_compress(fp8_buf)
    z_idx   = zstd_compress(idx_buf)
    z_scale = zstd_compress(scale_buf)

    def _ratio(coded, raw):
        return (len(coded) / len(raw)) if raw else 0.0

    print()
    print("[payload sizes & compression]")
    print(f"  fp8 values :   raw={len(fp8_buf):>12,}  zstd22={len(z_fp8):>12,}  "
          f"ratio={_ratio(z_fp8,fp8_buf)*100:5.2f}%  H={H_fp8:4.2f} bits/byte")
    print(f"  row deltas :   raw={len(idx_buf):>12,}  zstd22={len(z_idx):>12,}  "
          f"ratio={_ratio(z_idx,idx_buf)*100:5.2f}%  H={H_idx:4.2f} bits/byte")
    print(f"  fp16 scales:   raw={len(scale_buf):>12,}  zstd22={len(z_scale):>12,}  "
          f"ratio={_ratio(z_scale,scale_buf)*100:5.2f}%  H={H_scale:4.2f} bits/byte")

    # ---------- bpw accounting ----------
    # Old overlay bpw (Claim 18A / Claim 20 setup):
    #   per restored row: 8*I (fp8) + 16 (scale) + 32 (idx) bits
    # New overlay bpw (Claim 21):
    #   per restored row: [entropy-coded fp8] + 16 (scale) + [delta+entropy idx]
    avg_I = n_restored_params / max(n_restored_rows, 1)

    old_bits = n_restored_rows * (8 * avg_I + 16 + 32)
    new_bits = (len(z_fp8) + len(z_idx) + len(z_scale)) * 8

    old_overlay_bpw = old_bits / max(n_total_params, 1)
    new_overlay_bpw = new_bits / max(n_total_params, 1)

    old_effective = gbpw + old_overlay_bpw
    new_effective = gbpw + new_overlay_bpw

    saved_bpw  = old_effective - new_effective
    saved_pct  = 100 * (old_bits - new_bits) / max(old_bits, 1)

    print()
    print("[bpw accounting]")
    print(f"  base (v17 codebook):       {gbpw:.4f} bpw")
    print(f"  overlay (raw fp8+idx+scl): +{old_overlay_bpw:.4f}  -> {old_effective:.4f} bpw")
    print(f"  overlay (entropy coded) :  +{new_overlay_bpw:.4f}  -> {new_effective:.4f} bpw")
    print(f"  savings on overlay bits :  {saved_pct:.2f}% "
          f"({saved_bpw:+.4f} bpw absolute)")

    # ---------- Optional multi-codec sweep (Claim 21 strengthening) ----------
    codec_sweep = None
    if args.codec_sweep:
        import zlib as _zlib, bz2 as _bz2, lzma as _lzma
        import zstandard as _zstd
        try:
            import brotli as _brotli
            _have_brotli = True
        except ImportError:
            _have_brotli = False
        try:
            import lz4.frame as _lz4
            _have_lz4 = True
        except ImportError:
            _have_lz4 = False

        def _zstd_at(buf, lvl):
            return _zstd.ZstdCompressor(level=lvl).compress(buf)

        CODECS = [
            ("zstd-3",  lambda b: _zstd_at(b, 3)),
            ("zstd-9",  lambda b: _zstd_at(b, 9)),
            ("zstd-15", lambda b: _zstd_at(b, 15)),
            ("zstd-22", lambda b: _zstd_at(b, 22)),
            ("zlib-9",  lambda b: _zlib.compress(b, 9)),
            ("bz2-9",   lambda b: _bz2.compress(b, 9)),
            ("lzma-6",  lambda b: _lzma.compress(b, preset=6)),
        ]
        if _have_brotli:
            CODECS.append(("brotli-11", lambda b: _brotli.compress(b, quality=11)))
        if _have_lz4:
            CODECS.append(("lz4-hc",    lambda b: _lz4.compress(b, compression_level=16)))

        streams = {"fp8": fp8_buf, "idx_delta": idx_buf, "scale": scale_buf}
        shannon = {"fp8": H_fp8, "idx_delta": H_idx, "scale": H_scale}

        codec_sweep = {}
        print()
        print("[codec sweep] per-stream compressed size (bytes) and bits/byte")
        header = f"  {'stream':<10} {'raw':>12} {'Shannon':>8}  " + " ".join(f"{n:>8}" for n, _ in CODECS)
        print(header)
        for name, raw in streams.items():
            row = {
                "raw_bytes": len(raw),
                "shannon_bits_per_byte": shannon[name],
                "shannon_bytes": len(raw) * shannon[name] / 8.0,
                "codecs": {},
            }
            row_str = f"  {name:<10} {len(raw):>12,} {shannon[name]:>8.3f}  "
            for cname, cfn in CODECS:
                c = cfn(raw)
                row["codecs"][cname] = {
                    "bytes": len(c),
                    "bits_per_byte": 8.0 * len(c) / max(len(raw), 1),
                }
                row_str += f"{len(c):>8,} "
            print(row_str)
            codec_sweep[name] = row

        # Cross-codec overlay-bpw table (sum across 3 streams)
        print()
        print("[codec sweep] total overlay bpw + savings vs raw overlay")
        raw_overlay_bits = 8 * (len(fp8_buf) + len(idx_buf) + len(scale_buf))
        for cname, _ in CODECS:
            total_bytes = sum(codec_sweep[s]["codecs"][cname]["bytes"] for s in streams)
            total_bits = 8 * total_bytes
            bpw_overlay = total_bits / max(n_total_params, 1)
            saved_pct_c = 100 * (raw_overlay_bits - total_bits) / raw_overlay_bits
            shannon_total_bits = sum(codec_sweep[s]["shannon_bytes"] for s in streams) * 8
            gap_pct = 100 * (total_bits - shannon_total_bits) / max(raw_overlay_bits - shannon_total_bits, 1)
            print(f"  {cname:<8}  total={total_bytes:>12,} bytes  "
                  f"bpw={bpw_overlay:.5f}  saved={saved_pct_c:5.2f}%  "
                  f"gap->H={gap_pct:+5.2f}%")

    # ---------- emit JSON ----------
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "claim": 21,
        "model": args.model,
        "rho": args.rho,
        "score_mode": args.score,
        "n_restored_rows": int(n_restored_rows),
        "n_total_rows": int(n_total_rows),
        "n_restored_params": int(n_restored_params),
        "n_total_params": int(n_total_params),
        "avg_row_width": float(avg_I),
        "base_bpw": gbpw,
        "payload_bytes": {
            "fp8_raw":        int(len(fp8_buf)),
            "fp8_zstd22":     int(len(z_fp8)),
            "idx_delta_raw":  int(len(idx_buf)),
            "idx_delta_zstd": int(len(z_idx)),
            "scale_raw":      int(len(scale_buf)),
            "scale_zstd":     int(len(z_scale)),
        },
        "shannon_entropy_bits_per_byte": {
            "fp8":   H_fp8,
            "idx":   H_idx,
            "scale": H_scale,
        },
        "old_overlay_bpw":  float(old_overlay_bpw),
        "new_overlay_bpw":  float(new_overlay_bpw),
        "old_effective_bpw": float(old_effective),
        "new_effective_bpw": float(new_effective),
        "saved_pct_of_overlay_bits": float(saved_pct),
        "saved_bpw_absolute": float(saved_bpw),
    }
    if codec_sweep is not None:
        result["codec_sweep"] = codec_sweep
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[entropy-code] wrote {out_path}")


if __name__ == "__main__":
    main()
