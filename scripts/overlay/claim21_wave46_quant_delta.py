"""claim21_wave46_quant_delta.py -- wave 46.

LOSSY DELTA QUANTIZATION.

Builds directly on wave 45. Wave 45 established that brotli-11(delta)
is ~3x smaller than brotli-11(ft) losslessly. This probe asks: since
deltas are small-magnitude (|ft - base| << |ft|), can we quantize the
delta to int8 / int4 with negligible quality cost and get another 2-4x
on top?

For each pair, per 2-D linear weight, we compute per-tensor symmetric
absmax quantization of the delta at multiple bit-widths, measure:

  - raw delta bf16 bytes
  - raw delta int8 bytes (scales stored as fp16)
  - raw delta int4 bytes (nibble-packed, scales fp16)
  - brotli-11 of each

and also the reconstruction error so we can honestly report a
quality-vs-size Pareto. No hidden losses.

Reconstructed ft = base (bf16 as shipped) + dequant(delta_qN).
Error metric per tensor: rel Frobenius ||dequant - true_delta||_F /
                                         ||true_delta||_F .

This does NOT retrain or tune; it is a pure post-hoc quant probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import brotli
import numpy as np
import torch
from safetensors.torch import load_file as safe_load
from huggingface_hub import snapshot_download


def load_hf_state_dict(repo_id: str) -> dict:
    local = Path(snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"]))
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {local}")
    sd = {}
    for s in shards:
        sd.update(safe_load(str(s)))
    return sd


def matching_linear_keys(sd_base: dict, sd_ft: dict) -> list[str]:
    keys = []
    for k, v in sd_base.items():
        if not k.endswith(".weight"):
            continue
        if v.ndim != 2:
            continue
        if k not in sd_ft:
            continue
        if sd_ft[k].shape != v.shape:
            continue
        keys.append(k)
    return sorted(keys)


def quant_symmetric(delta_fp32: torch.Tensor, bits: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric absmax quant.

    Returns (quantized ints, scale fp32 scalar).
    """
    absmax = delta_fp32.abs().max().clamp(min=1e-12)
    qmax = 2 ** (bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(delta_fp32 / scale).clamp(-qmax - 1, qmax).to(torch.int32)
    return q, scale


def pack_int4(q: torch.Tensor) -> bytes:
    """Pack int4 [-8..+7] values two-per-byte, low nibble first."""
    n = q.numel()
    # Shift to unsigned 0..15 for packing
    u = (q.to(torch.int32) + 8).clamp(0, 15).to(torch.uint8).flatten()
    if n % 2 == 1:
        u = torch.cat([u, torch.zeros(1, dtype=torch.uint8)])
    lo = u[::2]
    hi = u[1::2]
    return ((hi << 4) | lo).numpy().tobytes()


def pack_int8(q: torch.Tensor) -> bytes:
    return q.to(torch.int8).numpy().tobytes()


def dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def brotli_bytes(b: bytes) -> int:
    return len(brotli.compress(b, quality=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"== wave 46 QUANT-DELTA base={args.base} ft={args.ft} ==")

    print("  loading state_dicts...")
    sd_b = load_hf_state_dict(args.base)
    sd_f = load_hf_state_dict(args.ft)
    keys = matching_linear_keys(sd_b, sd_f)
    print(f"  matched 2-D linear weights: {len(keys)}")

    bf16_parts: list[bytes] = []
    int8_parts: list[bytes] = []
    int4_parts: list[bytes] = []
    scales8: list[float] = []
    scales4: list[float] = []
    relerr8_num = 0.0
    relerr8_den = 0.0
    relerr4_num = 0.0
    relerr4_den = 0.0
    n_params = 0

    for k in keys:
        Wb = sd_b[k].to(torch.bfloat16).to(torch.float32)
        Wf = sd_f[k].to(torch.bfloat16).to(torch.float32)
        D = Wf - Wb
        n_params += D.numel()

        # bf16 reference (matches wave 45 stream byte-exactly)
        bf16_parts.append(
            D.to(torch.bfloat16).contiguous()
             .view(torch.int16).numpy().tobytes())

        # int8 symmetric
        # Reported relerr uses fp16 round-tripped scale to match what a real
        # decoder reading scales8_bytes (fp16) would observe (decoder-faithful).
        q8, s8 = quant_symmetric(D, 8)
        int8_parts.append(pack_int8(q8))
        scales8.append(float(s8))
        s8_fp16 = float(torch.tensor(float(s8), dtype=torch.float16))
        dq8 = dequant(q8, s8_fp16)
        relerr8_num += float(((dq8 - D) ** 2).sum())
        relerr8_den += float((D ** 2).sum())

        # int4 symmetric
        q4, s4 = quant_symmetric(D, 4)
        int4_parts.append(pack_int4(q4))
        scales4.append(float(s4))
        s4_fp16 = float(torch.tensor(float(s4), dtype=torch.float16))
        dq4 = dequant(q4, s4_fp16)
        relerr4_num += float(((dq4 - D) ** 2).sum())
        relerr4_den += float((D ** 2).sum())

    bf16 = b"".join(bf16_parts)
    i8   = b"".join(int8_parts)
    i4   = b"".join(int4_parts)
    # Scales stored as fp16: 2 bytes per tensor, amortized over the whole stream.
    scales8_bytes = np.array(scales8, dtype=np.float16).tobytes()
    scales4_bytes = np.array(scales4, dtype=np.float16).tobytes()

    print(f"  {n_params:,} params, bf16={len(bf16):,}  "
          f"int8+scales={len(i8)+len(scales8_bytes):,}  "
          f"int4+scales={len(i4)+len(scales4_bytes):,}")

    print("  brotli-11 bf16 ...")
    t = time.time(); br_bf16 = brotli_bytes(bf16); t_bf16 = time.time() - t
    print(f"    {br_bf16:,} B ({t_bf16:.0f}s)")
    print("  brotli-11 int8+scales ...")
    t = time.time(); br_i8 = brotli_bytes(i8 + scales8_bytes); t_i8 = time.time() - t
    print(f"    {br_i8:,} B ({t_i8:.0f}s)")
    print("  brotli-11 int4+scales ...")
    t = time.time(); br_i4 = brotli_bytes(i4 + scales4_bytes); t_i4 = time.time() - t
    print(f"    {br_i4:,} B ({t_i4:.0f}s)")

    relerr8 = (relerr8_num / relerr8_den) ** 0.5
    relerr4 = (relerr4_num / relerr4_den) ** 0.5

    out = {
        "claim": 21,
        "wave": 46,
        "experiment": "quant_delta",
        "base_repo": args.base,
        "ft_repo":   args.ft,
        "n_params": n_params,
        "sha256": {
            "bf16":  hashlib.sha256(bf16).hexdigest(),
            "int8":  hashlib.sha256(i8 + scales8_bytes).hexdigest(),
            "int4":  hashlib.sha256(i4 + scales4_bytes).hexdigest(),
        },
        "raw_bytes": {
            "bf16":  len(bf16),
            "int8+scales_fp16":  len(i8) + len(scales8_bytes),
            "int4+scales_fp16":  len(i4) + len(scales4_bytes),
        },
        "brotli_11_bytes": {
            "bf16":  br_bf16,
            "int8+scales_fp16":  br_i8,
            "int4+scales_fp16":  br_i4,
        },
        "rel_frobenius_reconstruction_error": {
            "int8":  relerr8,
            "int4":  relerr4,
        },
        "ratios_vs_brotli11_bf16_delta": {
            "int8":  br_bf16 / br_i8,
            "int4":  br_bf16 / br_i4,
        },
        "wall_seconds_total": time.time() - t0,
    }
    print()
    print("  HEADLINE:")
    print(f"    brotli-11(bf16 delta) / brotli-11(int8+sc) = "
          f"{out['ratios_vs_brotli11_bf16_delta']['int8']:.3f}x  "
          f"(relerr {relerr8:.4e})")
    print(f"    brotli-11(bf16 delta) / brotli-11(int4+sc) = "
          f"{out['ratios_vs_brotli11_bf16_delta']['int4']:.3f}x  "
          f"(relerr {relerr4:.4e})")

    out_path = Path(args.out)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
