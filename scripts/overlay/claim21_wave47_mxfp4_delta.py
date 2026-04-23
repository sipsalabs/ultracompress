"""claim21_wave47_mxfp4_delta.py -- wave 47.

BLOCK-FLOAT (MX-LIKE) DELTA QUANTIZATION.

Wave 46 showed that per-tensor int4 delta is aggressive but coarse:
one absmax per tensor means a few outliers in the delta inflate the
scale, wasting the int4 dynamic range on small values that dominate
the distribution.

Wave 47 refines this: group the delta into contiguous blocks of
BLOCK=32 elements along the flattened axis, store one fp16 scale per
block, quantize the block's values to int4 against that per-block
absmax. This is the OCP MXFP4 block-float pattern adapted to the
delta. The hypothesis: delta is strongly leptokurtic so per-block
scales follow the local magnitude far better than a single per-tensor
scale, shrinking relerr at fixed bits, or enabling lower bits at fixed
error.

Reported metrics (all honest, fp16-round-tripped scales):
  - raw bytes (bf16 delta, int4+block_scales, int8+block_scales)
  - brotli-11 of each
  - rel-Frobenius ||dequant - D||_F / ||D||_F
  - scale overhead (bytes of block scales vs quantized values)

No retraining.
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


BLOCK = 32  # OCP MX standard block size


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
        if not k.endswith(".weight") or v.ndim != 2:
            continue
        if k not in sd_ft or sd_ft[k].shape != v.shape:
            continue
        keys.append(k)
    return sorted(keys)


def block_quant(delta_fp32: torch.Tensor, bits: int, block: int = BLOCK
                ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-block symmetric absmax quant.

    Returns (q_ints_flat, scales_fp32_per_block, pad_len).
    Pads the tail to a full block; pad ints are 0 (exact zero).
    """
    flat = delta_fp32.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    b = flat.view(-1, block)                                        # (B, block)
    absmax = b.abs().max(dim=1).values.clamp(min=1e-12)             # (B,)
    qmax = 2 ** (bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(b / scale.unsqueeze(1)).clamp(-qmax - 1, qmax).to(torch.int32)
    return q.reshape(-1), scale, pad


def pack_int4(q: torch.Tensor) -> bytes:
    n = q.numel()
    u = (q.to(torch.int32) + 8).clamp(0, 15).to(torch.uint8).flatten()
    if n % 2 == 1:
        u = torch.cat([u, torch.zeros(1, dtype=torch.uint8)])
    lo = u[::2]; hi = u[1::2]
    return ((hi << 4) | lo).numpy().tobytes()


def pack_int8(q: torch.Tensor) -> bytes:
    return q.to(torch.int8).numpy().tobytes()


def dequant_blocked(q_flat: torch.Tensor, scale_per_block: torch.Tensor,
                    block: int, n_real: int) -> torch.Tensor:
    b = q_flat.reshape(-1, block).to(torch.float32)
    out = (b * scale_per_block.unsqueeze(1)).reshape(-1)
    return out[:n_real]


def brotli_bytes(b: bytes) -> int:
    return len(brotli.compress(b, quality=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"== wave 47 MXFP4-DELTA base={args.base} ft={args.ft} block={BLOCK} ==")

    sd_b = load_hf_state_dict(args.base)
    sd_f = load_hf_state_dict(args.ft)
    keys = matching_linear_keys(sd_b, sd_f)
    print(f"  matched 2-D linear weights: {len(keys)}")

    bf16_parts: list[bytes] = []
    int8_parts: list[bytes] = []
    int4_parts: list[bytes] = []
    scales8_parts: list[bytes] = []   # concatenated per-block fp16 scales
    scales4_parts: list[bytes] = []
    relerr8_num = 0.0
    relerr8_den = 0.0
    relerr4_num = 0.0
    relerr4_den = 0.0
    n_params = 0
    n_blocks = 0

    for k in keys:
        Wb = sd_b[k].to(torch.bfloat16).to(torch.float32)
        Wf = sd_f[k].to(torch.bfloat16).to(torch.float32)
        D = Wf - Wb
        n_real = D.numel()
        n_params += n_real

        bf16_parts.append(
            D.to(torch.bfloat16).contiguous()
             .view(torch.int16).numpy().tobytes())

        # int8 blocked
        q8, s8, _pad8 = block_quant(D, 8, BLOCK)
        int8_parts.append(pack_int8(q8))
        s8_fp16_t = s8.to(torch.float16)
        scales8_parts.append(
            s8_fp16_t.numpy().astype(np.float16).tobytes())
        s8_rt = s8_fp16_t.to(torch.float32)
        dq8 = dequant_blocked(q8, s8_rt, BLOCK, n_real).reshape(D.shape)
        relerr8_num += float(((dq8 - D) ** 2).sum())
        relerr8_den += float((D ** 2).sum())

        # int4 blocked
        q4, s4, _pad4 = block_quant(D, 4, BLOCK)
        int4_parts.append(pack_int4(q4))
        s4_fp16_t = s4.to(torch.float16)
        scales4_parts.append(
            s4_fp16_t.numpy().astype(np.float16).tobytes())
        s4_rt = s4_fp16_t.to(torch.float32)
        dq4 = dequant_blocked(q4, s4_rt, BLOCK, n_real).reshape(D.shape)
        relerr4_num += float(((dq4 - D) ** 2).sum())
        relerr4_den += float((D ** 2).sum())

        n_blocks += s8.numel()  # same block count for int8 and int4

    bf16 = b"".join(bf16_parts)
    i8   = b"".join(int8_parts)
    i4   = b"".join(int4_parts)
    s8_all = b"".join(scales8_parts)
    s4_all = b"".join(scales4_parts)

    print(f"  {n_params:,} params  {n_blocks:,} blocks (block={BLOCK})")
    print(f"  scale overhead bytes: int8 {len(s8_all):,}  int4 {len(s4_all):,}  "
          f"(2 B/block)")

    t = time.time(); br_bf16 = brotli_bytes(bf16); t_bf = time.time() - t
    print(f"  br-11 bf16      {br_bf16:,} B ({t_bf:.0f}s)")
    t = time.time(); br_i8 = brotli_bytes(i8 + s8_all); t_i8 = time.time() - t
    print(f"  br-11 int8+sc   {br_i8:,} B ({t_i8:.0f}s)")
    t = time.time(); br_i4 = brotli_bytes(i4 + s4_all); t_i4 = time.time() - t
    print(f"  br-11 int4+sc   {br_i4:,} B ({t_i4:.0f}s)")

    relerr8 = (relerr8_num / relerr8_den) ** 0.5
    relerr4 = (relerr4_num / relerr4_den) ** 0.5

    out = {
        "claim": 21,
        "wave": 47,
        "experiment": "mxfp4_block_float_delta",
        "block_size": BLOCK,
        "base_repo": args.base,
        "ft_repo":   args.ft,
        "n_params": n_params,
        "n_blocks": n_blocks,
        "sha256": {
            "bf16":  hashlib.sha256(bf16).hexdigest(),
            "int8":  hashlib.sha256(i8 + s8_all).hexdigest(),
            "int4":  hashlib.sha256(i4 + s4_all).hexdigest(),
        },
        "raw_bytes": {
            "bf16":  len(bf16),
            "int8+block_scales_fp16":  len(i8) + len(s8_all),
            "int4+block_scales_fp16":  len(i4) + len(s4_all),
            "block_scales_overhead_int8": len(s8_all),
            "block_scales_overhead_int4": len(s4_all),
        },
        "brotli_11_bytes": {
            "bf16":  br_bf16,
            "int8+block_scales_fp16":  br_i8,
            "int4+block_scales_fp16":  br_i4,
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
    print(f"    br(bf16 Δ)/br(int8 block+sc) = {out['ratios_vs_brotli11_bf16_delta']['int8']:.3f}x "
          f"(relerr {relerr8:.4e})")
    print(f"    br(bf16 Δ)/br(int4 block+sc) = {out['ratios_vs_brotli11_bf16_delta']['int4']:.3f}x "
          f"(relerr {relerr4:.4e})")

    # Atomic JSON write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
