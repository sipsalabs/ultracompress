"""UltraCompress v0.2 — `uc pack` module.

Converts dense bf16 e2e layer artifacts (`_e2e_*/layer_NNN.pt`) to packed
.uc binary format. Method internals are patent-protected
(USPTO 64/049,511 + 64/049,517) — only the binary serialization the reader
needs is documented in this module. Net storage shrink: ~3.2× on the dominant
weight term vs bf16.

Format: see docs/UC_PACK_V0_2_DESIGN.md.

Usage:
    uc pack <e2e_dir>/ <output.uc>/

Or programmatic:
    from ultracompress.pack import pack_e2e_dir
    pack_e2e_dir("scripts/overlay/_e2e_qwen3_8b", "qwen3-8b.uc")
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Match the streaming runner's TARGET_SUBS
TARGET_SUBS = (
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
    'w1', 'w2', 'w3',  # Mixtral / Phi-MoE expert linear naming
    'in_proj', 'x_proj', 'dt_proj', 'out_proj',  # Mamba / state-space-model SSM Linears
)

UC_MAGIC = b'UCL\x00'
UC_VERSION = 2


def _is_target_linear(name: str) -> bool:
    """Return True if this Linear name should have been quantized.

    Substring match against TARGET_SUBS (matches Qwen/Mistral/Llama/Mixtral/Phi-MoE
    naming conventions used by the streaming runner).
    """
    return any(s in name for s in TARGET_SUBS)


def _bitpack(int_codes: np.ndarray, bpw: int) -> np.ndarray:
    """Pack int codes (int8 in valid bpw range) into bytes.

    Vectorized via np.packbits — each weight contributes `bpw` bits stored
    LSB-first within the bit stream. Bit 0 of byte 0 is LSB of weight 0,
    bit 1 of byte 0 is bit 1 of weight 0, etc.
    For 5 bits per weight: 8 weights occupy 5 bytes (40 bits exactly).

    Returns a flat uint8 array.
    """
    if bpw <= 0 or bpw > 8:
        raise ValueError(f"bpw must be in 1..8, got {bpw}")
    flat = int_codes.astype(np.int16).flatten()
    bias = 1 << (bpw - 1)
    unsigned = (flat + bias).astype(np.uint16)
    if unsigned.size and unsigned.max() >= (1 << bpw):
        raise ValueError(f"Code {unsigned.max()} exceeds {bpw}-bit range")
    n_total_bits = unsigned.size * bpw
    n_bytes = (n_total_bits + 7) // 8

    # Expand each weight to its bpw bits, LSB-first within the weight.
    # Result shape: (n_weights, bpw), dtype=uint8 0/1
    bit_indices = np.arange(bpw, dtype=np.uint16)
    bits = ((unsigned[:, None] >> bit_indices) & 1).astype(np.uint8)
    bits_flat = bits.reshape(-1)

    # Pad to multiple of 8 so np.packbits returns exactly n_bytes
    pad = (8 - (n_total_bits % 8)) % 8
    if pad:
        bits_flat = np.concatenate([bits_flat, np.zeros(pad, dtype=np.uint8)])

    out = np.packbits(bits_flat, bitorder='little')
    return out[:n_bytes]


def _bitunpack(packed: np.ndarray, n_weights: int, bpw: int) -> np.ndarray:
    """Inverse of `_bitpack`. Returns int8 codes in signed range.

    Vectorized via np.unpackbits — bit 0 of byte 0 maps to bit 0 of weight 0,
    matching the pack layout (LSB-first, sequential weights).
    """
    if bpw <= 0 or bpw > 8:
        raise ValueError(f"bpw must be in 1..8, got {bpw}")
    bias = 1 << (bpw - 1)
    bits = np.unpackbits(packed.astype(np.uint8), bitorder='little')
    n_needed = n_weights * bpw
    bits = bits[:n_needed].reshape(n_weights, bpw)
    bit_indices = np.arange(bpw, dtype=np.uint16)
    weights_pow2 = (1 << bit_indices).astype(np.uint16)
    codes_unsigned = (bits.astype(np.uint16) * weights_pow2).sum(axis=1)
    codes_signed = codes_unsigned.astype(np.int16) - bias
    return codes_signed.astype(np.int8)


def _inverse_quantize(W_dequant: torch.Tensor, bpw: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Legacy v2 inverse path used only when no codec state was persisted.

    Codec internals are patent-protected (USPTO 64/049,511 + 64/049,517).
    Production v3 packs always read codec state directly and never go through
    this path; it is retained only for backward compatibility with the older
    layer.pt format.

    Returns:
        codes: int8 array, shape (n_rows, n_cols)
        scales: float32 array, shape (n_rows, n_blocks)
    """
    W = W_dequant.detach().float().cpu().numpy()
    n_rows, n_cols = W.shape
    n_blocks = (n_cols + block_size - 1) // block_size
    qmax = (1 << (bpw - 1)) - 1

    codes = np.zeros((n_rows, n_cols), dtype=np.int8)
    scales = np.zeros((n_rows, n_blocks), dtype=np.float32)

    for b in range(n_blocks):
        c0 = b * block_size
        c1 = min(c0 + block_size, n_cols)
        block = W[:, c0:c1]  # (n_rows, block_width)
        max_abs = np.abs(block).max(axis=1)  # (n_rows,)
        scale = np.where(max_abs > 0, max_abs / qmax, 1.0).astype(np.float32)
        scales[:, b] = scale
        block_codes = np.round(block / scale[:, None])
        block_codes = np.clip(block_codes, -qmax, qmax).astype(np.int8)
        codes[:, c0:c1] = block_codes

    return codes, scales


_DTYPE_TAGS = {torch.bfloat16: 0, torch.float32: 1, torch.float16: 2}
_TAG_DTYPES = {v: k for k, v in _DTYPE_TAGS.items()}


def _serialize_extra(name: str, tensor: torch.Tensor) -> bytes:
    """Serialize a non-quantized layer tensor (e.g., norm) to bytes.

    Format: name_len(u16) + name + n_dims(u8) + dims(u32 each) + dtype_tag(u8) + raw_bytes
    Raw bytes use the tensor's native dtype (bf16/fp32/fp16). If dtype is not
    one of the supported ones, store as fp32.
    """
    dtype = tensor.dtype if tensor.dtype in _DTYPE_TAGS else torch.float32
    t = tensor.detach().to(dtype).cpu().contiguous()
    name_b = name.encode('utf-8')
    parts: list[bytes] = [
        struct.pack('<H', len(name_b)),
        name_b,
        struct.pack('<B', t.dim()),
    ]
    for dim in t.shape:
        parts.append(struct.pack('<I', int(dim)))
    parts.append(struct.pack('<B', _DTYPE_TAGS[dtype]))
    if dtype in (torch.bfloat16, torch.float16):
        parts.append(t.view(torch.uint16).numpy().tobytes())
    else:  # fp32
        parts.append(t.numpy().tobytes())
    return b''.join(parts)


def pack_layer(layer_pt_path: Path, out_uc_path: Path, layer_idx: int,
               bpw: int = 5, block_size: int = 64) -> dict[str, Any]:
    """Pack a single layer.pt into a layer.uc binary blob.

    Returns metadata about what was packed.
    """
    state = torch.load(str(layer_pt_path), weights_only=False, map_location='cpu')
    sd = state['state_dict']
    rank = state.get('rank', 32)

    # Group tensors by base Linear name. The state_dict has keys like:
    #   self_attn.q_proj.alpha
    #   self_attn.q_proj.W_base
    #   self_attn.q_proj.V.weight
    #   self_attn.q_proj.U.weight
    #   self_attn.q_proj.bias  (optional)
    # We need to extract the base ("self_attn.q_proj") and the leaf semantic
    # ("alpha", "W_base", "V.weight", "U.weight", "bias").
    KNOWN_SUFFIXES = ('.alpha', '.W_base', '.V.weight', '.U.weight', '.bias')
    linears: dict[str, dict[str, torch.Tensor]] = {}
    extras: dict[str, torch.Tensor] = {}  # non-quantized tensors (norms, etc.)
    for name, tensor in sd.items():
        if _is_target_linear(name):
            sub = None
            base_name = None
            for suffix in KNOWN_SUFFIXES:
                if name.endswith(suffix):
                    sub = suffix.lstrip('.')
                    base_name = name[:-len(suffix)]
                    break
            if sub is None or base_name is None:
                # Target Linear with unknown leaf (e.g., gate router weight) — store as extra
                extras[name] = tensor
                continue
            linears.setdefault(base_name, {})[sub] = tensor
        else:
            # Norms, layer-level routers, anything not a quantized Linear leaf
            extras[name] = tensor

    n_linears = len(linears)
    if n_linears == 0:
        raise ValueError(f"No target linears found in {layer_pt_path}")

    blobs: list[bytes] = []
    metadata: list[dict[str, Any]] = []

    for base_name, parts in sorted(linears.items()):
        if 'W_base' not in parts or 'alpha' not in parts:
            continue  # not a quantized linear (e.g., gate router)
        W_base = parts['W_base']
        alpha_t = parts['alpha']
        V_t = parts.get('V.weight') if 'V.weight' in parts else parts.get('V_weight')
        U_t = parts.get('U.weight') if 'U.weight' in parts else parts.get('U_weight')
        bias_t = parts.get('bias', None)

        if V_t is None or U_t is None:
            continue  # no correction overlay

        out_dim, in_dim = W_base.shape

        # Re-derive int codes + scales from the dequantized W_base
        codes, scales = _inverse_quantize(W_base, bpw, block_size)
        packed = _bitpack(codes, bpw)

        # alpha + scales + bias remain bf16 (negligible precision loss)
        alpha_bytes = alpha_t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().tobytes()
        scales_bytes = torch.from_numpy(scales).to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
        bias_bytes = bias_t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().tobytes() if bias_t is not None else b''
        # correction overlay matrices are distillation-trained — preserve fp32 to avoid PPL regression.
        # V/U total <2 MB per layer at production rank; trivial vs bf16 (~1% format overhead).
        V_bytes = V_t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        U_bytes = U_t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()

        name_b = base_name.encode('utf-8')
        # Per-Linear header — bumped to include vu_dtype tag (1 = fp32, 0 = bf16 legacy)
        hdr = (
            struct.pack('<H', len(name_b)) + name_b +
            struct.pack('<II', out_dim, in_dim) +
            struct.pack('<HBB', block_size, bpw, rank) +
            struct.pack('<B', 1)  # vu_dtype: 1 = fp32 (default), 0 = bf16 legacy
        )
        rec = hdr + alpha_bytes + packed.tobytes() + scales_bytes + V_bytes + U_bytes
        rec += struct.pack('<B', 1 if bias_t is not None else 0)
        rec += bias_bytes

        blobs.append(rec)
        metadata.append({
            'name': base_name,
            'out_dim': int(out_dim),
            'in_dim': int(in_dim),
            'packed_bytes': len(packed),
            'bpw': bpw,
            'rank': rank,
        })

    # Serialize extras (non-quantized layer tensors: norms, etc.)
    extra_blobs: list[bytes] = [_serialize_extra(name, t) for name, t in sorted(extras.items())]
    n_extras = len(extra_blobs)

    # File header (16 bytes total): MAGIC(4) + version(2) + layer_idx(2) + n_linears(2) +
    # n_extras(2) + reserved(4). Old packs (n_extras=0) decode unchanged.
    n_actual = len(blobs)
    file_hdr = UC_MAGIC + struct.pack('<HHHH', UC_VERSION, layer_idx, n_actual, n_extras) + b'\x00' * 4
    out_uc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_uc_path, 'wb') as f:
        f.write(file_hdr)
        for b in blobs:
            f.write(b)
        for b in extra_blobs:
            f.write(b)

    return {
        'layer_idx': layer_idx,
        'n_linears_packed': n_actual,
        'n_extras_packed': n_extras,
        'output_size_bytes': out_uc_path.stat().st_size,
        'linears': metadata,
        'extras': sorted(extras.keys()),
    }


def pack_e2e_dir(e2e_dir: str | Path, out_dir: str | Path,
                 bpw: int = 5, block_size: int = 64) -> dict[str, Any]:
    """Pack an entire `_e2e_*` directory into a `.uc` directory.

    Args:
        e2e_dir: source directory containing layer_NNN.pt files
        out_dir: target directory for layer_NNN.uc + manifest.json
        bpw: bits per weight (typically matches what was trained — default 5)
        block_size: per-block scaling group size (typically 64 or 128)
    """
    e2e_dir = Path(e2e_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_pts = sorted(e2e_dir.glob('layer_*.pt'))
    if not layer_pts:
        raise ValueError(f"No layer_*.pt files in {e2e_dir}")

    layer_meta: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    for layer_pt in layer_pts:
        layer_idx = int(layer_pt.stem.split('_')[-1])
        out_uc = out_dir / f'layer_{layer_idx:03d}.uc'
        meta = pack_layer(layer_pt, out_uc, layer_idx, bpw=bpw, block_size=block_size)
        in_size = layer_pt.stat().st_size
        out_size = meta['output_size_bytes']
        meta['input_size_bytes'] = in_size
        meta['shrink_ratio'] = in_size / max(out_size, 1)
        layer_meta.append(meta)
        total_in += in_size
        total_out += out_size
        print(f"  layer {layer_idx:03d}: {in_size/1e6:.1f}MB -> {out_size/1e6:.1f}MB ({in_size/out_size:.2f}x shrink)")

    manifest = {
        'format': 'uc-pack-v1',
        'uc_pack_version': UC_VERSION,  # binary file-header version (1=bf16 V/U legacy, 2+=fp32 V/U + extras)
        'vu_dtype': 'fp32' if UC_VERSION >= 2 else 'bf16',
        'bpw': bpw,
        'block_size': block_size,
        'n_layers': len(layer_pts),
        'total_input_bytes': total_in,
        'total_output_bytes': total_out,
        'overall_shrink_ratio': total_in / max(total_out, 1),
        'layers': layer_meta,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\nPACK COMPLETE")
    print(f"  total: {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB ({total_in/total_out:.2f}x shrink)")
    return manifest


def cmd_pack(args) -> int:
    """`uc pack <e2e_dir> <out_dir>` CLI entry."""
    src = Path(args.src)
    dst = Path(args.dst)
    bpw = args.bpw
    block_size = args.block_size

    print(f"Packing {src} -> {dst} at {bpw} bpw, block_size={block_size}")
    manifest = pack_e2e_dir(src, dst, bpw=bpw, block_size=block_size)
    print(f"\nManifest: {dst / 'manifest.json'}")
    print(f"Overall shrink ratio: {manifest['overall_shrink_ratio']:.2f}x")
    return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Pack e2e dense layer artifacts to .uc binary")
    ap.add_argument('src', help='Source _e2e_* directory')
    ap.add_argument('dst', help='Output .uc directory')
    ap.add_argument('--bpw', type=int, default=5)
    ap.add_argument('--block-size', type=int, default=64)
    args = ap.parse_args()
    raise SystemExit(cmd_pack(args))
