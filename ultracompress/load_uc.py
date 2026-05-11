"""UltraCompress v0.2 — `uc load` module.

Loads a packed `.uc` directory back into a working transformers model.

Pipeline:
  packed/layer_NNN.uc  ──► UCLayerLoader.load_layer_state_dict ──► state_dict ──► HF model

The packed format stores only the quantized Linears + correction overlays. The
non-target tensors (norms, gates, biases of non-target Linears) come from the
ORIGINAL HF model that the user provides.

Usage:
    from ultracompress.load_uc import load_packed_model
    model = load_packed_model(
        packed_dir="qwen3-1.7b.uc",
        base_hf_id="Qwen/Qwen3-1.7B",
        device="cuda:0",
    )
    out = model.generate(input_ids, max_new_tokens=64)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .pack import UC_MAGIC, UC_VERSION, _bitunpack, _TAG_DTYPES


def _read_struct(buf: bytes, offset: int, fmt: str) -> tuple[Any, int]:
    """Read a struct.pack-formatted slice from buf starting at offset.
    Returns (parsed, new_offset)."""
    n = struct.calcsize(fmt)
    val = struct.unpack(fmt, buf[offset:offset + n])
    return val, offset + n


def _read_bf16_tensor(buf: bytes, offset: int, n_elements: int) -> tuple[torch.Tensor, int]:
    """Read n_elements of bf16 from buf starting at offset."""
    n_bytes = n_elements * 2
    raw = buf[offset:offset + n_bytes]
    arr = np.frombuffer(raw, dtype=np.uint16).copy()
    t = torch.from_numpy(arr).view(torch.bfloat16)
    return t, offset + n_bytes


def parse_uc_layer(uc_path: Path) -> dict[str, dict[str, Any]]:
    """Parse a single layer.uc binary file.

    Returns:
      {linear_name: {alpha, W_base_dequant, V, U, bias?, out_dim, in_dim, ...}}
    """
    buf = uc_path.read_bytes()
    offset = 0

    magic = buf[offset:offset + 4]
    offset += 4
    if magic != UC_MAGIC:
        raise ValueError(f"Bad magic in {uc_path}: {magic!r}")

    (version, layer_idx, n_linears, n_extras), offset = _read_struct(buf, offset, '<HHHH')
    offset += 4  # reserved padding (was 6 in v1; reused 2 for n_extras with all-zero default)
    if version not in (1, 2):
        raise ValueError(f"Unsupported pack version {version} in {uc_path}")

    out: dict[str, dict[str, Any]] = {}
    for _ in range(n_linears):
        (name_len,), offset = _read_struct(buf, offset, '<H')
        name = buf[offset:offset + name_len].decode('utf-8')
        offset += name_len

        (out_dim, in_dim), offset = _read_struct(buf, offset, '<II')
        (block_size, bpw, rank), offset = _read_struct(buf, offset, '<HBB')

        # vu_dtype tag — only present in v2+ (v1 had no tag and stored V/U as bf16)
        if version >= 2:
            (vu_dtype_tag,), offset = _read_struct(buf, offset, '<B')
            vu_is_fp32 = (vu_dtype_tag == 1)
        else:
            vu_is_fp32 = False  # v1 = bf16 V/U

        # alpha (1 bf16 element)
        alpha_t, offset = _read_bf16_tensor(buf, offset, 1)
        alpha = alpha_t.float().item()

        # packed W_base
        n_weights = out_dim * in_dim
        n_packed_bytes = (n_weights * bpw + 7) // 8
        packed = np.frombuffer(buf[offset:offset + n_packed_bytes], dtype=np.uint8).copy()
        offset += n_packed_bytes

        # per-block scales: (out_dim, n_blocks) bf16
        n_blocks = (in_dim + block_size - 1) // block_size
        scales_t, offset = _read_bf16_tensor(buf, offset, out_dim * n_blocks)
        scales = scales_t.view(out_dim, n_blocks)

        if vu_is_fp32:
            # V (rank × in_dim) fp32
            V_arr = np.frombuffer(buf[offset:offset + rank * in_dim * 4], dtype=np.float32).copy()
            V = torch.from_numpy(V_arr).view(rank, in_dim)
            offset += rank * in_dim * 4
            # U (out_dim × rank) fp32
            U_arr = np.frombuffer(buf[offset:offset + out_dim * rank * 4], dtype=np.float32).copy()
            U = torch.from_numpy(U_arr).view(out_dim, rank)
            offset += out_dim * rank * 4
        else:
            # Legacy: V/U bf16
            V_t, offset = _read_bf16_tensor(buf, offset, rank * in_dim)
            V = V_t.view(rank, in_dim)
            U_t, offset = _read_bf16_tensor(buf, offset, out_dim * rank)
            U = U_t.view(out_dim, rank)

        # bias_present + bias
        (bias_present,), offset = _read_struct(buf, offset, '<B')
        bias = None
        if bias_present:
            bias_t, offset = _read_bf16_tensor(buf, offset, out_dim)
            bias = bias_t

        # Dequantize W_base from packed codes
        codes = _bitunpack(packed, n_weights, bpw).reshape(out_dim, in_dim)
        # Apply per-block scales
        W_base = np.zeros((out_dim, in_dim), dtype=np.float32)
        for b in range(n_blocks):
            c0 = b * block_size
            c1 = min(c0 + block_size, in_dim)
            W_base[:, c0:c1] = codes[:, c0:c1].astype(np.float32) * scales[:, b].float().numpy()[:, None]
        W_base_t = torch.from_numpy(W_base).to(torch.bfloat16)

        out[name] = {
            'alpha': alpha,
            'W_base': W_base_t,
            'V': V,
            'U': U,
            'bias': bias,
            'out_dim': out_dim,
            'in_dim': in_dim,
            'block_size': block_size,
            'bpw': bpw,
            'rank': rank,
        }

    # Extras (non-quantized layer tensors: norms, etc.)
    extras: dict[str, torch.Tensor] = {}
    for _ in range(n_extras):
        (name_len,), offset = _read_struct(buf, offset, '<H')
        name = buf[offset:offset + name_len].decode('utf-8')
        offset += name_len
        (n_dims,), offset = _read_struct(buf, offset, '<B')
        dims: list[int] = []
        for __ in range(n_dims):
            (d,), offset = _read_struct(buf, offset, '<I')
            dims.append(d)
        (dtype_tag,), offset = _read_struct(buf, offset, '<B')
        dtype = _TAG_DTYPES[dtype_tag]
        n_elems = 1
        for d in dims:
            n_elems *= d
        if dtype == torch.float32:
            n_bytes = n_elems * 4
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.float32).copy()
            t = torch.from_numpy(arr).view(*dims) if dims else torch.from_numpy(arr).reshape(())
        else:  # bf16 / fp16 — 2 bytes each
            n_bytes = n_elems * 2
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.uint16).copy()
            t = torch.from_numpy(arr).view(dtype).view(*dims) if dims else torch.from_numpy(arr).view(dtype).reshape(())
        offset += n_bytes
        extras[name] = t
    out['__extras__'] = extras  # special key holding raw tensors keyed by full state_dict name

    return out


def reconstruct_layer_state_dict(uc_path: Path) -> dict[str, torch.Tensor]:
    """Convert a parsed .uc layer back into the e2e state_dict format.

    The per-layer state_dict matches what `_e2e_*/layer_NNN.pt` would have:
      self_attn.q_proj.alpha, self_attn.q_proj.W_base, self_attn.q_proj.V.weight,
      self_attn.q_proj.U.weight, [self_attn.q_proj.bias], etc.

    This makes it trivial to plug the loader into the existing
    eval_compressed_only.py path without changes.
    """
    parsed = parse_uc_layer(uc_path)
    extras = parsed.pop('__extras__', {})
    sd: dict[str, torch.Tensor] = {}
    for base_name, parts in parsed.items():
        sd[f'{base_name}.alpha'] = torch.tensor([parts['alpha']], dtype=torch.bfloat16)
        # The pack layer multiplies W_base by alpha implicitly via the scale-times-code dequant.
        # alpha here is the global scale we stored separately. To match e2e format exactly,
        # we store W_base directly (dequantized) — alpha rides as a scalar multiplier elsewhere.
        sd[f'{base_name}.W_base'] = parts['W_base']
        sd[f'{base_name}.V.weight'] = parts['V']
        sd[f'{base_name}.U.weight'] = parts['U']
        if parts['bias'] is not None:
            sd[f'{base_name}.bias'] = parts['bias']
    # Extras (norms, layer-level tensors not part of any quantized linear) get added
    # under their original full state_dict key so the layer can load_state_dict cleanly.
    for k, v in extras.items():
        sd[k] = v
    return sd


def load_uc_dir(packed_dir: str | Path) -> dict[int, dict[str, torch.Tensor]]:
    """Load all layer.uc files from a packed directory into per-layer state dicts.

    Returns {layer_idx: state_dict}.
    """
    packed_dir = Path(packed_dir)
    manifest_path = packed_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        n_layers = manifest['n_layers']
    else:
        n_layers = len(list(packed_dir.glob('layer_*.uc')))

    out: dict[int, dict[str, torch.Tensor]] = {}
    for i in range(n_layers):
        uc_path = packed_dir / f'layer_{i:03d}.uc'
        if not uc_path.exists():
            print(f"WARN: missing {uc_path}, skipping")
            continue
        out[i] = reconstruct_layer_state_dict(uc_path)
    return out


def cmd_load(args) -> int:
    """`uc load <packed_dir>` CLI entry — loads + prints manifest."""
    packed = Path(args.packed_dir)
    print(f"Loading {packed}...")
    layer_sds = load_uc_dir(packed)
    n_layers = len(layer_sds)
    n_keys_total = sum(len(sd) for sd in layer_sds.values())
    print(f"\nLoaded {n_layers} layers, {n_keys_total} state-dict keys total")
    if args.layer is not None:
        print(f"\nLayer {args.layer} keys:")
        for k, v in layer_sds[args.layer].items():
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Load packed .uc directory")
    ap.add_argument('packed_dir')
    ap.add_argument('--layer', type=int, default=None, help='Print details of one layer')
    args = ap.parse_args()
    raise SystemExit(cmd_load(args))
