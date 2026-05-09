"""UltraCompress v0.3 pack format — LOSSLESS round-trip via trainer-persisted codec.

Key difference from v0.2:
  - v0.2: reverse-derives (scale, code) from bf16 W_base assuming uniform symmetric
    grid {-15..15}/15. Lossy because trainer used k-means LEARNED grid.
  - v0.3: reads (grid, codes, absmax) directly from `state['gsq_codecs']` (added
    by `streaming_compression_runner.py:compress_single_layer` 2026-05-07 PM).

Format (per-Linear blob, version 3):
  name_len(u16) + name(utf-8) +
  out_dim(u32) + in_dim(u32) +
  block_size(u16) + bpw(u8) + rank(u8) + grid_K(u16) +     # 12 bytes after dims
  alpha(bf16, 2 bytes) +
  grid(fp32, K * 4 bytes) +                                  # K = 32 for 5bpw
  absmax(fp32, out_dim * n_blocks * 4 bytes) +
  packed_codes(ceil(n_weights * bpw / 8) bytes) +
  V(fp32, rank * in_dim * 4 bytes) +
  U(fp32, out_dim * rank * 4 bytes) +
  bias_present(u8) + bias(bf16, 2 * out_dim if present)

File header (16 bytes total, same as v2):
  MAGIC(4 bytes "UCL\\0") + version(u16, =3) + layer_idx(u16) +
  n_linears(u16) + n_extras(u16) + reserved(4 bytes)

Then n_linears blobs followed by n_extras (norms etc, same as v2 extras).

Reconstruction: W_base = (absmax × grid[codes]).reshape(out_dim, in_dim)
                  → bit-identical to source W_base (fp32 ops; cast to bf16 at end if needed).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .pack import (
    UC_MAGIC,
    TARGET_SUBS,
    _bitpack,
    _bitunpack,
    _is_target_linear,
    _serialize_extra,
    _DTYPE_TAGS,
    _TAG_DTYPES,
)
from .aux_pack import (
    DEFAULT_AUX_KEYS,
    collect_aux_tensors_from_model,
    write_aux_weights,
)

UC_VERSION_V3 = 3
PACK_FORMAT_VERSION_V3 = "3.0"
PACK_FORMAT_VERSION_V3_5 = "3.5"  # adds self-contained aux_weights.uc


def pack_layer_v3(layer_pt_path: Path, out_uc_path: Path, layer_idx: int,
                  bpw: int = 5, block_size: int = 64) -> dict[str, Any]:
    """Pack a layer.pt that has trainer-persisted `gsq_codecs` into a v3 .uc file.

    Returns metadata about what was packed. Raises ValueError if state lacks `gsq_codecs`.
    """
    state = torch.load(str(layer_pt_path), weights_only=False, map_location='cpu')
    sd = state['state_dict']
    rank = state.get('rank', 32)
    codecs = state.get('gsq_codecs', {})
    if not codecs:
        raise ValueError(f"layer.pt at {layer_pt_path} has no gsq_codecs — re-compress with v0.3 trainer")

    # Group state_dict tensors by base Linear name (same as v2)
    KNOWN_SUFFIXES = ('.alpha', '.W_base', '.V.weight', '.U.weight', '.bias')
    linears: dict[str, dict[str, torch.Tensor]] = {}
    extras: dict[str, torch.Tensor] = {}
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
                extras[name] = tensor
                continue
            linears.setdefault(base_name, {})[sub] = tensor
        else:
            extras[name] = tensor

    blobs: list[bytes] = []
    metadata: list[dict[str, Any]] = []

    for base_name, parts in sorted(linears.items()):
        if 'W_base' not in parts or 'alpha' not in parts:
            continue
        if base_name not in codecs:
            # Trainer didn't persist codec for this Linear — fail loud rather than silently lossy
            raise ValueError(f"linear {base_name} has W_base but no gsq_codec entry")
        codec = codecs[base_name]
        grid = codec['grid']        # (K,) fp32
        codes = codec['codes']      # (out_dim, n_blocks, block) int16
        absmax = codec['absmax']    # (out_dim, n_blocks) fp32

        W_base = parts['W_base']
        alpha_t = parts['alpha']
        V_t = parts.get('V.weight') if 'V.weight' in parts else parts.get('V_weight')
        U_t = parts.get('U.weight') if 'U.weight' in parts else parts.get('U_weight')
        bias_t = parts.get('bias', None)
        if V_t is None or U_t is None:
            continue

        out_dim, in_dim = W_base.shape
        K = grid.shape[0]
        # Sanity: codes shape should align
        n_blocks = codes.shape[1]
        block_w = codes.shape[2]
        assert codes.shape[0] == out_dim
        assert n_blocks * block_w == in_dim, f"codes shape {codes.shape} != ({out_dim}, {in_dim/block_w}, {block_w})"

        # Bit-pack codes (5-bit unsigned, in [0, K))
        codes_flat = codes.flatten().to(torch.int16)
        # Validate range
        if codes_flat.max() >= K or codes_flat.min() < 0:
            raise ValueError(f"codes out of range [0,{K}): min={codes_flat.min()} max={codes_flat.max()}")
        # _bitpack expects signed int codes in [-(K/2), K/2-1]; we have unsigned [0, K-1].
        # Shift to signed range so _bitpack's bias mapping reproduces the original index.
        # Actually _bitpack uses bias = 1<<(bpw-1) and stores unsigned = code + bias.
        # If we feed signed_code = unsigned_code - bias, then unsigned = signed + bias = original. Works.
        bias = 1 << (bpw - 1)
        signed_codes = (codes_flat.numpy().astype(np.int16) - bias).astype(np.int8)
        packed = _bitpack(signed_codes, bpw)

        # Serialize tensors
        # alpha as fp32 (single value per linear; v0.3 fix — bf16 lost precision -> +13% PPL)
        alpha_bytes = alpha_t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        grid_bytes = grid.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        absmax_bytes = absmax.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        V_bytes = V_t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        U_bytes = U_t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        bias_bytes = bias_t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().tobytes() if bias_t is not None else b''

        name_b = base_name.encode('utf-8')
        # Per-Linear header (note: includes grid_K vs v2)
        hdr = (
            struct.pack('<H', len(name_b)) + name_b +
            struct.pack('<II', out_dim, in_dim) +
            struct.pack('<HBB', block_size, bpw, rank) +
            struct.pack('<H', K)
        )
        rec = hdr + alpha_bytes + grid_bytes + absmax_bytes + packed.tobytes() + V_bytes + U_bytes
        rec += struct.pack('<B', 1 if bias_t is not None else 0)
        rec += bias_bytes

        blobs.append(rec)
        metadata.append({
            'name': base_name,
            'out_dim': int(out_dim),
            'in_dim': int(in_dim),
            'K': int(K),
            'packed_bytes': len(packed),
            'bpw': bpw,
            'rank': rank,
        })

    # Serialize extras
    extra_blobs = [_serialize_extra(name, t) for name, t in sorted(extras.items())]
    n_extras = len(extra_blobs)

    n_actual = len(blobs)
    file_hdr = UC_MAGIC + struct.pack('<HHHH', UC_VERSION_V3, layer_idx, n_actual, n_extras) + b'\x00' * 4
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
        'uc_version': UC_VERSION_V3,
    }


def parse_uc_layer_v3(uc_path: Path) -> dict[str, Any]:
    """Parse a v3 .uc layer file.

    Returns:
      {linear_name: {grid, codes, absmax, V, U, alpha, bias?, out_dim, in_dim, ...}}
      with special key '__extras__' for non-quantized tensors and '__version__' = 3
    """
    buf = uc_path.read_bytes()
    offset = 0
    magic = buf[offset:offset + 4]; offset += 4
    if magic != UC_MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")
    (version, layer_idx, n_linears, n_extras) = struct.unpack('<HHHH', buf[offset:offset + 8])
    offset += 8
    offset += 4  # reserved
    if version != UC_VERSION_V3:
        raise ValueError(f"Expected v3 file, got version {version}")

    out: dict[str, Any] = {'__version__': version, '__layer_idx__': layer_idx}

    for _ in range(n_linears):
        (name_len,) = struct.unpack('<H', buf[offset:offset + 2]); offset += 2
        name = buf[offset:offset + name_len].decode('utf-8'); offset += name_len
        (out_dim, in_dim) = struct.unpack('<II', buf[offset:offset + 8]); offset += 8
        (block_size, bpw, rank) = struct.unpack('<HBB', buf[offset:offset + 4]); offset += 4
        (K,) = struct.unpack('<H', buf[offset:offset + 2]); offset += 2

        # alpha (1 fp32 — v0.3 fix; was bf16 which lost precision)
        alpha_arr = np.frombuffer(buf[offset:offset + 4], dtype=np.float32).copy()
        alpha = float(alpha_arr[0])
        offset += 4

        # grid (K fp32)
        grid_arr = np.frombuffer(buf[offset:offset + K * 4], dtype=np.float32).copy()
        grid = torch.from_numpy(grid_arr)
        offset += K * 4

        # absmax (out_dim * n_blocks fp32)
        n_blocks = (in_dim + block_size - 1) // block_size
        absmax_arr = np.frombuffer(buf[offset:offset + out_dim * n_blocks * 4], dtype=np.float32).copy()
        absmax = torch.from_numpy(absmax_arr).view(out_dim, n_blocks)
        offset += out_dim * n_blocks * 4

        # packed codes
        n_weights = out_dim * in_dim
        n_packed_bytes = (n_weights * bpw + 7) // 8
        packed = np.frombuffer(buf[offset:offset + n_packed_bytes], dtype=np.uint8).copy()
        offset += n_packed_bytes
        # Unpack: gives signed codes in [-(K/2), K/2-1]; shift back to unsigned [0, K-1]
        bias = 1 << (bpw - 1)
        codes_signed = _bitunpack(packed, n_weights, bpw)
        codes_unsigned = (codes_signed.astype(np.int16) + bias).astype(np.int16)
        codes = torch.from_numpy(codes_unsigned).reshape(out_dim, n_blocks, block_size)

        # V (rank × in_dim, fp32)
        V_arr = np.frombuffer(buf[offset:offset + rank * in_dim * 4], dtype=np.float32).copy()
        V = torch.from_numpy(V_arr).view(rank, in_dim)
        offset += rank * in_dim * 4
        # U (out_dim × rank, fp32)
        U_arr = np.frombuffer(buf[offset:offset + out_dim * rank * 4], dtype=np.float32).copy()
        U = torch.from_numpy(U_arr).view(out_dim, rank)
        offset += out_dim * rank * 4

        # bias
        (bias_present,) = struct.unpack('<B', buf[offset:offset + 1]); offset += 1
        bias_t = None
        if bias_present:
            bias_arr = np.frombuffer(buf[offset:offset + out_dim * 2], dtype=np.uint16).copy()
            bias_t = torch.from_numpy(bias_arr).view(torch.bfloat16)
            offset += out_dim * 2

        # Reconstruct W_base = (absmax × grid[codes]).reshape(out_dim, in_dim)
        # codes shape: (out_dim, n_blocks, block); grid[codes] same shape
        # absmax shape: (out_dim, n_blocks) → broadcast on last dim with [:,:,None]
        W_base = (grid[codes.long()] * absmax.unsqueeze(-1)).reshape(out_dim, in_dim).to(torch.bfloat16)

        out[name] = {
            'alpha': alpha,
            'W_base': W_base,
            'V': V,
            'U': U,
            'bias': bias_t,
            'grid': grid,
            'codes': codes,
            'absmax': absmax,
            'out_dim': out_dim,
            'in_dim': in_dim,
            'block_size': block_size,
            'bpw': bpw,
            'rank': rank,
            'K': K,
        }

    # Extras (norms etc — same format as v2)
    extras: dict[str, torch.Tensor] = {}
    for _ in range(n_extras):
        (name_len,) = struct.unpack('<H', buf[offset:offset + 2]); offset += 2
        name = buf[offset:offset + name_len].decode('utf-8'); offset += name_len
        (n_dims,) = struct.unpack('<B', buf[offset:offset + 1]); offset += 1
        dims = []
        for __ in range(n_dims):
            (d,) = struct.unpack('<I', buf[offset:offset + 4]); offset += 4
            dims.append(d)
        (dtype_tag,) = struct.unpack('<B', buf[offset:offset + 1]); offset += 1
        dtype = _TAG_DTYPES[dtype_tag]
        n_elems = 1
        for d in dims:
            n_elems *= d
        if dtype == torch.float32:
            n_bytes = n_elems * 4
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.float32).copy()
            t = torch.from_numpy(arr).view(*dims) if dims else torch.from_numpy(arr).reshape(())
        else:
            n_bytes = n_elems * 2
            arr = np.frombuffer(buf[offset:offset + n_bytes], dtype=np.uint16).copy()
            t = torch.from_numpy(arr).view(dtype).view(*dims) if dims else torch.from_numpy(arr).view(dtype).reshape(())
        offset += n_bytes
        extras[name] = t
    out['__extras__'] = extras
    return out


def reconstruct_layer_state_dict_v3(uc_path: Path) -> dict[str, torch.Tensor]:
    """Convert a parsed v3 .uc layer back into the e2e state_dict format.

    Same key layout as v2 reconstruct_layer_state_dict so eval_compressed_only.py
    can load it without modification.
    """
    parsed = parse_uc_layer_v3(uc_path)
    extras = parsed.pop('__extras__', {})
    parsed.pop('__version__', None)
    parsed.pop('__layer_idx__', None)
    sd: dict[str, torch.Tensor] = {}
    for base_name, parts in parsed.items():
        # alpha as fp32 (matches source state_dict dtype)
        sd[f'{base_name}.alpha'] = torch.tensor([parts['alpha']], dtype=torch.float32)
        sd[f'{base_name}.W_base'] = parts['W_base']
        sd[f'{base_name}.V.weight'] = parts['V']
        sd[f'{base_name}.U.weight'] = parts['U']
        if parts['bias'] is not None:
            sd[f'{base_name}.bias'] = parts['bias']
    for k, v in extras.items():
        sd[k] = v
    return sd


def pack_e2e_dir_v3(e2e_dir: str | Path, out_dir: str | Path,
                    bpw: int = 5, block_size: int = 64,
                    include_aux: bool = True,
                    base_hf_id: str | None = None,
                    aux_keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Pack an entire `_e2e_*` directory into v3 .uc files (lossless via gsq_codecs).

    Args:
        e2e_dir: Source directory containing layer_*.pt files.
        out_dir: Output dir for layer_*.uc + manifest.json + (optionally) aux_weights.uc.
        bpw: Bits per weight for the GSQ codes.
        block_size: GSQ per-block scale block size.
        include_aux: If True (default), also emit `aux_weights.uc` containing
            embed_tokens / model.norm / lm_head from the base HF model. This
            makes the pack self-contained — customer no longer needs to download
            the original safetensors from HF to reconstruct the model. If False,
            the legacy v3.0 layout is produced (smaller pack, but customer must
            also fetch the base bf16 weights for non-Linear tensors).
        base_hf_id: HF model id used to source the aux tensors when
            `include_aux=True`. If None, reads from `e2e_dir / "manifest.json"`
            via `base_model_hf_id` / `hf_id` keys.
        aux_keys: Override which model-level keys to pack into aux_weights.uc.
            Defaults to DEFAULT_AUX_KEYS (embed_tokens / norm / lm_head).
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
        meta = pack_layer_v3(layer_pt, out_uc, layer_idx, bpw=bpw, block_size=block_size)
        in_size = layer_pt.stat().st_size
        out_size = meta['output_size_bytes']
        meta['input_size_bytes'] = in_size
        meta['shrink_ratio'] = in_size / max(out_size, 1)
        layer_meta.append(meta)
        total_in += in_size
        total_out += out_size
        print(f"  layer {layer_idx:03d}: {in_size/1e6:.1f}MB -> {out_size/1e6:.1f}MB ({in_size/out_size:.2f}x shrink)", flush=True)

    manifest: dict[str, Any] = {
        'format': 'uc-pack-v1',  # legacy field for compat
        'uc_pack_version': UC_VERSION_V3,
        'pack_format_version': PACK_FORMAT_VERSION_V3,
        'vu_dtype': 'fp32',
        'codec_source': 'trainer-persisted',  # v3 marker — vs v2 'reverse-derived'
        'bpw': bpw,
        'block_size': block_size,
        'n_layers': len(layer_pts),
        'total_input_bytes': total_in,
        'total_output_bytes': total_out,
        'overall_shrink_ratio': total_in / max(total_out, 1),
        'layers': layer_meta,
    }

    if include_aux:
        aux_meta = _emit_aux_for_pack(
            e2e_dir=e2e_dir,
            out_dir=out_dir,
            base_hf_id=base_hf_id,
            aux_keys=aux_keys or DEFAULT_AUX_KEYS,
        )
        if aux_meta is not None:
            manifest['pack_format_version'] = PACK_FORMAT_VERSION_V3_5
            manifest['aux_file'] = aux_meta['path']
            manifest['aux_sha256'] = aux_meta['sha256']
            manifest['aux_size_bytes'] = aux_meta['size_bytes']
            manifest['aux_keys'] = aux_meta['keys']
            manifest['aux_n_tensors'] = aux_meta['n_tensors']
            manifest['aux_version'] = aux_meta['version']
            if base_hf_id:
                manifest['base_model_hf_id'] = base_hf_id

    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\nPACK V3 COMPLETE")
    print(f"  total: {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB ({total_in/total_out:.2f}x shrink)")
    if 'aux_file' in manifest:
        aux_mb = manifest['aux_size_bytes'] / 1e6
        print(f"  aux:   {manifest['aux_file']}  {aux_mb:.1f} MB  (sha256 {manifest['aux_sha256'][:16]}...)")
        print(f"         pack is SELF-CONTAINED — no base-model download needed at inference time")
    return manifest


def _emit_aux_for_pack(
    *, e2e_dir: Path, out_dir: Path,
    base_hf_id: str | None, aux_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    """Resolve the base HF id, load the model, dump aux weights to out_dir.

    Returns the aux metadata dict, or None if aux generation was skipped
    (e.g., no base id resolvable). Side effect: writes `out_dir/aux_weights.uc`.
    """
    # Resolve base HF id from arg or e2e dir's manifest.
    if base_hf_id is None:
        e2e_manifest = e2e_dir / 'manifest.json'
        if e2e_manifest.exists():
            try:
                m = json.loads(e2e_manifest.read_text(encoding='utf-8'))
                for key in ('base_model_hf_id', 'base_model', 'hf_id'):
                    val = m.get(key)
                    if isinstance(val, str) and val.strip():
                        base_hf_id = val.strip()
                        break
            except json.JSONDecodeError:
                pass
    if base_hf_id is None:
        print("  [aux] WARN: no base_hf_id resolvable; skipping aux_weights.uc "
              "(pass --base-model or set base_model_hf_id in source manifest.json)")
        return None

    print(f"  [aux] sourcing model-level tensors from {base_hf_id}")
    try:
        # Lazy-imported so the pack CLI works without transformers installed for
        # the legacy `--no-aux` path.
        from transformers import AutoModelForCausalLM
        import torch as _torch

        model = AutoModelForCausalLM.from_pretrained(
            base_hf_id,
            torch_dtype=_torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:  # pragma: no cover — runtime depends on HF
        print(f"  [aux] WARN: failed to load {base_hf_id}: {exc}")
        return None

    aux_tensors = collect_aux_tensors_from_model(model, keys=aux_keys)
    del model
    if not aux_tensors:
        print(f"  [aux] WARN: no aux keys ({aux_keys}) found in {base_hf_id}; skipping")
        return None

    aux_path = out_dir / 'aux_weights.uc'
    meta = write_aux_weights(aux_path, aux_tensors)
    print(f"  [aux] wrote {aux_path.name}  {meta['size_bytes']/1e6:.1f} MB  "
          f"sha256={meta['sha256'][:16]}...  keys={meta['keys']}")
    return meta


def add_aux_to_existing_pack(
    packed_dir: str | Path, base_hf_id: str | None = None,
    aux_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Retrofit an EXISTING v3 pack with self-contained aux_weights.uc.

    Use case: customer downloads a v3.0 pack from HF (no aux file), runs
    `uc pack-aux <packed_dir>` once locally with the base HF model id, and
    converts it to v3.5 self-contained — no need to re-pack the layer files.
    The original layer_*.uc files are untouched; only `aux_weights.uc` is
    created and `manifest.json` is updated in place.

    Returns the updated manifest dict. Idempotent: re-running with the same
    base model produces the same aux file (deterministic serialization).
    """
    packed_dir = Path(packed_dir).expanduser().resolve()
    if not packed_dir.is_dir():
        raise FileNotFoundError(f"packed_dir does not exist: {packed_dir}")
    manifest_path = packed_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {packed_dir}")

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if base_hf_id is None:
        for key in ('base_model_hf_id', 'base_model', 'hf_id'):
            val = manifest.get(key)
            if isinstance(val, str) and val.strip():
                base_hf_id = val.strip()
                break
    if base_hf_id is None:
        raise ValueError(
            "base_hf_id not in manifest.json; pass it explicitly. "
            "Look it up on the HF model card if unsure."
        )

    print(f"[pack-aux] retrofitting {packed_dir.name} from {base_hf_id}")
    aux_meta = _emit_aux_for_pack(
        e2e_dir=packed_dir,  # not used for resolution since base_hf_id is set
        out_dir=packed_dir,
        base_hf_id=base_hf_id,
        aux_keys=aux_keys or DEFAULT_AUX_KEYS,
    )
    if aux_meta is None:
        raise RuntimeError(
            f"aux generation failed for {base_hf_id}; check transformers + HF cache"
        )

    manifest['pack_format_version'] = PACK_FORMAT_VERSION_V3_5
    manifest['aux_file'] = aux_meta['path']
    manifest['aux_sha256'] = aux_meta['sha256']
    manifest['aux_size_bytes'] = aux_meta['size_bytes']
    manifest['aux_keys'] = aux_meta['keys']
    manifest['aux_n_tensors'] = aux_meta['n_tensors']
    manifest['aux_version'] = aux_meta['version']
    manifest['base_model_hf_id'] = base_hf_id
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[pack-aux] manifest.json updated; pack_format_version={manifest['pack_format_version']}")
    return manifest


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Pack e2e layer artifacts to v3 .uc binaries (lossless)")
    ap.add_argument('src', help='Source _e2e_* directory (must have gsq_codecs)')
    ap.add_argument('dst', help='Output .uc directory')
    ap.add_argument('--bpw', type=int, default=5)
    ap.add_argument('--block-size', type=int, default=64)
    args = ap.parse_args()
    pack_e2e_dir_v3(args.src, args.dst, bpw=args.bpw, block_size=args.block_size)
