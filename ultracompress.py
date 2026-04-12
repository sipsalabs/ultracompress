#!/usr/bin/env python3
"""
UltraCompress CLI — Compress any LLM into a .ucz archive.

Pipeline:
    1. Profile  — Analyze model sensitivity per layer
    2. Prune    — Remove redundant layers + attention heads
    3. Factorize — SVD per remaining weight matrix
    4. Quantize — Mixed precision (critical=4bit, bulk=2bit, norms=FP16)
    5. Package  — Save as .ucz (ZIP with manifest.json + compressed bins)

Usage:
    python ultracompress.py compress --model Qwen/Qwen3-0.6B
    python ultracompress.py compress --model ./my_model/ --output my_model.ucz
    python ultracompress.py run --model compressed.ucz --prompt "Hello world"
    python ultracompress.py info --model compressed.ucz
"""

import argparse
import sys
import os
import json
import time
import struct
import zipfile
import io
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerPlan:
    """Compression plan for a single tensor."""
    name: str
    original_shape: list
    original_bytes: int
    action: str           # "keep_fp16", "prune", "factorize_quantize", "quantize_only"
    quant_bits: int = 2   # quantization bits
    svd_rank: int = 0     # 0 means skip SVD
    energy_retained: float = 1.0
    pruned: bool = False


@dataclass
class CompressionManifest:
    """Metadata stored in manifest.json inside .ucz."""
    model_source: str
    n_layers_original: int
    n_layers_kept: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    avg_bits_per_weight: float = 0.0
    layer_plans: list = field(default_factory=list)
    kept_layer_indices: list = field(default_factory=list)
    pruned_heads: dict = field(default_factory=dict)
    pipeline_version: str = "1.0"


# ---------------------------------------------------------------------------
# Stage 1: Profile
# ---------------------------------------------------------------------------

def stage_profile(weight_dict: dict, n_layers: int, verbose: bool = True):
    """Analyze model sensitivity per layer using spectral analysis."""
    from ultracompress.profiler import profile_layer

    if verbose:
        print("\n[Stage 1/5] Profiling layer sensitivity...")

    profiles = {}
    for name, tensor in weight_dict.items():
        if tensor.ndim >= 2 and tensor.numel() > 256:
            prof = profile_layer(name, tensor)
            profiles[name] = prof

    if verbose:
        # Summarize by layer index
        layer_sensitivities = {}
        for name, prof in profiles.items():
            if 'model.layers.' in name:
                try:
                    idx = int(name.split('model.layers.')[1].split('.')[0])
                    if idx not in layer_sensitivities:
                        layer_sensitivities[idx] = []
                    layer_sensitivities[idx].append(prof.spectral_entropy)
                except (ValueError, IndexError):
                    pass

        if layer_sensitivities:
            import numpy as np
            print(f"  Profiled {len(profiles)} tensors across {len(layer_sensitivities)} layers")
            for idx in sorted(layer_sensitivities.keys()):
                avg_ent = np.mean(layer_sensitivities[idx])
                marker = " ***" if idx <= 1 or idx >= n_layers - 2 else ""
                print(f"    Layer {idx:3d}: avg entropy={avg_ent:.3f}{marker}")

    return profiles


# ---------------------------------------------------------------------------
# Stage 2: Prune
# ---------------------------------------------------------------------------

def stage_prune(
    weight_dict: dict,
    n_layers: int,
    hidden_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    layer_prune_ratio: float = 0.30,
    head_prune_ratio: float = 0.40,
    profiles: dict = None,
    verbose: bool = True,
) -> Tuple[dict, list, dict]:
    """Remove redundant middle layers and attention heads.

    Returns (pruned_weight_dict, kept_layer_indices, pruned_heads_per_layer).
    """
    import torch
    import numpy as np

    if verbose:
        print(f"\n[Stage 2/5] Pruning (layers={layer_prune_ratio:.0%}, heads={head_prune_ratio:.0%})...")

    # --- Layer pruning: remove middle layers ---
    # Keep first 2 and last 2 layers always. Prune from the middle.
    n_to_prune = int(n_layers * layer_prune_ratio)
    protected_first = min(2, n_layers // 4)
    protected_last = min(2, n_layers // 4)
    prunable_range = list(range(protected_first, n_layers - protected_last))

    # Score prunability: higher entropy = less important (more redundant)
    if profiles:
        layer_scores = {}
        for idx in prunable_range:
            entropies = []
            for name, prof in profiles.items():
                if f'model.layers.{idx}.' in name:
                    entropies.append(prof.spectral_entropy)
            layer_scores[idx] = np.mean(entropies) if entropies else 0.5
        # Sort by entropy descending (most redundant first)
        prunable_sorted = sorted(prunable_range, key=lambda i: layer_scores.get(i, 0.5), reverse=True)
    else:
        # Default: prune evenly from middle
        mid = n_layers // 2
        prunable_sorted = sorted(prunable_range, key=lambda i: abs(i - mid))

    prune_set = set(prunable_sorted[:min(n_to_prune, len(prunable_sorted))])
    kept_indices = [i for i in range(n_layers) if i not in prune_set]

    if verbose:
        print(f"  Keeping {len(kept_indices)}/{n_layers} layers: {kept_indices[:5]}...{kept_indices[-5:]}")

    # --- Attention head pruning ---
    # For each kept layer, identify least important heads by Q/K weight magnitude
    pruned_heads = {}
    n_heads_to_prune = max(1, int(n_heads * head_prune_ratio))

    for layer_idx in kept_indices:
        q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        if q_key not in weight_dict:
            continue

        q_weight = weight_dict[q_key]
        # Reshape Q weight into heads: (n_heads * head_dim, hidden) -> (n_heads, head_dim, hidden)
        q_per_head = q_weight.reshape(n_heads, head_dim, hidden_size)
        # Score each head by L2 norm of its Q projection
        head_norms = q_per_head.float().norm(dim=(1, 2))  # (n_heads,)

        # Keep top heads, prune bottom ones
        _, sorted_heads = head_norms.sort()
        heads_to_prune = sorted(sorted_heads[:n_heads_to_prune].tolist())
        pruned_heads[layer_idx] = heads_to_prune

    if verbose:
        if pruned_heads:
            sample_layer = kept_indices[len(kept_indices) // 2]
            if sample_layer in pruned_heads:
                print(f"  Pruning {n_heads_to_prune}/{n_heads} heads per layer "
                      f"(e.g. layer {sample_layer}: heads {pruned_heads[sample_layer][:5]}...)")

    # --- Build pruned weight dict ---
    pruned_wd = {}
    for name, tensor in weight_dict.items():
        # Non-layer weights: keep as-is
        if 'model.layers.' not in name:
            pruned_wd[name] = tensor
            continue

        # Parse layer index
        try:
            idx = int(name.split('model.layers.')[1].split('.')[0])
        except (ValueError, IndexError):
            pruned_wd[name] = tensor
            continue

        # Skip pruned layers
        if idx in prune_set:
            continue

        # Re-index to new position
        new_idx = kept_indices.index(idx)
        new_name = name.replace(f'model.layers.{idx}.', f'model.layers.{new_idx}.')

        # Apply head pruning to Q, K, V, O projections
        if idx in pruned_heads and any(proj in name for proj in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight']):
            heads_to_prune = pruned_heads[idx]
            tensor = _prune_heads_from_weight(
                tensor, name, heads_to_prune,
                n_heads, n_kv_heads, head_dim, hidden_size,
            )

        pruned_wd[new_name] = tensor

    if verbose:
        orig_params = sum(t.numel() for t in weight_dict.values())
        pruned_params = sum(t.numel() for t in pruned_wd.values())
        print(f"  Parameters: {orig_params:,} -> {pruned_params:,} ({pruned_params/orig_params:.1%})")

    return pruned_wd, kept_indices, pruned_heads


def _prune_heads_from_weight(
    weight, name, heads_to_prune, n_heads, n_kv_heads, head_dim, hidden_size,
):
    """Zero out pruned attention heads in Q/K/V/O projection weights."""
    import torch

    w = weight.clone()

    if 'q_proj' in name:
        # Q: (n_heads * head_dim, hidden_size) -- zero out rows for pruned heads
        for h in heads_to_prune:
            start = h * head_dim
            end = start + head_dim
            if end <= w.shape[0]:
                w[start:end] = 0

    elif 'k_proj' in name or 'v_proj' in name:
        # K/V use n_kv_heads (GQA). Map pruned query heads to KV head groups.
        heads_per_kv = n_heads // max(n_kv_heads, 1)
        kv_heads_to_prune = set()
        for h in heads_to_prune:
            kv_idx = h // heads_per_kv
            # Only prune KV head if ALL its query heads are pruned
            group_start = kv_idx * heads_per_kv
            group_heads = set(range(group_start, group_start + heads_per_kv))
            if group_heads.issubset(set(heads_to_prune)):
                kv_heads_to_prune.add(kv_idx)
        for kv_h in kv_heads_to_prune:
            start = kv_h * head_dim
            end = start + head_dim
            if end <= w.shape[0]:
                w[start:end] = 0

    elif 'o_proj' in name:
        # O: (hidden_size, n_heads * head_dim) -- zero out columns for pruned heads
        for h in heads_to_prune:
            start = h * head_dim
            end = start + head_dim
            if end <= w.shape[1]:
                w[:, start:end] = 0

    return w


# ---------------------------------------------------------------------------
# Stage 3: Factorize
# ---------------------------------------------------------------------------

def stage_factorize(
    weight_dict: dict,
    min_rank: int = 64,
    max_rank: int = 128,
    energy_target: float = 0.99,
    min_params_for_svd: int = 65536,
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """SVD factorize large weight matrices.

    Returns (factorized_dict, svd_info).
    factorized_dict maps name -> tensor (reconstructed from SVD at target rank).
    svd_info maps name -> {rank, energy_retained, U_shape, V_shape}.
    For small tensors or norms, keeps originals.
    """
    import torch
    from ultracompress.factorize import factorize_weight

    if verbose:
        print(f"\n[Stage 3/5] Factorizing (rank {min_rank}-{max_rank}, energy={energy_target})...")

    factorized = {}
    svd_info = {}
    total_original = 0
    total_factorized = 0

    for name, tensor in weight_dict.items():
        total_original += tensor.numel()

        # Skip 1D tensors (norms, biases) and small tensors
        if tensor.ndim < 2 or tensor.numel() < min_params_for_svd:
            factorized[name] = tensor
            total_factorized += tensor.numel()
            continue

        # Skip embeddings and lm_head (they need exact lookup)
        if 'embed_tokens' in name or 'lm_head' in name:
            factorized[name] = tensor
            total_factorized += tensor.numel()
            continue

        try:
            fw = factorize_weight(
                tensor,
                energy_target=energy_target,
                min_rank=min_rank,
                max_rank=max_rank,
                device="cpu",
            )

            # Store the factors' product (reconstructed weight) for next stage
            reconstructed = (fw.U @ fw.V)
            if tensor.ndim > 2:
                reconstructed = reconstructed.reshape(tensor.shape)

            factorized[name] = reconstructed
            factor_params = fw.U.numel() + fw.V.numel()
            total_factorized += factor_params

            svd_info[name] = {
                'rank': fw.rank,
                'energy_retained': fw.energy_retained,
                'U_shape': list(fw.U.shape),
                'V_shape': list(fw.V.shape),
                'original_params': tensor.numel(),
                'factor_params': factor_params,
            }

        except Exception as e:
            # Fallback: keep original
            factorized[name] = tensor
            total_factorized += tensor.numel()
            if verbose:
                print(f"    SVD failed for {name}: {e}")

    if verbose:
        n_factorized = len(svd_info)
        avg_rank = sum(v['rank'] for v in svd_info.values()) / max(n_factorized, 1)
        avg_energy = sum(v['energy_retained'] for v in svd_info.values()) / max(n_factorized, 1)
        orig_factor_params = sum(v['original_params'] for v in svd_info.values())
        new_factor_params = sum(v['factor_params'] for v in svd_info.values())
        print(f"  Factorized {n_factorized} tensors, avg rank={avg_rank:.0f}, avg energy={avg_energy:.4f}")
        if orig_factor_params > 0:
            print(f"  SVD parameter reduction: {orig_factor_params:,} -> {new_factor_params:,} "
                  f"({new_factor_params/orig_factor_params:.1%})")

    return factorized, svd_info


# ---------------------------------------------------------------------------
# Stage 4: Quantize
# ---------------------------------------------------------------------------

def _classify_for_quant(name: str, layer_idx: int, n_layers: int) -> Tuple[int, str]:
    """Assign quantization bits based on tensor type and position.

    Returns (bits, reason).
    """
    name_lower = name.lower()

    # Norms: keep FP16 (tiny, critical)
    if 'norm' in name_lower or 'layernorm' in name_lower:
        return 16, "norm_fp16"

    # Embeddings and LM head: 4-bit
    if 'embed_tokens' in name_lower or 'lm_head' in name_lower:
        return 4, "embed_or_head"

    # First/last 2 layers: 4-bit for Q/K, 3-bit for rest
    is_boundary = (layer_idx >= 0 and (layer_idx < 2 or layer_idx >= n_layers - 2))

    if 'q_proj' in name_lower or 'k_proj' in name_lower:
        if is_boundary:
            return 4, "qk_boundary"
        return 4, "qk_inner"

    # V and O projections
    if 'v_proj' in name_lower or 'o_proj' in name_lower:
        if is_boundary:
            return 4, "vo_boundary"
        return 2, "vo_inner"

    # MLP layers
    if any(k in name_lower for k in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
        if is_boundary:
            return 3, "mlp_boundary"
        return 2, "mlp_inner"

    # Default
    return 2, "default"


def stage_quantize(
    weight_dict: dict,
    n_layers: int,
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """Mixed-precision quantization.

    Returns (quantized_dict, quant_info).
    quantized_dict maps name -> QuantizedTensor or raw half tensor.
    """
    import torch
    from ultracompress.quantize import quantize_absmax, QuantizedTensor

    if verbose:
        print(f"\n[Stage 4/5] Quantizing (mixed precision)...")

    quantized = {}
    quant_info = {}
    bits_distribution = {}

    for name, tensor in weight_dict.items():
        # Detect layer index
        layer_idx = -1
        if 'model.layers.' in name:
            try:
                layer_idx = int(name.split('model.layers.')[1].split('.')[0])
            except (ValueError, IndexError):
                pass

        bits, reason = _classify_for_quant(name, layer_idx, n_layers)
        bits_distribution[bits] = bits_distribution.get(bits, 0) + tensor.numel()

        if bits >= 16:
            # Keep as FP16
            quantized[name] = tensor.half()
            quant_info[name] = {
                'bits': 16, 'reason': reason,
                'bytes': tensor.numel() * 2,
            }
        else:
            # Group quantize
            group_size = 128 if tensor.numel() >= 1024 else max(32, tensor.numel() // 4)
            qt = quantize_absmax(tensor, bits=bits, group_size=group_size)
            quantized[name] = qt
            quant_info[name] = {
                'bits': bits, 'reason': reason,
                'bytes': qt.storage_bytes(),
                'group_size': group_size,
            }

    if verbose:
        total_params = sum(t.numel() if hasattr(t, 'numel') and not isinstance(t, QuantizedTensor) else 0
                          for t in weight_dict.values())
        total_params += sum(v.get('bytes', 0) // 2 for v in quant_info.values() if v['bits'] >= 16)
        print(f"  Bit allocation:")
        for bits in sorted(bits_distribution.keys()):
            count = bits_distribution[bits]
            pct = count / max(sum(bits_distribution.values()), 1) * 100
            print(f"    {bits:2d}-bit: {count:>12,} params ({pct:.1f}%)")

        total_bytes = sum(v['bytes'] for v in quant_info.values())
        total_orig_bytes = sum(t.numel() * 2 for t in weight_dict.values())
        avg_bpw = total_bytes * 8 / max(sum(t.numel() for t in weight_dict.values()), 1)
        print(f"  Compressed: {total_orig_bytes/1e6:.1f} MB -> {total_bytes/1e6:.1f} MB ({avg_bpw:.2f} BPW)")

    return quantized, quant_info


# ---------------------------------------------------------------------------
# Stage 5: Package
# ---------------------------------------------------------------------------

def _serialize_tensor(tensor) -> bytes:
    """Serialize a tensor or QuantizedTensor to bytes."""
    import torch
    from ultracompress.quantize import QuantizedTensor

    buf = io.BytesIO()

    if isinstance(tensor, QuantizedTensor):
        # Pack: header(1) + bits(1) + group_size(4) + shape + codes + scales + zeros
        buf.write(b'\x01')  # marker for QuantizedTensor
        buf.write(struct.pack('<B', tensor.bits))
        buf.write(struct.pack('<I', tensor.group_size))
        # Original shape
        shape = tensor.original_shape
        buf.write(struct.pack('<B', len(shape)))
        for dim in shape:
            buf.write(struct.pack('<I', dim))
        buf.write(struct.pack('<I', tensor.n_elements))
        # Codes
        codes_bytes = tensor.codes.cpu().numpy().tobytes()
        buf.write(struct.pack('<I', len(codes_bytes)))
        buf.write(codes_bytes)
        # Scales
        scales_bytes = tensor.scales.cpu().half().numpy().tobytes()
        buf.write(struct.pack('<I', len(scales_bytes)))
        buf.write(scales_bytes)
        # Zeros
        zeros_bytes = tensor.zeros.cpu().half().numpy().tobytes()
        buf.write(struct.pack('<I', len(zeros_bytes)))
        buf.write(zeros_bytes)
    else:
        # Raw FP16 tensor
        buf.write(b'\x00')  # marker for raw tensor
        shape = tuple(tensor.shape)
        buf.write(struct.pack('<B', len(shape)))
        for dim in shape:
            buf.write(struct.pack('<I', dim))
        data = tensor.cpu().half().numpy().tobytes()
        buf.write(struct.pack('<I', len(data)))
        buf.write(data)

    return buf.getvalue()


def _deserialize_tensor(data: bytes):
    """Deserialize bytes back to a tensor or QuantizedTensor."""
    import torch
    import numpy as np
    from ultracompress.quantize import QuantizedTensor

    buf = io.BytesIO(data)
    marker = buf.read(1)

    if marker == b'\x01':
        # QuantizedTensor
        bits = struct.unpack('<B', buf.read(1))[0]
        group_size = struct.unpack('<I', buf.read(4))[0]
        n_dims = struct.unpack('<B', buf.read(1))[0]
        shape = tuple(struct.unpack('<I', buf.read(4))[0] for _ in range(n_dims))
        n_elements = struct.unpack('<I', buf.read(4))[0]
        # Codes
        codes_len = struct.unpack('<I', buf.read(4))[0]
        codes_np = np.frombuffer(buf.read(codes_len), dtype=np.uint8)
        codes = torch.from_numpy(codes_np.copy())
        # Scales
        scales_len = struct.unpack('<I', buf.read(4))[0]
        scales_np = np.frombuffer(buf.read(scales_len), dtype=np.float16)
        scales = torch.from_numpy(scales_np.copy())
        # Zeros
        zeros_len = struct.unpack('<I', buf.read(4))[0]
        zeros_np = np.frombuffer(buf.read(zeros_len), dtype=np.float16)
        zeros = torch.from_numpy(zeros_np.copy())

        return QuantizedTensor(
            codes=codes, scales=scales, zeros=zeros,
            bits=bits, group_size=group_size,
            original_shape=shape, n_elements=n_elements,
        )
    else:
        # Raw FP16 tensor
        n_dims = struct.unpack('<B', buf.read(1))[0]
        shape = tuple(struct.unpack('<I', buf.read(4))[0] for _ in range(n_dims))
        data_len = struct.unpack('<I', buf.read(4))[0]
        tensor_np = np.frombuffer(buf.read(data_len), dtype=np.float16)
        return torch.from_numpy(tensor_np.copy()).reshape(shape).float()


def stage_package(
    quantized_dict: dict,
    manifest: CompressionManifest,
    output_path: str,
    verbose: bool = True,
) -> int:
    """Package compressed model as .ucz (ZIP with manifest + binary layers).

    Returns compressed file size in bytes.
    """
    if verbose:
        print(f"\n[Stage 5/5] Packaging -> {output_path}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        manifest_json = json.dumps(asdict(manifest), indent=2, default=str)
        zf.writestr('manifest.json', manifest_json)

        # Write each tensor as a binary file
        n_tensors = len(quantized_dict)
        for i, (name, tensor) in enumerate(quantized_dict.items()):
            # Use sanitized filename
            safe_name = name.replace('/', '_').replace('\\', '_')
            bin_data = _serialize_tensor(tensor)
            zf.writestr(f'tensors/{safe_name}.bin', bin_data)
            if verbose and (i % 20 == 0 or i == n_tensors - 1):
                print(f"  Packaged {i+1}/{n_tensors} tensors...")

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"  Written: {output_path} ({file_size / 1e6:.1f} MB)")

    return file_size


# ---------------------------------------------------------------------------
# Load .ucz for inference
# ---------------------------------------------------------------------------

def load_ucz(ucz_path: str, device: str = "cpu") -> Tuple[dict, CompressionManifest]:
    """Load a .ucz archive, returning (weight_dict, manifest).

    All quantized tensors are decompressed back to float32.
    """
    import torch
    from ultracompress.quantize import QuantizedTensor

    weight_dict = {}
    manifest = None

    with zipfile.ZipFile(ucz_path, 'r') as zf:
        # Read manifest
        manifest_data = json.loads(zf.read('manifest.json'))
        manifest = CompressionManifest(**{
            k: v for k, v in manifest_data.items()
            if k in CompressionManifest.__dataclass_fields__
        })

        # Read tensors
        tensor_files = [n for n in zf.namelist() if n.startswith('tensors/') and n.endswith('.bin')]
        for tf in tensor_files:
            # Recover original name from filename
            safe_name = tf.replace('tensors/', '').replace('.bin', '')
            data = zf.read(tf)
            obj = _deserialize_tensor(data)

            if isinstance(obj, QuantizedTensor):
                # Decompress to float
                weight_dict[safe_name] = obj.decompress().to(device)
            else:
                weight_dict[safe_name] = obj.float().to(device)

    return weight_dict, manifest


# ---------------------------------------------------------------------------
# Model loader helper
# ---------------------------------------------------------------------------

def _load_source_weights(source: str, verbose: bool = True) -> Tuple[dict, dict]:
    """Load model weights from various sources.

    Returns (weight_dict, config_dict).
    config_dict has keys: n_layers, n_heads, n_kv_heads, hidden_size,
                         intermediate_size, vocab_size, head_dim.
    """
    import torch

    weight_dict = {}

    # Determine source type and load
    if os.path.isdir(source):
        # Local directory with safetensors
        from ultracompress.safetensors_loader import load_safetensors_dir
        for name, tensor in load_safetensors_dir(source):
            weight_dict[name] = tensor
        # Try to load config.json
        config_path = os.path.join(source, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                hf_config = json.load(f)
        else:
            hf_config = None
    elif source.endswith('.safetensors'):
        from ultracompress.safetensors_loader import load_safetensors_file
        for name, tensor in load_safetensors_file(source):
            weight_dict[name] = tensor
        hf_config = None
    elif source.endswith(('.bin', '.pt')):
        weight_dict = torch.load(source, map_location='cpu', weights_only=True)
        hf_config = None
    else:
        # HuggingFace model ID — download
        from ultracompress.safetensors_loader import load_hf_model
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                source,
                allow_patterns=["*.safetensors", "config.json"],
            )
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    hf_config = json.load(f)
            else:
                hf_config = None
        except Exception:
            hf_config = None

        for name, tensor in load_hf_model(source):
            weight_dict[name] = tensor

    # Detect architecture from weights if no config
    config = _detect_config(weight_dict, hf_config)

    if verbose:
        total_params = sum(t.numel() for t in weight_dict.values())
        total_bytes = sum(t.numel() * t.element_size() for t in weight_dict.values())
        print(f"Loaded {len(weight_dict)} tensors, {total_params:,} params ({total_bytes/1e6:.0f} MB)")
        print(f"Architecture: {config['n_layers']}L / {config['hidden_size']}H / "
              f"{config['n_heads']}heads / {config['n_kv_heads']}kv / "
              f"{config['intermediate_size']}ff / {config['vocab_size']}vocab")

    return weight_dict, config


def _detect_config(weight_dict: dict, hf_config: dict = None) -> dict:
    """Detect model architecture from weights and/or HF config."""
    if hf_config:
        hidden = hf_config.get('hidden_size', 0)
        n_heads = hf_config.get('num_attention_heads', 0)
        n_kv = hf_config.get('num_key_value_heads', n_heads)
        n_layers = hf_config.get('num_hidden_layers', 0)
        intermediate = hf_config.get('intermediate_size', 0)
        vocab = hf_config.get('vocab_size', 0)
        head_dim = hf_config.get('head_dim', hidden // max(n_heads, 1))
        norm_eps = hf_config.get('rms_norm_eps', 1e-6)
        rope_theta = hf_config.get('rope_theta', 10000.0)

        if hidden and n_heads and n_layers:
            return {
                'n_layers': n_layers, 'n_heads': n_heads, 'n_kv_heads': n_kv,
                'hidden_size': hidden, 'intermediate_size': intermediate,
                'vocab_size': vocab, 'head_dim': head_dim,
                'norm_eps': norm_eps, 'rope_theta': rope_theta,
            }

    # Fallback: detect from weight shapes
    n_layers = 0
    hidden = 0
    for key in weight_dict:
        if 'model.layers.' in key:
            try:
                idx = int(key.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                pass
        if 'q_proj.weight' in key and hidden == 0:
            hidden = weight_dict[key].shape[1]

    head_dim = 128
    for k in weight_dict:
        if 'q_norm' in k and 'layers.0' in k:
            head_dim = weight_dict[k].shape[0]
            break

    q_keys = [k for k in weight_dict if 'layers.0' in k and 'q_proj.weight' in k]
    k_keys = [k for k in weight_dict if 'layers.0' in k and 'k_proj.weight' in k]
    n_heads = weight_dict[q_keys[0]].shape[0] // head_dim if q_keys else 16
    n_kv = weight_dict[k_keys[0]].shape[0] // head_dim if k_keys else 8

    intermediate = 0
    for k in weight_dict:
        if 'gate_proj.weight' in k and 'layers.0' in k:
            intermediate = weight_dict[k].shape[0]
            break

    vocab = weight_dict.get('model.embed_tokens.weight', None)
    vocab = vocab.shape[0] if vocab is not None else 151936

    return {
        'n_layers': n_layers, 'n_heads': n_heads, 'n_kv_heads': n_kv,
        'hidden_size': hidden, 'intermediate_size': intermediate,
        'vocab_size': vocab, 'head_dim': head_dim,
        'norm_eps': 1e-6, 'rope_theta': 10000.0,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_compress(args):
    """Full 5-stage compression pipeline."""
    import torch

    t_start = time.time()
    print("=" * 65)
    print("  UltraCompress v1.0 -- LLM Compression Pipeline")
    print("=" * 65)
    print(f"  Source:  {args.model}")
    print(f"  Output:  {args.output or '(auto)'}")
    print()

    # --- Load ---
    print("[Loading model...]")
    weight_dict, config = _load_source_weights(args.model)

    n_layers = config['n_layers']
    hidden_size = config['hidden_size']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = config['head_dim']
    intermediate_size = config['intermediate_size']
    vocab_size = config['vocab_size']

    original_bytes = sum(t.numel() * 2 for t in weight_dict.values())  # FP16 baseline

    # --- Stage 1: Profile ---
    profiles = stage_profile(weight_dict, n_layers, verbose=not args.quiet)

    # --- Stage 2: Prune ---
    pruned_wd, kept_indices, pruned_heads = stage_prune(
        weight_dict, n_layers, hidden_size, n_heads, n_kv_heads, head_dim,
        layer_prune_ratio=args.layer_prune,
        head_prune_ratio=args.head_prune,
        profiles=profiles,
        verbose=not args.quiet,
    )
    del weight_dict  # free memory

    n_layers_kept = len(kept_indices)

    # --- Stage 3: Factorize ---
    if args.skip_svd:
        factorized_wd = pruned_wd
        svd_info = {}
        if not args.quiet:
            print("\n[Stage 3/5] Factorize: SKIPPED (--skip-svd)")
    else:
        factorized_wd, svd_info = stage_factorize(
            pruned_wd,
            min_rank=args.svd_min_rank,
            max_rank=args.svd_max_rank,
            energy_target=args.svd_energy,
            verbose=not args.quiet,
        )
    del pruned_wd

    # --- Stage 4: Quantize ---
    quantized_wd, quant_info = stage_quantize(
        factorized_wd, n_layers_kept,
        verbose=not args.quiet,
    )
    del factorized_wd

    # --- Stage 5: Package ---
    manifest = CompressionManifest(
        model_source=args.model,
        n_layers_original=n_layers,
        n_layers_kept=n_layers_kept,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        norm_eps=config.get('norm_eps', 1e-6),
        rope_theta=config.get('rope_theta', 10000.0),
        original_size_bytes=original_bytes,
        kept_layer_indices=kept_indices,
        pruned_heads={str(k): v for k, v in pruned_heads.items()},
    )

    # Auto output name
    if args.output:
        output_path = args.output
    else:
        basename = os.path.basename(args.model.rstrip('/'))
        output_path = f"{basename}.ucz"

    compressed_bytes = stage_package(
        quantized_wd, manifest, output_path,
        verbose=not args.quiet,
    )

    # Update manifest with final stats
    manifest.compressed_size_bytes = compressed_bytes
    manifest.compression_ratio = original_bytes / max(compressed_bytes, 1)
    total_params = sum(
        t.numel() if hasattr(t, 'numel') else t.n_elements
        for t in quantized_wd.values()
    )
    manifest.avg_bits_per_weight = (compressed_bytes * 8) / max(total_params, 1)

    elapsed = time.time() - t_start

    # --- Final report ---
    print()
    print("=" * 65)
    print("  Compression Complete")
    print("=" * 65)
    print(f"  Source model:     {args.model}")
    print(f"  Original size:    {original_bytes / 1e6:.0f} MB (FP16)")
    print(f"  Compressed size:  {compressed_bytes / 1e6:.1f} MB (.ucz)")
    print(f"  Compression:      {manifest.compression_ratio:.1f}x")
    print(f"  Layers:           {n_layers} -> {n_layers_kept} (pruned {n_layers - n_layers_kept})")
    print(f"  Head pruning:     {args.head_prune:.0%} per layer")
    print(f"  SVD tensors:      {len(svd_info)}")
    print(f"  Output:           {output_path}")
    print(f"  Time:             {elapsed:.1f}s")
    print("=" * 65)


def cmd_run(args):
    """Load .ucz and run inference."""
    import torch
    import torch.nn.functional as F

    print(f"Loading: {args.model}")

    if not args.model.endswith('.ucz'):
        print("Error: --model must be a .ucz file. Use 'compress' first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight_dict, manifest = load_ucz(args.model, device=device)

    print(f"Model: {manifest.model_source}")
    print(f"Architecture: {manifest.n_layers_kept}L / {manifest.hidden_size}H / "
          f"{manifest.n_heads}heads")
    print(f"Compressed: {manifest.compressed_size_bytes / 1e6:.1f} MB "
          f"({manifest.compression_ratio:.1f}x)")
    print()

    # Build a minimal transformer for inference
    from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer, RMSNorm

    mc = ModelConfig(
        n_layers=manifest.n_layers_kept,
        n_heads=manifest.n_heads,
        n_kv_heads=manifest.n_kv_heads,
        hidden_size=manifest.hidden_size,
        intermediate_size=manifest.intermediate_size,
        vocab_size=manifest.vocab_size,
        head_dim=manifest.head_dim,
        norm_eps=manifest.norm_eps,
        rope_theta=manifest.rope_theta,
    )

    # Map HF-style weight names to GGUF-style names expected by MiniTransformer
    gguf_wd = {}
    name_map_global = {
        'model.embed_tokens.weight': 'token_embd.weight',
        'model.norm.weight': 'output_norm.weight',
        'lm_head.weight': 'output.weight',
    }
    layer_name_map = {
        'self_attn.q_proj.weight': 'attn_q.weight',
        'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight',
        'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
        'self_attn.q_norm.weight': 'attn_q_norm.weight',
        'self_attn.k_norm.weight': 'attn_k_norm.weight',
    }

    for name, tensor in weight_dict.items():
        if name in name_map_global:
            gguf_wd[name_map_global[name]] = tensor
        elif 'model.layers.' in name:
            try:
                idx = int(name.split('model.layers.')[1].split('.')[0])
                suffix = name.split(f'model.layers.{idx}.')[1]
                if suffix in layer_name_map:
                    gguf_name = f"blk.{idx}.{layer_name_map[suffix]}"
                    gguf_wd[gguf_name] = tensor
            except (ValueError, IndexError):
                pass
        else:
            gguf_wd[name] = tensor

    # If no lm_head, tie to embedding
    if 'output.weight' not in gguf_wd and 'token_embd.weight' in gguf_wd:
        gguf_wd['output.weight'] = gguf_wd['token_embd.weight']

    model = MiniTransformer(mc, device)
    model.load_weights(gguf_wd)

    print(f"Loaded {len(model.layers)} transformer layers on {device}")

    # Tokenize
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            manifest.model_source, trust_remote_code=True,
        )
        input_ids = tok.encode(args.prompt)
        tokens = torch.tensor([input_ids], device=device)
        has_tokenizer = True
    except Exception:
        # Fallback: simple byte encoding
        print("(No tokenizer available, using raw byte encoding)")
        input_ids = list(args.prompt.encode('utf-8'))
        tokens = torch.tensor([input_ids], device=device)
        has_tokenizer = False

    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens...\n")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            tokens, max_new=args.max_tokens,
            temperature=args.temperature,
        )

    if has_tokenizer:
        full_ids = input_ids + generated
        output_text = tok.decode(full_ids, skip_special_tokens=True)
        print(output_text)
    else:
        try:
            output_bytes = bytes(generated)
            print(output_bytes.decode('utf-8', errors='replace'))
        except Exception:
            print(f"Generated token IDs: {generated}")


def cmd_info(args):
    """Show info about a .ucz file."""
    if not args.model.endswith('.ucz'):
        print("Error: --model must be a .ucz file.")
        return

    file_size = os.path.getsize(args.model)

    with zipfile.ZipFile(args.model, 'r') as zf:
        manifest_data = json.loads(zf.read('manifest.json'))
        tensor_files = [n for n in zf.namelist() if n.startswith('tensors/')]

    print("=" * 55)
    print("  UltraCompress Archive Info")
    print("=" * 55)
    print(f"  File:             {args.model}")
    print(f"  File size:        {file_size / 1e6:.1f} MB")
    print(f"  Source model:     {manifest_data.get('model_source', 'unknown')}")
    print(f"  Pipeline version: {manifest_data.get('pipeline_version', '?')}")
    print()
    print(f"  Architecture:")
    print(f"    Original layers: {manifest_data.get('n_layers_original', '?')}")
    print(f"    Kept layers:     {manifest_data.get('n_layers_kept', '?')}")
    print(f"    Hidden size:     {manifest_data.get('hidden_size', '?')}")
    print(f"    Attention heads: {manifest_data.get('n_heads', '?')}")
    print(f"    KV heads:        {manifest_data.get('n_kv_heads', '?')}")
    print(f"    Head dim:        {manifest_data.get('head_dim', '?')}")
    print(f"    Intermediate:    {manifest_data.get('intermediate_size', '?')}")
    print(f"    Vocab size:      {manifest_data.get('vocab_size', '?')}")
    print()
    print(f"  Compression:")
    print(f"    Original size:   {manifest_data.get('original_size_bytes', 0) / 1e6:.0f} MB")
    print(f"    Compressed size: {manifest_data.get('compressed_size_bytes', 0) / 1e6:.1f} MB")
    print(f"    Ratio:           {manifest_data.get('compression_ratio', 0):.1f}x")
    print(f"    Avg BPW:         {manifest_data.get('avg_bits_per_weight', 0):.2f}")
    print(f"    Tensors:         {len(tensor_files)}")

    kept = manifest_data.get('kept_layer_indices', [])
    if kept:
        print(f"    Kept layers:     {kept}")

    print("=" * 55)


def cmd_list(args):
    """List tensors inside a .ucz archive."""
    if not args.model.endswith('.ucz'):
        print("Error: --model must be a .ucz file.")
        return

    with zipfile.ZipFile(args.model, 'r') as zf:
        tensor_files = sorted([n for n in zf.namelist() if n.startswith('tensors/')])
        print(f"Tensors in {args.model} ({len(tensor_files)}):")
        total_size = 0
        for tf in tensor_files:
            info = zf.getinfo(tf)
            size = info.compress_size
            total_size += size
            name = tf.replace('tensors/', '').replace('.bin', '')
            print(f"  {name:<60} {size:>10,} bytes")
        print(f"\n  Total: {total_size:,} bytes ({total_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UltraCompress -- LLM Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ultracompress.py compress --model Qwen/Qwen3-0.6B
    python ultracompress.py compress --model ./my_model/ --output small.ucz
    python ultracompress.py run --model compressed.ucz --prompt "Hello"
    python ultracompress.py info --model compressed.ucz
    python ultracompress.py list --model compressed.ucz
        """,
    )
    sub = parser.add_subparsers(dest='command')

    # --- compress ---
    p = sub.add_parser('compress', help='Compress a model into .ucz format')
    p.add_argument('--model', required=True,
                   help='HuggingFace model ID, local directory, or weights file')
    p.add_argument('--output', '-o', help='Output .ucz path (default: auto)')
    p.add_argument('--layer-prune', type=float, default=0.30,
                   help='Fraction of middle layers to prune (default: 0.30)')
    p.add_argument('--head-prune', type=float, default=0.40,
                   help='Fraction of attention heads to prune (default: 0.40)')
    p.add_argument('--svd-min-rank', type=int, default=64,
                   help='Minimum SVD rank (default: 64)')
    p.add_argument('--svd-max-rank', type=int, default=128,
                   help='Maximum SVD rank (default: 128)')
    p.add_argument('--svd-energy', type=float, default=0.99,
                   help='SVD energy retention target (default: 0.99)')
    p.add_argument('--skip-svd', action='store_true',
                   help='Skip SVD factorization stage')
    p.add_argument('--quiet', '-q', action='store_true',
                   help='Suppress per-stage progress output')

    # --- run ---
    p = sub.add_parser('run', help='Run inference with a .ucz model')
    p.add_argument('--model', required=True, help='.ucz compressed model file')
    p.add_argument('--prompt', default='The meaning of life is',
                   help='Prompt text')
    p.add_argument('--max-tokens', type=int, default=50,
                   help='Max tokens to generate')
    p.add_argument('--temperature', type=float, default=0.7,
                   help='Sampling temperature')

    # --- info ---
    p = sub.add_parser('info', help='Show .ucz archive info')
    p.add_argument('--model', required=True, help='.ucz file')

    # --- list ---
    p = sub.add_parser('list', help='List tensors in a .ucz archive')
    p.add_argument('--model', required=True, help='.ucz file')

    args = parser.parse_args()

    if args.command == 'compress':
        cmd_compress(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
