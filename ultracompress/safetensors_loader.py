"""
Safetensors / HuggingFace Model Loader — Load FP16/BF16 source weights.

For maximum compression quality, we need the original unquantized weights.
GGUF weights are already quantized (Q4/Q5), so their spectra are flattened
and recompression is fighting uphill.

FP16 source weights have steep singular value spectra → SVD captures
most energy at low rank → SVD+VQ achieves sub-0.1 BPW.

Supports:
  - Local .safetensors files
  - HuggingFace Hub downloads (requires huggingface_hub)
  - PyTorch .bin files
"""

import os
import torch
from pathlib import Path
from typing import Iterator, Tuple, Optional


def load_safetensors_file(
    path: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Load tensors from a .safetensors file."""
    from safetensors.torch import load_file

    tensors = load_file(path, device=device)
    count = 0
    for name, tensor in tensors.items():
        if name_filter and name_filter not in name:
            continue
        yield name, tensor.float()
        count += 1
        if max_tensors and count >= max_tensors:
            break


def load_safetensors_dir(
    model_dir: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Load all .safetensors files from a directory (sharded models)."""
    model_path = Path(model_dir)
    shard_files = sorted(model_path.glob("*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    total_size = sum(f.stat().st_size for f in shard_files)
    print(f"Loading from: {model_dir}")
    print(f"Shards: {len(shard_files)}, Total size: {total_size / 1e9:.2f} GB")

    count = 0
    for shard_file in shard_files:
        for name, tensor in load_safetensors_file(
            str(shard_file), name_filter=name_filter, device=device,
        ):
            yield name, tensor
            count += 1
            if max_tensors and count >= max_tensors:
                return


def load_hf_model(
    model_id: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
    revision: str = None,
    cache_dir: str = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Download and load a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B")
        max_tensors: limit number of tensors
        name_filter: only load tensors containing this string
        device: target device
        revision: specific commit/branch
        cache_dir: custom cache directory

    Yields:
        (tensor_name, tensor) pairs in float32
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )

    print(f"Downloading model: {model_id}")
    model_dir = snapshot_download(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "config.json"],
    )
    print(f"Model cached at: {model_dir}")

    yield from load_safetensors_dir(
        model_dir, max_tensors=max_tensors,
        name_filter=name_filter, device=device,
    )


def load_pytorch_bin(
    path: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Load from a PyTorch .bin or .pt file."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    count = 0
    for name, tensor in state_dict.items():
        if name_filter and name_filter not in name:
            continue
        yield name, tensor.float()
        count += 1
        if max_tensors and count >= max_tensors:
            break


def load_model(
    source: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Universal loader: auto-detect source type.

    Args:
        source: Can be:
          - Path to .safetensors file
          - Path to directory with .safetensors shards
          - Path to .bin/.pt file
          - HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B")
          - Ollama model name (e.g., "qwen3:4b")
    """
    # Check if it's a local path
    if os.path.exists(source):
        if os.path.isdir(source):
            # Directory with safetensors shards
            return load_safetensors_dir(
                source, max_tensors=max_tensors,
                name_filter=name_filter, device=device,
            )
        elif source.endswith(".safetensors"):
            return load_safetensors_file(
                source, max_tensors=max_tensors,
                name_filter=name_filter, device=device,
            )
        elif source.endswith((".bin", ".pt")):
            return load_pytorch_bin(
                source, max_tensors=max_tensors,
                name_filter=name_filter, device=device,
            )
        elif source.endswith(".gguf"):
            from .gguf_loader import load_gguf_tensors
            return load_gguf_tensors(
                source, max_tensors=max_tensors,
                name_filter=name_filter, device=device,
            )

    # Check if it looks like an Ollama model name (simple name:tag format)
    if "/" not in source and not source.startswith("."):
        from .gguf_loader import load_ollama_model
        return load_ollama_model(
            source, max_tensors=max_tensors,
            name_filter=name_filter, device=device,
        )

    # Assume HuggingFace model ID
    return load_hf_model(
        source, max_tensors=max_tensors,
        name_filter=name_filter, device=device,
    )
