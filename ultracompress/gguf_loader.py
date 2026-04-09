"""
GGUF Model Loader — Extract weight tensors from Ollama models.

Ollama stores models as GGUF files. This module reads them and yields
(name, tensor) pairs for the compression pipeline.

Note: GGUF weights may already be quantized. We dequantize to FP16 first,
then apply our own compression. This simulates the scenario of starting
from a full-precision model.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional


def find_ollama_model_path(model_name: str) -> Optional[str]:
    """Find the GGUF blob for an Ollama model."""
    # Ollama stores manifests that point to blobs
    home = os.path.expanduser("~")
    ollama_dir = os.path.join(home, ".ollama", "models")

    # Parse manifest to find the model blob
    manifest_base = os.path.join(ollama_dir, "manifests", "registry.ollama.ai", "library")

    # Handle model:tag format
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"

    manifest_path = os.path.join(manifest_base, name, tag)

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None

    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Find the largest layer (the model weights)
    blobs_dir = os.path.join(ollama_dir, "blobs")
    largest_blob = None
    largest_size = 0

    for layer in manifest.get("layers", []):
        digest = layer.get("digest", "")
        size = layer.get("size", 0)
        media_type = layer.get("mediaType", "")

        if "model" in media_type and size > largest_size:
            # Digest format: sha256:abc123... -> sha256-abc123...
            blob_name = digest.replace(":", "-")
            blob_path = os.path.join(blobs_dir, blob_name)
            if os.path.exists(blob_path):
                largest_blob = blob_path
                largest_size = size

    return largest_blob


def load_gguf_tensors(
    model_path: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Load tensors from a GGUF file, dequantizing to float32.

    Args:
        model_path: path to the GGUF file
        max_tensors: limit number of tensors loaded (for testing)
        name_filter: only load tensors containing this string
        device: target device

    Yields:
        (tensor_name, tensor) pairs
    """
    from gguf import GGUFReader, dequantize, GGMLQuantizationType

    reader = GGUFReader(model_path)
    count = 0

    for tensor_info in reader.tensors:
        name = tensor_info.name

        if name_filter and name_filter not in name:
            continue

        # Dequantize properly — handles Q4_K, Q5_K, Q8_0, etc.
        data = tensor_info.data
        qtype = tensor_info.tensor_type

        if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            # Already float — just convert
            if isinstance(data, np.ndarray):
                arr = data.copy().astype(np.float32)
            else:
                arr = np.array(data, dtype=np.float32)
        else:
            # Dequantize from quantized format to float32
            arr = dequantize(data, qtype)

        t = torch.from_numpy(arr).float()

        # Reshape to logical dimensions (GGUF stores shape in reversed order)
        shape = tensor_info.shape
        if shape is not None and hasattr(shape, '__len__') and len(shape) > 0:
            target_shape = list(reversed(shape))
            try:
                t = t.reshape(target_shape)
            except RuntimeError:
                pass  # Keep dequantized shape if reshape fails

        yield name, t.to(device)

        count += 1
        if max_tensors and count >= max_tensors:
            break


def list_gguf_tensors(model_path: str) -> list:
    """List all tensor names and shapes in a GGUF file."""
    from gguf import GGUFReader

    reader = GGUFReader(model_path)
    tensors = []
    for tensor_info in reader.tensors:
        shape = list(tensor_info.shape) if tensor_info.shape is not None and len(tensor_info.shape) > 0 else []
        n_params = int(np.prod(shape)) if len(shape) > 0 else tensor_info.data.size
        tensors.append({
            "name": tensor_info.name,
            "shape": shape,
            "n_params": n_params,
            "dtype": str(tensor_info.tensor_type),
        })
    return tensors


def load_ollama_model(
    model_name: str,
    max_tensors: int = None,
    name_filter: str = None,
    device: str = "cpu",
) -> Iterator[Tuple[str, torch.Tensor]]:
    """High-level: load tensors from an Ollama model by name."""
    model_path = find_ollama_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(f"Could not find Ollama model: {model_name}")

    print(f"Loading from: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1e9:.2f} GB")

    yield from load_gguf_tensors(
        model_path,
        max_tensors=max_tensors,
        name_filter=name_filter,
        device=device,
    )
