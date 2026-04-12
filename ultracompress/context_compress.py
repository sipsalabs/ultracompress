"""
Context-Aware Compression — Automatically select the best compression
method for each layer based on its structural role.

- Attention projections are naturally low-rank  -> SVD / low-rank
- FFN layers have distributed weights          -> quantization
- Embeddings have discrete structure           -> codebook / hashing

One analysis pass inspects each layer, then the right compressor fires.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any
from ultracompress.quantize import quantize_tensor, QuantizedTensor
from ultracompress.codebook import compress_codebook, CodebookCompressed


@dataclass
class LayerProfile:
    name: str
    kind: str          # "attention", "ffn", "embedding", "other"
    rank_ratio: float  # effective rank / full rank (low = good for SVD)
    sparsity: float    # fraction of near-zero weights


def classify_layer(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")):
        return "attention"
    if any(k in n for k in ("gate", "up_proj", "down_proj", "mlp", "ffn")):
        return "ffn"
    if any(k in n for k in ("embed", "lm_head", "token")):
        return "embedding"
    return "other"


class ContextAnalyzer:
    """Profile every layer to decide compression strategy."""

    @staticmethod
    def analyze(name: str, weight: torch.Tensor) -> LayerProfile:
        w = weight.float()
        if w.ndim < 2:
            return LayerProfile(name, classify_layer(name), 1.0, 0.0)
        sv = torch.linalg.svdvals(w[:min(w.shape[0], 512), :min(w.shape[1], 512)])
        eff_rank = (sv.sum() ** 2 / (sv ** 2).sum()).item() / min(w.shape)
        sparsity = (w.abs() < 1e-6).float().mean().item()
        return LayerProfile(name, classify_layer(name), eff_rank, sparsity)


class AdaptiveCompressor:
    """Apply the optimal compression per layer based on its profile."""

    @staticmethod
    def compress(profile: LayerProfile, weight: torch.Tensor) -> Dict[str, Any]:
        if profile.kind == "attention":
            U, S, V = torch.linalg.svd(weight.float(), full_matrices=False)
            rank = max(1, int(min(weight.shape) * 0.25))
            return {"method": "low_rank", "U": U[:, :rank], "S": S[:rank], "V": V[:rank]}
        if profile.kind == "embedding":
            return {"method": "codebook", "data": compress_codebook(weight, bits=8)}
        # ffn and other -> quantization
        bits = 2 if profile.sparsity > 0.3 else 4
        return {"method": "quantize", "data": quantize_tensor(weight, bits=bits)}
