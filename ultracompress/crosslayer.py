"""
Cross-Layer Weight Sharing & Delta Compression

The key insight: in a 32-layer transformer, layer 0's attention weights
look remarkably similar to layer 31's. They're not identical, but the
shared structure is massive.

Instead of compressing each layer independently (what everyone does),
we extract the shared "base" pattern across all layers, then compress
only the per-layer deviations (deltas).

Compression math:
  Original: L layers * W weights/layer = L*W total weights
  Cross-layer: 1 base (W weights) + L deltas (D weights each, D << W)
  If deltas are 5% of original: compression = L*W / (W + L*0.05*W) = L/(1+0.05*L)
  For L=32: 32/2.6 = 12.3x compression BEFORE any quantization
  Then quantize the base + deltas → compounds with VQ for extreme ratios

Three strategies:
  1. Mean subtraction: base = mean(all layers), delta = layer - base
  2. SVD across layers: stack all layers, SVD the stack, keep top-k components
  3. Anchor + delta: pick best layer as anchor, store diffs from anchor

Strategy 2 is the most powerful — it finds the principal "layer patterns"
that explain the most variance across layers.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class CrossLayerCompressed:
    """Cross-layer compressed representation of a weight type.

    For a weight type (e.g., "attn_q") across L layers:
      - basis: k principal component matrices (each same shape as original weight)
      - coefficients: L x k matrix (how much of each component per layer)
      - residuals: per-layer residual tensors (optional, for lossless)

    Storage: k full matrices + L*k scalars + residual storage
    vs original: L full matrices

    If k << L, this is a huge win.
    """
    weight_key: str  # e.g., "attn_q.weight"
    basis: List[torch.Tensor]  # k basis matrices, each (m, n)
    coefficients: torch.Tensor  # (L, k) — how to combine basis per layer
    residuals: Optional[List[torch.Tensor]] = None
    original_shape: tuple = ()
    n_layers: int = 0
    n_components: int = 0


@dataclass
class CrossLayerResult:
    """Results of cross-layer analysis."""
    weight_key: str
    n_layers: int
    original_bytes: int
    compressed_bytes: int
    # Per-layer quality
    cosine_sims: List[float] = field(default_factory=list)
    relative_errors: List[float] = field(default_factory=list)
    # Component analysis
    n_components: int = 0
    variance_explained: float = 0.0

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)

    @property
    def avg_cosine_sim(self) -> float:
        return np.mean(self.cosine_sims) if self.cosine_sims else 0.0

    @property
    def min_cosine_sim(self) -> float:
        return min(self.cosine_sims) if self.cosine_sims else 0.0


def group_weights_by_type(named_weights: list) -> Dict[str, List[Tuple[int, torch.Tensor]]]:
    """Group weight tensors by their type across layers.

    Input: [("blk.0.attn_q.weight", tensor), ("blk.0.attn_v.weight", tensor), ...]
    Output: {"attn_q.weight": [(0, tensor0), (1, tensor1), ...], ...}
    """
    groups = {}
    for name, tensor in named_weights:
        # Extract layer index and weight type
        parts = name.split(".")
        layer_idx = -1
        weight_type = name

        # Handle common naming patterns
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_idx = int(part)
                # Weight type is everything after the layer index
                weight_type = ".".join(parts[i+1:])
                break
            elif part.startswith("blk") or part.startswith("layer"):
                # "blk.0.attn_q.weight" -> layer_idx=0, type="attn_q.weight"
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_idx = int(parts[i+1])
                    weight_type = ".".join(parts[i+2:])
                    break

        if layer_idx < 0:
            continue  # Skip non-layer weights (embeddings, etc.)

        if weight_type not in groups:
            groups[weight_type] = []
        groups[weight_type].append((layer_idx, tensor))

    # Sort each group by layer index
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups


def analyze_cross_layer_similarity(
    weight_group: List[Tuple[int, torch.Tensor]],
    device: str = "cuda",
) -> dict:
    """Analyze how similar a weight type is across layers.

    Returns statistics about cross-layer redundancy.
    """
    if len(weight_group) < 2:
        return {"n_layers": len(weight_group), "similarity": 0.0}

    # Filter to same-shape tensors only
    target_shape = weight_group[0][1].shape
    weight_group = [(i, t) for i, t in weight_group if t.shape == target_shape]
    if len(weight_group) < 2:
        return {"n_layers": len(weight_group), "similarity": 0.0}

    tensors = [t.float().to(device) for _, t in weight_group]

    # Compute pairwise cosine similarities
    flat = torch.stack([t.reshape(-1) for t in tensors])  # (L, N)
    norms = flat.norm(dim=1, keepdim=True)
    cosine_matrix = (flat @ flat.t()) / (norms @ norms.t() + 1e-10)

    # Mean of all same-type weights
    mean_weight = flat.mean(dim=0)
    mean_norm = mean_weight.norm()

    # How much each layer deviates from the mean
    deviations = flat - mean_weight.unsqueeze(0)
    relative_devs = deviations.norm(dim=1) / (mean_norm + 1e-10)

    # SVD of the stacked weights to find principal components
    # Center first. Use the layer dimension (small) not the weight dimension.
    # centered is (L, N) where L << N, so we compute L x L covariance instead.
    centered = flat - mean_weight.unsqueeze(0)
    # Gram matrix approach: (L, L) instead of (N, N) SVD
    gram = centered @ centered.t()  # (L, L) — tiny!
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    # Sort descending
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    S = eigenvalues.clamp(min=0).sqrt()

    # How many components explain 99% of variance?
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    for k99 in range(len(S)):
        if cumvar[k99] >= 0.99:
            break
    k99 += 1

    # Same for 99.9%
    for k999 in range(len(S)):
        if cumvar[k999] >= 0.999:
            break
    k999 += 1

    return {
        "n_layers": len(weight_group),
        "mean_pairwise_cosine": cosine_matrix.mean().item(),
        "min_pairwise_cosine": cosine_matrix.min().item(),
        "mean_deviation_ratio": relative_devs.mean().item(),
        "max_deviation_ratio": relative_devs.max().item(),
        "components_99pct": k99,
        "components_999pct": k999,
        "top_sv_ratio": (S[0] / S.sum()).item() if len(S) > 0 else 0,
        "singular_values": S[:10].cpu().tolist(),
    }


def compress_cross_layer_svd(
    weight_group: List[Tuple[int, torch.Tensor]],
    n_components: int = None,
    variance_target: float = 0.999,
    device: str = "cuda",
) -> Tuple[CrossLayerCompressed, CrossLayerResult]:
    """Compress a weight type across layers using cross-layer SVD.

    Stack all L copies of a weight matrix, run SVD on the stack,
    keep k components that explain variance_target of the variance.

    Storage: k basis matrices + L*k coefficients + residuals
    Original: L matrices

    For k=3 and L=32: 3 + 32*3/(m*n) ~ 3 matrices instead of 32
    That's ~10x compression before any per-weight quantization.
    """
    if len(weight_group) < 2:
        raise ValueError("Need at least 2 layers for cross-layer compression")

    layer_indices = [idx for idx, _ in weight_group]
    tensors = [t.float().to(device) for _, t in weight_group]
    original_shape = tuple(tensors[0].shape)
    n_elements = tensors[0].numel()
    L = len(tensors)

    # Stack all layers: (L, m*n)
    stacked = torch.stack([t.reshape(-1) for t in tensors])

    # Compute mean (the "base" weight pattern)
    mean_weight = stacked.mean(dim=0)

    # Center the data
    centered = stacked - mean_weight.unsqueeze(0)

    # SVD of centered stack
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Determine number of components
    if n_components is None:
        total_var = (S ** 2).sum()
        cumvar = torch.cumsum(S ** 2, dim=0) / total_var
        mask = cumvar >= variance_target
        if mask.any():
            n_components = mask.float().argmax().item() + 1
        else:
            n_components = len(S)
        n_components = max(1, min(n_components, L - 1))

    # Truncate to k components
    U_k = U[:, :n_components]  # (L, k) — per-layer coefficients
    S_k = S[:n_components]     # (k,)
    Vt_k = Vt[:n_components]   # (k, m*n) — basis vectors

    # Absorb singular values into basis vectors
    basis_flat = S_k.unsqueeze(1) * Vt_k  # (k, m*n)
    coefficients = U_k  # (L, k)

    # Reconstruct and measure quality
    reconstructed_centered = coefficients @ basis_flat  # (L, m*n)
    reconstructed = reconstructed_centered + mean_weight.unsqueeze(0)

    cosine_sims = []
    relative_errors = []
    for i in range(L):
        orig = stacked[i]
        recon = reconstructed[i]
        cos = torch.nn.functional.cosine_similarity(
            orig.unsqueeze(0), recon.unsqueeze(0)
        ).item()
        rel_err = torch.norm(orig - recon).item() / (torch.norm(orig).item() + 1e-10)
        cosine_sims.append(cos)
        relative_errors.append(rel_err)

    # Build basis matrices (mean + k components)
    basis = [mean_weight.reshape(original_shape)]
    for i in range(n_components):
        basis.append(basis_flat[i].reshape(original_shape))

    # Compute storage: (1 + k) full matrices + L*k coefficients
    # In practice, the basis matrices and coefficients will also be compressed
    original_bytes = L * n_elements * 2  # FP16
    # Mean (FP16) + k basis (FP16) + L*k coefficients (FP16)
    compressed_bytes = (1 + n_components) * n_elements * 2 + L * n_components * 2
    variance_explained = (S_k ** 2).sum().item() / ((S ** 2).sum().item() + 1e-10)

    # Compute residuals (optional, for higher quality)
    residual_flat = stacked - reconstructed
    residuals = [residual_flat[i].reshape(original_shape) for i in range(L)]

    result = CrossLayerResult(
        weight_key=weight_group[0][1].__class__.__name__,
        n_layers=L,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        cosine_sims=cosine_sims,
        relative_errors=relative_errors,
        n_components=n_components,
        variance_explained=variance_explained,
    )

    compressed = CrossLayerCompressed(
        weight_key="",
        basis=basis,
        coefficients=coefficients,
        residuals=residuals,
        original_shape=original_shape,
        n_layers=L,
        n_components=n_components,
    )

    return compressed, result


def compress_model_cross_layer(
    named_weights: list,
    variance_target: float = 0.999,
    device: str = "cuda",
) -> Dict[str, Tuple[CrossLayerCompressed, CrossLayerResult]]:
    """Compress an entire model using cross-layer sharing.

    Groups weights by type, then applies cross-layer SVD to each group.
    Returns per-type compressed representations.
    """
    groups = group_weights_by_type(named_weights)
    results = {}

    for weight_type, group in groups.items():
        if len(group) < 2:
            continue

        # All tensors in group must have same shape
        shapes = set(tuple(t.shape) for _, t in group)
        if len(shapes) > 1:
            continue  # Skip mixed shapes

        try:
            compressed, result = compress_cross_layer_svd(
                group, variance_target=variance_target, device=device,
            )
            compressed.weight_key = weight_type
            result.weight_key = weight_type
            results[weight_type] = (compressed, result)
        except Exception as e:
            print(f"  Skip {weight_type}: {e}")

    return results
