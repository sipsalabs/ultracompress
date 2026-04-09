"""
Calibration-Aware Product Quantization — Minimize OUTPUT error, not WEIGHT error.

The fundamental insight:
  Standard PQ minimizes: ||W - W'||  (weight reconstruction error)
  We should minimize:    ||XW - XW'|| (output reconstruction error)

Why this matters:
  A weight matrix W has dimensions (out, in). Not all input directions
  are equally important — some columns of W multiply high-activation
  inputs, others multiply near-zero inputs.

  If we know which input dimensions have high activation variance (the
  Hessian diagonal H = diag(X^T X / n)), we can weight the PQ objective:

  minimize ||sqrt(H) * (W - W')||

  This is equivalent to doing PQ in a "warped" space where important
  dimensions are stretched. The codebooks automatically allocate more
  precision to high-activation columns.

For a model where 80% of activations are near-zero (common with ReLU/GELU),
this can turn a 0.90 weight cosine into a 0.99+ output cosine — because
the 10% weight error is concentrated in dimensions the model doesn't use.

This is the core idea behind GPTQ, AWQ, and AQLM. We apply it to our
binary PQ framework for extreme compression with high output quality.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .product_quantize import ProductQuantized


@dataclass
class CalibratedPQ(ProductQuantized):
    """PQ with calibration-aware codebooks."""
    importance_weights: Optional[torch.Tensor] = None  # Per-column importance


def collect_activation_stats(
    weights: list,
    n_calibration_tokens: int = 256,
    seq_len: int = 32,
    vocab_size: int = 151936,
    device: str = "cuda",
) -> dict:
    """Compute per-column importance for each weight matrix.

    Combines two signals:
    1. Weight column norms: columns with large norms amplify their input more,
       so errors in those columns produce larger output errors.
    2. Embedding-derived activation variance: which input dimensions are
       activated more by typical inputs (Hessian diagonal proxy).

    The combination (AWQ-style): importance = activation_variance * weight_col_norm
    This captures both "which inputs fire" and "which columns matter when they fire".
    """
    stats = {}

    # Find embedding weight for activation stats
    embed_weight = None
    for name, w in weights:
        if 'embd' in name or 'embed' in name:
            embed_weight = w.float().to(device)
            break

    # Compute embedding activation variance (Hessian diagonal proxy)
    embed_h_diag = None
    if embed_weight is not None:
        tokens = torch.randint(0, min(vocab_size, embed_weight.shape[0]),
                              (n_calibration_tokens, seq_len), device=device)
        with torch.no_grad():
            x = torch.nn.functional.embedding(tokens, embed_weight)
            x_flat = x.reshape(-1, x.shape[-1])
            # Per-channel activation variance (Hessian diagonal)
            embed_h_diag = (x_flat ** 2).mean(dim=0)  # (embed_dim,)

    for name, w in weights:
        if w.ndim < 2 or w.numel() < 1024:
            continue

        w_f = w.float().to(device)
        in_dim = w_f.shape[1] if w_f.ndim == 2 else w_f.shape[-1]

        # Signal 1: Weight column norms — which columns amplify their input most
        if w_f.ndim == 2:
            col_norms = w_f.norm(dim=0)  # (in_dim,) — L2 norm of each column
        else:
            col_norms = w_f.reshape(-1, in_dim).norm(dim=0)

        # Signal 2: Activation variance (if dimensions match embedding)
        if embed_h_diag is not None and embed_h_diag.shape[0] == in_dim:
            act_var = embed_h_diag
        else:
            act_var = torch.ones(in_dim, device=device)

        # AWQ-style combination: importance = act_variance * col_norm
        importance = act_var * col_norms
        importance = importance.clamp(min=1e-10)

        stats[name] = importance

    return stats


def calibrated_product_quantize(
    weight: torch.Tensor,
    importance: torch.Tensor,
    n_subvectors: int = 8,
    codebook_size: int = 4,
    group_size: int = 256,
    n_iter: int = 20,
    importance_power: float = 0.5,
) -> CalibratedPQ:
    """Product Quantization with importance-weighted k-means.

    Correct approach (GPTQ/AWQ-style):
      1. Group and normalize the ORIGINAL weight (identical to standard PQ)
      2. Compute per-element importance weights for each sub-vector
      3. Scale sub-vectors by sqrt(importance) ONLY for k-means distance
      4. Compute centroids from UNSCALED data using importance-aware assignments

    This ensures:
      - Scales and group structure are identical to standard PQ (correct decompression)
      - k-means allocates codebook precision to important dimensions
      - Codebook entries are valid centroids of original data

    Args:
        weight: Input tensor (out_dim, in_dim)
        importance: Per-column importance weights (in_dim,)
        n_subvectors: M
        codebook_size: K
        group_size: G
        n_iter: k-means iterations
        importance_power: How aggressively to weight (0=uniform, 1=full, 0.5=sqrt)
    """
    original_shape = tuple(weight.shape)
    n_elements = weight.numel()
    device = weight.device

    assert group_size % n_subvectors == 0
    sub_vector_size = group_size // n_subvectors

    # === Step 1: Standard PQ grouping (identical to product_quantize) ===
    flat = weight.float().reshape(-1)

    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)
    n_groups = groups.shape[0]

    scales = groups.norm(dim=1).clamp(min=1e-10) / np.sqrt(group_size)
    normalized = groups / scales.unsqueeze(1)

    sub_vectors = normalized.reshape(n_groups, n_subvectors, sub_vector_size)

    # === Step 2: Build per-element importance for each sub-vector position ===
    imp = importance.float().to(device)
    imp = (imp / imp.max()).clamp(min=0.01)
    imp = imp.pow(importance_power)  # sqrt for moderate weighting

    # Tile importance to match flattened weight layout
    if weight.ndim == 2:
        # Weight is (out, in) — importance applies to in dimension (columns)
        # Flattened row-major: element i has importance imp[i % in_dim]
        in_dim = weight.shape[1]
        imp_flat = imp.repeat((flat.numel() + in_dim - 1) // in_dim)[:flat.numel()]
    else:
        imp_flat = torch.ones(flat.numel(), device=device)

    # Pad importance to match padded flat
    if imp_flat.numel() < groups.numel():
        imp_flat = torch.cat([imp_flat, torch.ones(groups.numel() - imp_flat.numel(), device=device)])

    # Reshape importance into sub-vector structure: (n_groups, M, sub_vector_size)
    imp_sv = imp_flat.reshape(n_groups, n_subvectors, sub_vector_size)

    actual_k = min(codebook_size, n_groups)

    # === Step 3: Importance-weighted k-means ===
    # Assignment uses weighted distance: d(x,c) = sum(imp * (x-c)^2)
    # Centroids are computed from UNWEIGHTED data
    codebooks = []
    all_indices = []

    for m in range(n_subvectors):
        data = sub_vectors[:, m, :]            # (n_groups, svs) — unscaled
        imp_m = imp_sv[:, m, :]                # (n_groups, svs) — per-element importance

        # Initialize codebook from random samples
        perm = torch.randperm(n_groups, device=device)[:actual_k]
        codebook = data[perm].clone()

        chunk_size = min(n_groups, 100000)

        for _ in range(n_iter):
            # Importance-weighted assignment:
            # d(x, c) = sum_j imp_j * (x_j - c_j)^2
            #          = sum_j imp_j * x_j^2 - 2 * sum_j imp_j * x_j * c_j + sum_j imp_j * c_j^2
            # The third term depends on both the data point (via imp) and codebook entry.
            # For efficiency with per-point importance, we compute it in chunks.
            indices = torch.zeros(n_groups, device=device, dtype=torch.int64)

            for start in range(0, n_groups, chunk_size):
                end = min(start + chunk_size, n_groups)
                chunk = data[start:end]          # (cs, svs)
                chunk_imp = imp_m[start:end]     # (cs, svs)

                # Weighted squared norms: sum(imp * x^2) per data point
                chunk_wsq = (chunk_imp * chunk ** 2).sum(dim=1)  # (cs,)

                # Weighted cross terms: sum(imp * x * c) = (imp * x) @ c^T
                weighted_chunk = chunk_imp * chunk  # (cs, svs)
                dots = weighted_chunk @ codebook.t()  # (cs, K)

                # Weighted codebook norms: sum(imp * c^2) — depends on data point's imp
                # cb_wsq[i, k] = sum_j imp_m[i,j] * codebook[k,j]^2
                cb_sq = codebook ** 2  # (K, svs)
                cb_wsq = chunk_imp @ cb_sq.t()  # (cs, K)

                dists = chunk_wsq.unsqueeze(1) + cb_wsq - 2 * dots  # (cs, K)
                indices[start:end] = dists.argmin(dim=1)

            # Centroids from UNWEIGHTED data (correct for decompression)
            sums = torch.zeros(actual_k, sub_vector_size, device=device)
            counts = torch.zeros(actual_k, device=device)
            sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, sub_vector_size), data)
            counts.scatter_add_(0, indices, torch.ones(n_groups, device=device))
            valid = counts > 0
            if valid.any():
                codebook[valid] = sums[valid] / counts[valid].unsqueeze(1)

        codebooks.append(codebook.half())
        all_indices.append(indices)

    indices_tensor = torch.stack(all_indices, dim=1).short()

    return CalibratedPQ(
        codebooks=codebooks,
        indices=indices_tensor,
        scales=scales.half(),
        n_subvectors=n_subvectors,
        codebook_size=actual_k,
        group_size=group_size,
        sub_vector_size=sub_vector_size,
        original_shape=original_shape,
        n_elements=n_elements,
        importance_weights=importance,
    )
