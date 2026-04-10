"""
Differentiable Product Quantization — Enabling End-to-End Gradient Flow

Standard PQ uses hard assignments (argmin over codebook entries), which
blocks gradients from flowing back to the weight generator. This module
provides differentiable approximations that enable joint optimization
of the generator and PQ codebooks.

Key components:
  1. Soft PQ Assignment via Gumbel-Softmax
  2. Differentiable Entropy Loss
  3. Straight-Through Estimator for hard PQ with soft gradients

Why this matters:
  The Weight Genome trains a generator to predict weights. If we only
  minimize ||W - W_gen||, the generator doesn't know which errors PQ
  can fix and which it can't. With differentiable PQ, the generator
  learns to produce residuals that are EASY to quantize — low entropy
  indices, clustered sub-vectors, smooth error surfaces.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def soft_pq_assign(
    sub_vectors: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable PQ assignment using softmax over distances.

    Args:
        sub_vectors: (n_groups, sub_vector_size) — data to quantize
        codebook: (K, sub_vector_size) — codebook entries
        temperature: lower = harder assignments, higher = softer

    Returns:
        soft_recon: (n_groups, sub_vector_size) — differentiable reconstruction
        soft_assignments: (n_groups, K) — soft assignment probabilities
    """
    # Compute squared distances: (n_groups, K)
    # d(x,c) = ||x||^2 - 2*x@c^T + ||c||^2
    x_sq = (sub_vectors ** 2).sum(dim=1, keepdim=True)   # (n_groups, 1)
    c_sq = (codebook ** 2).sum(dim=1, keepdim=True).t()  # (1, K)
    dots = sub_vectors @ codebook.t()                      # (n_groups, K)
    dists = x_sq + c_sq - 2 * dots                        # (n_groups, K)

    # Soft assignments via negative distance softmax
    soft_assignments = F.softmax(-dists / temperature, dim=1)  # (n_groups, K)

    # Soft reconstruction: weighted sum of codebook entries
    soft_recon = soft_assignments @ codebook  # (n_groups, sub_vector_size)

    return soft_recon, soft_assignments


def gumbel_pq_assign(
    sub_vectors: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PQ assignment using Gumbel-Softmax for better gradient estimation.

    Gumbel-Softmax adds noise that explores the assignment space,
    giving better gradient estimates than plain softmax.

    Args:
        sub_vectors: (n_groups, sub_vector_size)
        codebook: (K, sub_vector_size)
        temperature: Gumbel temperature
        hard: If True, use straight-through (hard forward, soft backward)

    Returns:
        recon: (n_groups, sub_vector_size)
        assignments: (n_groups, K) — one-hot if hard, soft otherwise
    """
    # Compute logits from negative distances
    x_sq = (sub_vectors ** 2).sum(dim=1, keepdim=True)
    c_sq = (codebook ** 2).sum(dim=1, keepdim=True).t()
    dots = sub_vectors @ codebook.t()
    dists = x_sq + c_sq - 2 * dots
    logits = -dists / max(temperature, 0.01)

    # Gumbel-Softmax
    assignments = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=1)

    # Reconstruct
    recon = assignments @ codebook

    return recon, assignments


def differentiable_entropy(soft_assignments: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute differentiable entropy of soft PQ assignments.

    Entropy H = -sum(p * log(p)) where p is the marginal distribution
    over codebook entries (averaged across all data points).

    Low entropy = few codebook entries used = easy to compress.
    High entropy = uniform usage = hard to compress.

    Args:
        soft_assignments: (n_groups, K) — per-data-point assignment probs

    Returns:
        entropy: scalar — bits per index
    """
    # Marginal distribution: average assignment probabilities across data
    marginal = soft_assignments.mean(dim=0)  # (K,)
    marginal = marginal.clamp(min=eps)

    # Shannon entropy in bits
    entropy = -(marginal * torch.log2(marginal)).sum()
    return entropy


def soft_pq_compress(
    weight: torch.Tensor,
    n_subvectors: int = 8,
    codebook_size: int = 4,
    group_size: int = 256,
    codebooks: list = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Full differentiable PQ compression pipeline.

    If codebooks are provided, uses them. Otherwise initializes from data.

    Returns:
        recon: reconstructed weight (same shape as input)
        total_entropy: entropy of all sub-vector assignments
        codebooks: list of M codebooks (for reuse)
    """
    device = weight.device
    flat = weight.float().reshape(-1)

    sub_vector_size = group_size // n_subvectors

    # Pad
    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)
    n_groups = groups.shape[0]

    # Per-group normalization
    scales = groups.norm(dim=1).clamp(min=1e-10) / np.sqrt(group_size)
    normalized = groups / scales.unsqueeze(1)

    sub_vectors = normalized.reshape(n_groups, n_subvectors, sub_vector_size)

    actual_k = min(codebook_size, n_groups)

    # Initialize codebooks if not provided
    if codebooks is None:
        codebooks = []
        for m in range(n_subvectors):
            data = sub_vectors[:, m, :].detach()
            perm = torch.randperm(n_groups, device=device)[:actual_k]
            codebooks.append(data[perm].clone().requires_grad_(True))

    # Soft PQ assignment per sub-vector
    all_recons = []
    total_entropy = torch.tensor(0.0, device=device)

    for m in range(n_subvectors):
        data = sub_vectors[:, m, :]  # (n_groups, svs)
        recon_m, assigns_m = soft_pq_assign(data, codebooks[m], temperature)
        all_recons.append(recon_m)
        total_entropy = total_entropy + differentiable_entropy(assigns_m)

    # Reconstruct groups
    recon_groups = torch.cat(all_recons, dim=1)  # (n_groups, group_size)
    recon_scaled = recon_groups * scales.unsqueeze(1)

    # Trim and reshape
    recon_flat = recon_scaled.reshape(-1)[:weight.numel()]
    recon = recon_flat.reshape(weight.shape)

    avg_entropy = total_entropy / n_subvectors
    return recon, avg_entropy, codebooks


def entropy_aware_loss(
    weight_pred: torch.Tensor,
    weight_true: torch.Tensor,
    activations: torch.Tensor = None,
    pq_config: tuple = (8, 4, 256),
    temperature: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Combined loss for Weight Genome training.

    L = alpha * MSE(W_true, W_pred)
      + beta  * entropy(PQ_indices(W_true - W_pred))
      + gamma * MSE(X @ W_true, X @ W_pred)

    The entropy term encourages the generator to produce predictions
    whose residuals have low-entropy PQ assignments — meaning PQ can
    compress them efficiently.

    Args:
        weight_pred: generated weight
        weight_true: target weight
        activations: optional input activations for output-aware loss
        pq_config: (M, K, G) for the soft PQ
        temperature: softmax temperature for PQ
        alpha: weight reconstruction loss coefficient
        beta: entropy loss coefficient
        gamma: output-aware loss coefficient

    Returns:
        total_loss: combined loss
        metrics: dict with individual loss components
    """
    M, K, G = pq_config

    # Weight MSE
    weight_mse = F.mse_loss(weight_pred, weight_true)

    # Residual entropy
    residual = weight_true - weight_pred
    _, entropy, _ = soft_pq_compress(
        residual, n_subvectors=M, codebook_size=K,
        group_size=G, temperature=temperature,
    )

    # Output-aware loss
    if activations is not None and activations.shape[-1] == weight_true.shape[1]:
        true_out = activations @ weight_true.t()
        pred_out = activations @ weight_pred.t()
        output_mse = F.mse_loss(pred_out, true_out)
    else:
        output_mse = torch.tensor(0.0, device=weight_pred.device)

    total_loss = alpha * weight_mse + beta * entropy + gamma * output_mse

    metrics = {
        'weight_mse': weight_mse.item(),
        'entropy_bpi': entropy.item(),
        'output_mse': output_mse.item(),
        'total_loss': total_loss.item(),
    }
    return total_loss, metrics
