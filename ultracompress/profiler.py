"""
Stage 1: Sensitivity-Aware Layer Profiling

Analyzes each weight matrix to determine how many bits it needs.
Sensitive layers (attention Q/K/V, first/last layers) get more bits.
Redundant layers (MLP intermediates) get fewer bits.

Uses weight magnitude statistics + spectral analysis as a proxy for
Fisher information (which would require calibration data).
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class LayerProfile:
    """Sensitivity profile for a single weight tensor."""
    name: str
    shape: tuple
    n_params: int
    # Spectral properties
    spectral_entropy: float  # How spread out singular values are (0=concentrated, 1=uniform)
    top_sv_ratio: float      # Ratio of top singular value to sum (dominance)
    effective_rank: float    # Numerical rank / actual rank
    # Magnitude properties
    weight_norm: float
    sparsity: float          # Fraction of near-zero weights
    # Assigned compression budget
    target_bpw: float = 0.34
    rank_fraction: float = 0.1  # What fraction of full rank to keep in SVD


def profile_layer(name: str, weight: torch.Tensor, global_target_bpw: float = 0.34) -> LayerProfile:
    """Profile a weight matrix and assign compression budget."""
    w = weight.float()
    shape = tuple(w.shape)
    n_params = w.numel()

    if w.ndim < 2:
        # Bias or 1D tensor — low sensitivity, just quantize
        return LayerProfile(
            name=name, shape=shape, n_params=n_params,
            spectral_entropy=0.0, top_sv_ratio=1.0, effective_rank=1.0,
            weight_norm=torch.norm(w).item(),
            sparsity=(w.abs() < 1e-6).float().mean().item(),
            target_bpw=max(global_target_bpw, 1.0),  # biases need at least 1 bit
            rank_fraction=1.0,
        )

    # Reshape to 2D if needed (conv layers etc)
    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)

    # Spectral analysis via SVD
    try:
        S = torch.linalg.svdvals(w)
    except Exception:
        S = torch.ones(min(w.shape))

    S_normalized = S / (S.sum() + 1e-10)
    spectral_entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum().item()
    max_entropy = np.log(len(S))
    spectral_entropy = spectral_entropy / (max_entropy + 1e-10)

    top_sv_ratio = (S[0] / (S.sum() + 1e-10)).item()

    # Effective rank: exponential of spectral entropy
    effective_rank = np.exp(spectral_entropy * max_entropy) / len(S)

    weight_norm = torch.norm(w).item()
    sparsity = (w.abs() < 1e-6).float().mean().item()

    # --- Bit budget allocation ---
    # Layers with concentrated spectra (low entropy, high top_sv_ratio)
    # compress better → assign fewer bits
    # Layers with spread spectra need more bits to preserve info

    sensitivity = spectral_entropy * (1.0 - sparsity)

    # Detect layer type from name for heuristic adjustments
    is_embedding = "embed" in name.lower()
    is_head = "head" in name.lower() or "lm_head" in name.lower()
    is_qkv = any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "qkv"])
    is_output = "o_proj" in name.lower() or "out_proj" in name.lower()
    is_gate = "gate" in name.lower()

    # Critical layers get 2-3x the bit budget
    if is_embedding or is_head:
        sensitivity_mult = 3.0
    elif is_qkv:
        sensitivity_mult = 2.0
    elif is_output or is_gate:
        sensitivity_mult = 1.5
    else:
        sensitivity_mult = 1.0

    adjusted_sensitivity = sensitivity * sensitivity_mult

    # Map sensitivity to BPW: range [0.1, 2.0] around the global target
    # More sensitive → more bits
    target_bpw = global_target_bpw * (0.5 + 2.0 * adjusted_sensitivity)
    target_bpw = np.clip(target_bpw, 0.1, 2.0)

    # Map to SVD rank fraction: more bits → keep more singular values
    # At 0.1 BPW we keep ~2% of rank, at 2.0 BPW we keep ~30%
    rank_fraction = 0.02 + (target_bpw / 2.0) * 0.28
    rank_fraction = np.clip(rank_fraction, 0.01, 0.5)

    return LayerProfile(
        name=name, shape=shape, n_params=n_params,
        spectral_entropy=spectral_entropy,
        top_sv_ratio=top_sv_ratio,
        effective_rank=effective_rank,
        weight_norm=weight_norm,
        sparsity=sparsity,
        target_bpw=target_bpw,
        rank_fraction=rank_fraction,
    )


def profile_model(named_weights: list, global_target_bpw: float = 0.34) -> list:
    """
    Profile all layers in a model.

    Args:
        named_weights: list of (name, tensor) pairs
        global_target_bpw: target average bits per weight

    Returns:
        list of LayerProfile objects
    """
    profiles = []
    for name, weight in named_weights:
        prof = profile_layer(name, weight, global_target_bpw)
        profiles.append(prof)

    # Normalize bit budgets so the weighted average matches the global target
    total_params = sum(p.n_params for p in profiles)
    current_avg_bpw = sum(p.target_bpw * p.n_params for p in profiles) / total_params

    if current_avg_bpw > 0:
        scale = global_target_bpw / current_avg_bpw
        for p in profiles:
            p.target_bpw = np.clip(p.target_bpw * scale, 0.05, 4.0)
            p.rank_fraction = 0.02 + (p.target_bpw / 2.0) * 0.28
            p.rank_fraction = np.clip(p.rank_fraction, 0.01, 0.5)

    return profiles
