"""
Stage 2: Low-Rank SVD Factorization (v2 — Energy-Adaptive)

Key change from v1: Instead of a fixed rank fraction, we keep enough
singular values to retain a target fraction of the Frobenius norm energy.

For 99.9% energy retention, most weight matrices need surprisingly few
singular values — the spectrum is typically heavy-tailed.

Also supports multi-level residual factorization:
  W ≈ U1@V1 + U2@V2 + ... (each level captures more detail)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class FactorizedWeight:
    """Low-rank factorization of a weight matrix."""
    U: torch.Tensor    # (m, r) - left singular vectors scaled by sqrt(S)
    V: torch.Tensor    # (r, n) - right singular vectors scaled by sqrt(S)
    rank: int
    original_shape: tuple
    energy_retained: float


@dataclass
class ResidualFactorization:
    """Multi-level residual factorization: W ≈ sum(U_i @ V_i)"""
    levels: List[FactorizedWeight] = field(default_factory=list)
    original_shape: tuple = ()
    total_energy_retained: float = 0.0

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct by summing all levels."""
        W = torch.zeros(self.levels[0].U.shape[0], self.levels[0].V.shape[1],
                        device=self.levels[0].U.device)
        for level in self.levels:
            W = W + level.U @ level.V
        if len(self.original_shape) != 2:
            W = W.reshape(self.original_shape)
        return W

    @property
    def total_rank(self) -> int:
        return sum(l.rank for l in self.levels)

    def total_elements(self) -> int:
        return sum(l.U.numel() + l.V.numel() for l in self.levels)


def factorize_weight(
    weight: torch.Tensor,
    energy_target: float = 0.999,
    min_rank: int = 4,
    max_rank: int = 1024,
    device: str = "cuda",
) -> FactorizedWeight:
    """
    Decompose a weight matrix via truncated SVD with energy-adaptive rank.

    Keeps the minimum number of singular values needed to retain
    `energy_target` fraction of the total Frobenius norm energy.
    """
    w = weight.float()
    original_shape = tuple(w.shape)

    if w.ndim < 2:
        return FactorizedWeight(
            U=w.unsqueeze(1).to(device),
            V=torch.ones(1, 1, device=device),
            rank=1, original_shape=original_shape, energy_retained=1.0,
        )

    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)

    m, n = w.shape
    full_rank = min(m, n)
    w_dev = w.to(device)

    # Adaptive randomized SVD:
    # Start with a moderate probe, check if it captures enough energy.
    # If not, increase rank up to full_rank.
    # Randomized SVD at rank r on (m,n) matrix is O(mn*r), much faster
    # than full SVD at O(mn*min(m,n)).
    probe_rank = min(max_rank, full_rank)

    if probe_rank >= full_rank * 0.8:
        # If we'd need most of the rank anyway, just do full SVD
        # (randomized SVD with q close to full_rank is slower than full)
        U, S, Vt = torch.linalg.svd(w_dev, full_matrices=False)
    else:
        U, S, Vt = torch.svd_lowrank(w_dev, q=probe_rank, niter=4)
        Vt = Vt.t()  # svd_lowrank returns V, not V^T

    # Find rank needed for target energy
    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    energy_fractions = cumulative_energy / total_energy

    # Find minimum rank where we hit the energy target
    mask = energy_fractions >= energy_target
    if mask.any():
        target_rank = mask.float().argmax().item() + 1
    else:
        target_rank = len(S)

    target_rank = max(min_rank, min(target_rank, max_rank, len(S)))

    # Truncate
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vt_r = Vt[:target_rank, :]

    energy_retained = (S_r ** 2).sum().item() / total_energy.item()

    # Absorb sqrt(S) into both factors
    sqrt_S = torch.sqrt(S_r)
    U_scaled = U_r * sqrt_S.unsqueeze(0)
    V_scaled = Vt_r * sqrt_S.unsqueeze(1)

    return FactorizedWeight(
        U=U_scaled, V=V_scaled,
        rank=target_rank, original_shape=original_shape,
        energy_retained=energy_retained,
    )


def factorize_residual(
    weight: torch.Tensor,
    n_levels: int = 3,
    energy_targets: list = None,
    min_rank: int = 2,
    max_rank: int = 512,
    device: str = "cuda",
) -> ResidualFactorization:
    """
    Multi-level residual factorization.

    Level 1: Capture 99% of energy with aggressive low-rank
    Level 2: Capture 99% of the RESIDUAL's energy
    Level 3: Capture 99% of the remaining residual
    ...

    The compounding effect: 0.99 * 0.99 * 0.99 = 0.9703 total energy,
    but with much lower total rank than a single 99.7% factorization.
    Each residual level has a flatter spectrum → compresses better.
    """
    if energy_targets is None:
        # Progressively capture more detail
        energy_targets = [0.99, 0.995, 0.999][:n_levels]

    w = weight.float()
    original_shape = tuple(w.shape)

    if w.ndim < 2:
        single = FactorizedWeight(
            U=w.unsqueeze(1).to(device),
            V=torch.ones(1, 1, device=device),
            rank=1, original_shape=original_shape, energy_retained=1.0,
        )
        return ResidualFactorization(
            levels=[single], original_shape=original_shape,
            total_energy_retained=1.0,
        )

    if w.ndim > 2:
        w = w.reshape(w.shape[0], -1)

    w_dev = w.to(device)
    residual = w_dev.clone()
    original_energy = torch.norm(w_dev, p="fro").item() ** 2

    result = ResidualFactorization(original_shape=original_shape)

    for level_idx, energy_target in enumerate(energy_targets):
        if torch.norm(residual, p="fro").item() ** 2 < original_energy * 1e-8:
            break  # Residual is negligible

        level = factorize_weight(
            residual, energy_target=energy_target,
            min_rank=min_rank,
            max_rank=max_rank // (level_idx + 1),  # Smaller ranks for residuals
            device=device,
        )
        level.original_shape = original_shape
        result.levels.append(level)

        # Compute residual for next level
        reconstructed = level.U @ level.V
        residual = residual - reconstructed

    # Calculate total energy retained
    total_reconstructed = torch.zeros_like(w_dev)
    for level in result.levels:
        total_reconstructed = total_reconstructed + level.U @ level.V
    retained = torch.norm(total_reconstructed, p="fro").item() ** 2
    result.total_energy_retained = retained / original_energy

    return result


def reconstruct_from_factors(factorized: FactorizedWeight) -> torch.Tensor:
    """Reconstruct the full weight matrix from low-rank factors."""
    W = factorized.U @ factorized.V
    if len(factorized.original_shape) != 2:
        W = W.reshape(factorized.original_shape)
    return W
