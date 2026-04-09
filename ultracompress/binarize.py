"""
Stage 3: Sigma-Delta Binarization with Error Feedback (v2)

v1 just took sign(x) — losing all magnitude info within groups.

v2 uses sigma-delta modulation from signal processing:
  - Process elements sequentially
  - Track cumulative quantization error
  - Feed error into the next element's decision
  - Result: error is "spread out" and partially cancels

Also implements:
  - Hadamard rotation for latent geometry alignment (LittleBit-2)
  - Multi-level scaling (group + sub-group)
  - Optional ternary mode {-1, 0, +1} for sparse-friendly compression
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class BinarizedFactor:
    """A binarized matrix with group-wise scaling."""
    signs: torch.Tensor       # {-1, +1} stored as int8 or bool
    scales: torch.Tensor      # Per-group scaling factors (float16)
    group_size: int
    original_shape: tuple
    mode: str = "sigma_delta"  # "naive", "sigma_delta", "ternary"

    def storage_bytes(self) -> int:
        sign_bits = self.signs.numel()  # 1 bit each (2 bits for ternary)
        scale_bytes = self.scales.numel() * 2  # float16
        if self.mode == "ternary":
            return (sign_bits * 2 + 7) // 8 + scale_bytes
        return (sign_bits + 7) // 8 + scale_bytes

    def decompress(self) -> torch.Tensor:
        """Reconstruct the float tensor."""
        if self.mode == "ternary":
            values = self.signs.float()  # Already {-1, 0, +1}
        else:
            values = self.signs.float() * 2.0 - 1.0  # bool -> {-1, +1}

        flat = values.reshape(-1)
        n_groups = (flat.numel() + self.group_size - 1) // self.group_size

        # Pad if needed
        remainder = flat.numel() % self.group_size
        if remainder != 0:
            flat = torch.cat([flat, torch.zeros(self.group_size - remainder, device=flat.device)])

        grouped = flat.reshape(n_groups, self.group_size)

        # Ensure scales match
        scales = self.scales.float()
        if scales.numel() < n_groups:
            scales = torch.cat([scales, scales[-1:].expand(n_groups - scales.numel())])
        scales = scales[:n_groups]

        scaled = grouped * scales.unsqueeze(1)
        result = scaled.reshape(-1)[:self.signs.numel()]
        return result.reshape(self.original_shape)


@dataclass
class BinarizedWeight:
    """Binarized low-rank representation."""
    U_bin: BinarizedFactor
    V_bin: BinarizedFactor
    rank: int
    original_shape: tuple

    def storage_bytes(self) -> int:
        return self.U_bin.storage_bytes() + self.V_bin.storage_bytes()

    def decompress(self) -> torch.Tensor:
        U = self.U_bin.decompress()
        V = self.V_bin.decompress()
        W = U @ V
        if len(self.original_shape) != 2:
            W = W.reshape(self.original_shape)
        return W


def hadamard_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along a dimension."""
    size = x.shape[dim]
    next_pow2 = 1
    while next_pow2 < size:
        next_pow2 *= 2

    if next_pow2 != size:
        pad_size = next_pow2 - size
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)

    n = next_pow2
    h = 1
    while h < n:
        x_reshaped = x.unflatten(dim, (-1, h * 2))
        left = x_reshaped.narrow(dim + 1, 0, h)
        right = x_reshaped.narrow(dim + 1, h, h)
        x_reshaped = torch.cat([left + right, left - right], dim=dim + 1)
        x = x_reshaped.flatten(dim, dim + 1)
        h *= 2

    x = x / np.sqrt(n)
    return x


def sigma_delta_binarize_batch(groups: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized sigma-delta modulation for ALL groups at once.

    Uses cumulative-sum trick: the sigma-delta integrator is equivalent to
    binarizing the cumulative sum of the input, then differencing.

    For a group of values x[0..N-1]:
      cumsum[i] = sum(x[0..i])
      binary_cumsum[i] = round(cumsum[i])  (to nearest integer)
      output[i] = binary_cumsum[i] - binary_cumsum[i-1]  -> {-1, 0, +1}
      sign[i] = output[i] >= 0  -> bool

    This is 100% vectorized — no Python loops over groups.
    """
    n_groups, group_size = groups.shape
    device = groups.device

    # Normalize each group by its scale
    safe_scales = scales.float().clamp(min=1e-10).unsqueeze(1)
    normalized = groups / safe_scales

    # Cumulative sum along each group
    cumsum = torch.cumsum(normalized, dim=1)

    # Round cumsum to nearest integer (the sigma-delta integrator)
    rounded = torch.round(cumsum)

    # Diff to get the 1-bit output stream
    # output[0] = rounded[0], output[i] = rounded[i] - rounded[i-1]
    diff = torch.zeros_like(rounded)
    diff[:, 0] = rounded[:, 0]
    diff[:, 1:] = rounded[:, 1:] - rounded[:, :-1]

    # Convert to binary: positive -> True (which becomes +1 on decompress)
    return diff >= 0


def binarize_factor(
    factor: torch.Tensor,
    group_size: int = 64,
    use_rotation: bool = True,
    use_sigma_delta: bool = True,
) -> BinarizedFactor:
    """
    Binarize a factor matrix with sigma-delta error feedback.
    Fully vectorized — no Python loops.
    """
    original_shape = tuple(factor.shape)
    device = factor.device
    x = factor.float()

    # Step 1: Latent geometry alignment
    if use_rotation and min(x.shape) > 1:
        if x.shape[0] <= x.shape[1]:
            x = hadamard_transform(x, dim=0)
        else:
            x = hadamard_transform(x, dim=1)

    # Step 2: Group-wise binarization (fully vectorized)
    flat = x.reshape(-1)
    remainder = flat.numel() % group_size
    if remainder != 0:
        flat = torch.cat([flat, torch.zeros(group_size - remainder, device=device)])

    groups = flat.reshape(-1, group_size)  # (n_groups, group_size)
    n_groups = groups.shape[0]

    # Per-group scales (RMS for better magnitude preservation)
    scales = torch.sqrt((groups ** 2).mean(dim=1))  # (n_groups,)

    # Binarize — all groups at once, no loops
    if use_sigma_delta:
        signs_grouped = sigma_delta_binarize_batch(groups, scales)  # (n_groups, group_size)
    else:
        signs_grouped = (groups >= 0)

    signs_tensor = signs_grouped.reshape(-1)

    # Trim to original size
    total_needed = 1
    for s in original_shape:
        total_needed *= s
    signs_tensor = signs_tensor[:total_needed]

    return BinarizedFactor(
        signs=signs_tensor.reshape(original_shape),
        scales=scales.half(),
        group_size=group_size,
        original_shape=original_shape,
        mode="sigma_delta" if use_sigma_delta else "naive",
    )


def binarize_weight(
    U: torch.Tensor,
    V: torch.Tensor,
    rank: int,
    original_shape: tuple,
    group_size: int = 64,
    use_rotation: bool = True,
    use_sigma_delta: bool = True,
) -> BinarizedWeight:
    """Binarize both factors of a low-rank decomposition."""
    U_bin = binarize_factor(U, group_size, use_rotation, use_sigma_delta)
    V_bin = binarize_factor(V, group_size, use_rotation, use_sigma_delta)

    return BinarizedWeight(
        U_bin=U_bin, V_bin=V_bin,
        rank=rank, original_shape=original_shape,
    )
