"""Compression quality metrics and reporting."""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CompressionResult:
    """Tracks compression stats for a single weight tensor."""
    name: str
    original_shape: tuple
    original_bytes: int
    compressed_bytes: int
    # Quality metrics
    mse: float = 0.0
    cosine_sim: float = 0.0
    relative_error: float = 0.0
    # Per-stage breakdown
    stage_details: dict = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)

    @property
    def bits_per_weight(self) -> float:
        n_params = 1
        for s in self.original_shape:
            n_params *= s
        return (self.compressed_bytes * 8) / n_params


@dataclass
class ModelCompressionReport:
    """Aggregate report for an entire model."""
    layers: list = field(default_factory=list)

    @property
    def total_original_bytes(self) -> int:
        return sum(l.original_bytes for l in self.layers)

    @property
    def total_compressed_bytes(self) -> int:
        return sum(l.compressed_bytes for l in self.layers)

    @property
    def overall_ratio(self) -> float:
        return self.total_original_bytes / max(self.total_compressed_bytes, 1)

    @property
    def avg_bits_per_weight(self) -> float:
        total_params = sum(np.prod(l.original_shape) for l in self.layers)
        return (self.total_compressed_bytes * 8) / total_params

    @property
    def avg_cosine_sim(self) -> float:
        if not self.layers:
            return 0.0
        return np.mean([l.cosine_sim for l in self.layers])

    @property
    def avg_relative_error(self) -> float:
        if not self.layers:
            return 0.0
        return np.mean([l.relative_error for l in self.layers])

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  UltraCompress - Model Compression Report",
            "=" * 60,
            f"  Layers compressed:    {len(self.layers)}",
            f"  Original size:        {self.total_original_bytes / 1e9:.3f} GB",
            f"  Compressed size:      {self.total_compressed_bytes / 1e6:.1f} MB",
            f"  Compression ratio:    {self.overall_ratio:.1f}x",
            f"  Avg bits/weight:      {self.avg_bits_per_weight:.3f}",
            f"  Avg cosine sim:       {self.avg_cosine_sim:.6f}",
            f"  Avg relative error:   {self.avg_relative_error:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def extrapolate(self, target_params_billions: float, vram_budget_gb: float = 20.0) -> str:
        """Project compression results to a larger model."""
        if not self.layers:
            return "No data to extrapolate."
        bpw = self.avg_bits_per_weight
        target_params = target_params_billions * 1e9
        original_gb = (target_params * 16) / 8 / 1e9  # FP16
        compressed_gb = (target_params * bpw) / 8 / 1e9
        # What BPW would we need to hit the VRAM budget?
        required_bpw = (vram_budget_gb * 1e9 * 8) / target_params
        lines = [
            "-" * 60,
            f"  Extrapolation to {target_params_billions:,.0f}B params:",
            f"  FP16 size:          {original_gb:.1f} GB",
            f"  Compressed size:    {compressed_gb:.2f} GB",
            f"  At {bpw:.3f} BPW, cosine sim {self.avg_cosine_sim:.4f}",
            f"  Fits in {vram_budget_gb:.0f}GB VRAM?  "
            + (f"YES ({vram_budget_gb - compressed_gb:.1f} GB headroom)"
               if compressed_gb <= vram_budget_gb
               else f"NO — need {compressed_gb:.1f} GB, requires {required_bpw:.3f} BPW"),
            "-" * 60,
        ]
        return "\n".join(lines)


def compute_quality(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute quality metrics between original and reconstructed weights."""
    with torch.no_grad():
        orig_flat = original.float().flatten()
        recon_flat = reconstructed.float().flatten()

        mse = torch.mean((orig_flat - recon_flat) ** 2).item()

        orig_norm = torch.norm(orig_flat)
        if orig_norm > 0:
            relative_error = torch.norm(orig_flat - recon_flat).item() / orig_norm.item()
        else:
            relative_error = 0.0

        if orig_norm > 0 and torch.norm(recon_flat) > 0:
            cosine_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
            ).item()
        else:
            cosine_sim = 0.0

    return {
        "mse": mse,
        "cosine_sim": cosine_sim,
        "relative_error": relative_error,
    }
