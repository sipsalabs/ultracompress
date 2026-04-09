"""
Generative Weight Compression (GWC) — The Paradigm Shift

Instead of STORING compressed weights, we train a small neural network
(the "generator") that PRODUCES weight values on demand.

Input:  (layer_id, tensor_type, row, col)  — coordinates in weight space
Output: weight value at that position

This is an Implicit Neural Representation (INR) of the weight space.
Like how NeRF represents a 3D scene as a function, we represent an
entire LLM as a function from coordinates to values.

Why this scales to 1000T:
  - A 10B parameter generator (~20GB) has enough capacity to learn
    the patterns in weight matrices
  - At 1000T params, redundancy is ASTRONOMICAL — the weights are
    not random, they're structured outputs of gradient descent
  - The generator discovers and exploits weight-space structure that
    PQ/VQ can never see (cross-layer patterns, weight correlations,
    algebraic structure)

Compression ratio:
  Generator size / Model size = 20GB / (1000T * 2 bytes) = 20 / 2,000,000
  That's 100,000x compression. 0.00016 BPW.

Key innovations over standard INR:
  1. SIREN activation (sin) for high-frequency weight patterns
  2. Modulated layers — layer_id modulates the generator, so it learns
     shared structure + per-layer variations
  3. Progressive training — coarse to fine, like multi-resolution hashing
  4. Fourier position encoding — high-frequency coordinate features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class GWCResult:
    """Result of generative weight compression."""
    generator: nn.Module
    generator_params: int
    generator_bytes: int
    original_params: int
    original_bytes: int
    compression_ratio: float
    bpw: float
    per_tensor_cosine: Dict[str, float]
    avg_cosine: float
    training_loss: float


def fourier_features(coords: torch.Tensor, n_freqs: int = 32) -> torch.Tensor:
    """Fourier position encoding — lets the network represent high frequencies.

    Without this, MLPs are biased toward smooth functions and can't capture
    the sharp, high-frequency patterns in weight matrices.

    Maps each coordinate x to: [sin(2^0 * pi * x), cos(2^0 * pi * x),
                                  sin(2^1 * pi * x), cos(2^1 * pi * x), ...]
    """
    freq_bands = torch.linspace(0, n_freqs - 1, n_freqs, device=coords.device)
    freq_bands = (2.0 ** freq_bands) * math.pi

    # coords: (..., D) -> (..., D * 2 * n_freqs)
    proj = coords.unsqueeze(-1) * freq_bands  # (..., D, n_freqs)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1).reshape(
        *coords.shape[:-1], -1
    )


class SirenLayer(nn.Module):
    """Sinusoidal activation layer (SIREN) — better than ReLU for INR.

    sin activation lets the network represent arbitrary frequency content.
    Key insight from SIREN paper: careful initialization is critical.
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class WeightGenerator(nn.Module):
    """Neural network that generates weight values from coordinates.

    Architecture:
      Input: fourier_features(layer_id, tensor_type, row_norm, col_norm)
      → SIREN layers with skip connections
      → Modulation by layer embedding (each layer has a learned style vector)
      → Output: predicted weight value

    The modulation is key: it lets the network learn SHARED structure
    (things all layers have in common) + PER-LAYER variations (each layer's
    unique contribution). This is far more parameter-efficient than
    having independent representations per layer.
    """
    def __init__(
        self,
        n_layers: int = 36,
        n_tensor_types: int = 8,
        hidden_dim: int = 256,
        n_hidden: int = 4,
        n_fourier_freqs: int = 32,
        modulation_dim: int = 64,
    ):
        super().__init__()
        self.n_fourier_freqs = n_fourier_freqs

        # Coordinate input: 4 coords (layer, type, row, col) * 2 * n_freqs
        coord_dim = 4 * 2 * n_fourier_freqs

        # Layer and tensor type embeddings (modulation)
        self.layer_embed = nn.Embedding(n_layers, modulation_dim)
        self.type_embed = nn.Embedding(n_tensor_types, modulation_dim)

        # SIREN backbone
        self.first_layer = SirenLayer(coord_dim + modulation_dim * 2, hidden_dim, is_first=True)

        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden - 1):
            self.hidden_layers.append(SirenLayer(hidden_dim, hidden_dim))

        # Output: single weight value
        self.output_layer = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            # Small init for output — start near zero
            self.output_layer.weight.uniform_(-1e-3, 1e-3)
            self.output_layer.bias.zero_()

        # Per-tensor scale and shift (learned during training)
        # This handles the fact that different tensors have very different scales
        self.tensor_scales = nn.ParameterDict()
        self.tensor_shifts = nn.ParameterDict()

    def register_tensor(self, name: str, scale: float, shift: float):
        """Register a tensor's scale/shift for denormalization."""
        safe_name = name.replace(".", "_")
        self.tensor_scales[safe_name] = nn.Parameter(torch.tensor(scale))
        self.tensor_shifts[safe_name] = nn.Parameter(torch.tensor(shift))

    def forward(self, coords: torch.Tensor, layer_ids: torch.Tensor,
                type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, 4) — [layer_norm, type_norm, row_norm, col_norm]
            layer_ids: (batch,) — integer layer indices
            type_ids: (batch,) — integer tensor type indices

        Returns:
            (batch,) — predicted weight values (normalized)
        """
        # Fourier encode coordinates
        ff = fourier_features(coords, self.n_fourier_freqs)

        # Get modulation vectors
        layer_mod = self.layer_embed(layer_ids)
        type_mod = self.type_embed(type_ids)

        # Concatenate everything
        x = torch.cat([ff, layer_mod, type_mod], dim=-1)

        # SIREN forward
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x) + x  # Skip connection

        return self.output_layer(x).squeeze(-1)

    def generate_weight(self, name: str, layer_id: int, type_id: int,
                        rows: int, cols: int, device: str = "cuda",
                        chunk_size: int = 65536) -> torch.Tensor:
        """Generate a full weight matrix.

        Creates the coordinate grid for a (rows, cols) matrix and
        runs the generator to produce all weight values.
        """
        safe_name = name.replace(".", "_")
        total = rows * cols
        weight = torch.zeros(total, device=device)

        # Generate in chunks to control memory
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            n = end - start

            # Build coordinates: normalized row and col positions
            indices = torch.arange(start, end, device=device)
            row_idx = (indices // cols).float() / max(rows - 1, 1)
            col_idx = (indices % cols).float() / max(cols - 1, 1)
            layer_norm = torch.full((n,), layer_id / 36.0, device=device)
            type_norm = torch.full((n,), type_id / 8.0, device=device)

            coords = torch.stack([layer_norm, type_norm, row_idx, col_idx], dim=-1)
            layer_ids = torch.full((n,), layer_id, device=device, dtype=torch.long)
            type_ids = torch.full((n,), type_id, device=device, dtype=torch.long)

            with torch.no_grad():
                vals = self.forward(coords, layer_ids, type_ids)
                weight[start:end] = vals

        # Denormalize
        if safe_name in self.tensor_scales:
            scale = self.tensor_scales[safe_name]
            shift = self.tensor_shifts[safe_name]
            weight = weight * scale + shift

        return weight.reshape(rows, cols)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def storage_bytes(self) -> int:
        # FP16 storage
        return self.count_parameters() * 2


# Tensor type mapping for the generator
TENSOR_TYPE_MAP = {
    "attn_q": 0,
    "attn_k": 1,
    "attn_v": 2,
    "attn_output": 3,
    "ffn_gate": 4,
    "ffn_up": 5,
    "ffn_down": 6,
    "embed": 7,
}


def get_tensor_type(name: str) -> int:
    """Map a GGUF tensor name to a type ID."""
    for key, tid in TENSOR_TYPE_MAP.items():
        if key in name:
            return tid
    return 0


def get_layer_id(name: str) -> int:
    """Extract layer index from GGUF tensor name."""
    if "blk." in name:
        try:
            return int(name.split("blk.")[1].split(".")[0])
        except (ValueError, IndexError):
            pass
    return 0


def train_generator(
    weights: List[Tuple[str, torch.Tensor]],
    hidden_dim: int = 256,
    n_hidden: int = 4,
    n_fourier_freqs: int = 32,
    n_epochs: int = 50,
    batch_size: int = 65536,
    lr: float = 1e-4,
    device: str = "cuda",
    max_samples_per_tensor: int = 100000,
    progress_callback=None,
) -> GWCResult:
    """Train a weight generator on a set of weight tensors.

    This is the main entry point. Given a list of (name, tensor) pairs,
    trains a neural network to predict weight values from coordinates.

    The training process:
      1. Normalize each weight tensor (zero mean, unit variance)
      2. Sample random (coordinate, value) pairs
      3. Train the generator to minimize MSE on these pairs
      4. The generator learns cross-layer patterns automatically

    Args:
        weights: List of (name, tensor) pairs from the model
        hidden_dim: Generator hidden layer width
        n_hidden: Number of hidden layers
        n_fourier_freqs: Fourier feature frequencies
        n_epochs: Training epochs
        batch_size: Samples per batch
        lr: Learning rate
        device: cuda or cpu
        max_samples_per_tensor: Cap samples per tensor to balance training
    """
    # Filter to 2D weight matrices
    weight_list = [(n, w) for n, w in weights if w.ndim >= 2 and w.numel() >= 256]

    if not weight_list:
        raise ValueError("No valid weight tensors found")

    # Find architecture params
    n_layers = max(get_layer_id(n) for n, _ in weight_list) + 1
    n_types = len(TENSOR_TYPE_MAP)

    # Build generator
    gen = WeightGenerator(
        n_layers=max(n_layers, 1),
        n_tensor_types=n_types,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_fourier_freqs=n_fourier_freqs,
    ).to(device)

    # Prepare training data: sample coordinates and values from all tensors
    all_coords = []
    all_layer_ids = []
    all_type_ids = []
    all_values = []
    tensor_info = {}  # name -> (rows, cols, scale, shift)

    total_original_params = 0
    total_original_bytes = 0

    for name, tensor in weight_list:
        w = tensor.float()
        if w.ndim > 2:
            w = w.reshape(w.shape[0], -1)

        rows, cols = w.shape
        total_original_params += w.numel()
        total_original_bytes += w.numel() * 2  # FP16

        # Normalize
        shift = w.mean().item()
        scale = w.std().item()
        if scale < 1e-10:
            scale = 1.0
        w_norm = (w - shift) / scale

        # Register scale/shift
        gen.register_tensor(name, scale, shift)
        tensor_info[name] = (rows, cols, scale, shift)

        layer_id = get_layer_id(name)
        type_id = get_tensor_type(name)

        # Sample random positions
        n_total = rows * cols
        n_samples = min(n_total, max_samples_per_tensor)
        sample_idx = torch.randperm(n_total)[:n_samples]

        row_idx = (sample_idx // cols).float() / max(rows - 1, 1)
        col_idx = (sample_idx % cols).float() / max(cols - 1, 1)
        layer_norm = torch.full((n_samples,), layer_id / max(n_layers - 1, 1))
        type_norm = torch.full((n_samples,), type_id / max(n_types - 1, 1))

        coords = torch.stack([layer_norm, type_norm, row_idx, col_idx], dim=-1)
        values = w_norm.reshape(-1)[sample_idx]

        all_coords.append(coords)
        all_layer_ids.append(torch.full((n_samples,), layer_id, dtype=torch.long))
        all_type_ids.append(torch.full((n_samples,), type_id, dtype=torch.long))
        all_values.append(values)

    # Concatenate all training data
    all_coords = torch.cat(all_coords, dim=0)
    all_layer_ids = torch.cat(all_layer_ids, dim=0)
    all_type_ids = torch.cat(all_type_ids, dim=0)
    all_values = torch.cat(all_values, dim=0)

    n_total_samples = all_coords.shape[0]
    print(f"  Training data: {n_total_samples:,} samples from {len(weight_list)} tensors")
    print(f"  Generator: {gen.count_parameters():,} params ({gen.storage_bytes()/1e6:.1f} MB)")
    print(f"  Original model: {total_original_params:,} params ({total_original_bytes/1e9:.2f} GB)")
    print(f"  Target ratio: {total_original_bytes / max(gen.storage_bytes(), 1):.0f}x")

    # Move generator params to device (already done), move to device for training
    gen = gen.to(device)
    optimizer = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    gen.train()
    best_loss = float('inf')

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_total_samples)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_total_samples, batch_size):
            end = min(start + batch_size, n_total_samples)
            idx = perm[start:end]

            coords_b = all_coords[idx].to(device)
            layer_b = all_layer_ids[idx].to(device)
            type_b = all_type_ids[idx].to(device)
            values_b = all_values[idx].to(device)

            pred = gen(coords_b, layer_b, type_b)
            loss = F.mse_loss(pred, values_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if progress_callback:
            progress_callback(epoch, n_epochs, avg_loss)
        elif epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Evaluate: reconstruct each tensor and measure cosine similarity
    gen.eval()
    per_tensor_cosine = {}
    cosines = []

    with torch.no_grad():
        for name, tensor in weight_list:
            w = tensor.float()
            if w.ndim > 2:
                w = w.reshape(w.shape[0], -1)
            rows, cols = w.shape
            layer_id = get_layer_id(name)
            type_id = get_tensor_type(name)

            recon = gen.generate_weight(name, layer_id, type_id, rows, cols, device)
            recon = recon.to(w.device)

            cos = F.cosine_similarity(w.reshape(1, -1), recon.reshape(1, -1)).item()
            per_tensor_cosine[name] = cos
            cosines.append(cos)

    avg_cosine = np.mean(cosines) if cosines else 0
    gen_bytes = gen.storage_bytes()
    bpw = (gen_bytes * 8) / total_original_params if total_original_params > 0 else 0

    return GWCResult(
        generator=gen,
        generator_params=gen.count_parameters(),
        generator_bytes=gen_bytes,
        original_params=total_original_params,
        original_bytes=total_original_bytes,
        compression_ratio=total_original_bytes / max(gen_bytes, 1),
        bpw=bpw,
        per_tensor_cosine=per_tensor_cosine,
        avg_cosine=avg_cosine,
        training_loss=best_loss,
    )
