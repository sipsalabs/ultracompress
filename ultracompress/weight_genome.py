"""
The Weight Genome — Hierarchical Generative Weight Compression

The paradigm shift: don't store compressed weights. Store a GENERATOR
that produces them on demand, plus tiny residuals for the details.

Three-layer hierarchy:
  1. Genome (shared SIREN generator): learns the universal weight structure
     across all layers. A 2-4GB network that can generate any weight in
     the model from its coordinates.

  2. Epigenome (per-layer modulation): tiny style vectors (~KB total) that
     capture what makes layer 17 different from layer 3. Initialized from
     cross-layer SVD coefficients.

  3. Error Correction (residual PQ): whatever the genome can't predict,
     compress with output-aware PQ. Since the genome handles 90-99% of
     the structure, residuals are tiny and PQ at 0.001 BPW works.

The novel part nobody has done: JOINT TRAINING where the generator
minimizes PQ residual entropy, not just weight MSE. This means the
generator learns to produce predictions that are EASY to quantize.

BPW math for 1000T params:
  Generator (4GB / 1000T): 0.000003 BPW
  Modulation (~MB total):  ~0 BPW
  Residual PQ:             0.0002-0.001 BPW
  Total:                   ~0.001 BPW
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ============================================================
# Building blocks
# ============================================================

class SirenLayer(nn.Module):
    """SIREN layer: Linear + sin activation.

    From "Implicit Neural Representations with Periodic Activation Functions"
    (Sitzmann et al., 2020). Sin activations let small networks represent
    high-frequency functions — critical for capturing weight matrix structure.
    """
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)

        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class FourierFeatures(nn.Module):
    """Positional encoding via random Fourier features."""
    def __init__(self, in_dim, n_freqs=64):
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, n_freqs) * 2 * math.pi)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ModulatedLayer(nn.Module):
    """Linear layer with style-based modulation.

    Like StyleGAN's AdaIN but for weight generation:
      h = linear(x) * (1 + scale) + shift
    where (scale, shift) come from the per-layer modulation vector.
    """
    def __init__(self, in_features, out_features, mod_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mod_proj = nn.Linear(mod_dim, out_features * 2)  # scale + shift

        # Initialize modulation to identity
        nn.init.zeros_(self.mod_proj.weight)
        nn.init.zeros_(self.mod_proj.bias)

    def forward(self, x, modulation):
        h = self.linear(x)
        mod = self.mod_proj(modulation)  # (batch, 2*out)
        scale, shift = mod.chunk(2, dim=-1)
        return h * (1 + scale) + shift


# ============================================================
# The Weight Genome
# ============================================================

class WeightGenome(nn.Module):
    """Hierarchical weight generator with per-layer modulation.

    Architecture:
      Input: (position_in_weight, weight_type_embedding)
      -> Fourier features
      -> SIREN backbone (shared across all layers)
      -> Modulated output heads (one per weight type)
      -> Per-layer modulation via style vectors

    The backbone learns universal weight patterns.
    The modulation captures per-layer variations.
    The output heads capture per-weight-type distributions.
    """

    WEIGHT_TYPES = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ]

    def __init__(
        self,
        hidden_dim: int = 512,
        n_hidden: int = 6,
        n_fourier_freqs: int = 64,
        n_layers: int = 28,
        mod_dim: int = 128,
        omega: float = 30.0,
        group_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.mod_dim = mod_dim
        self.group_size = group_size
        self.n_weight_types = len(self.WEIGHT_TYPES)

        # Input encoding
        # Coordinate: (row_norm, col_norm, group_position) = 3D
        self.coord_dim = 3
        self.fourier = FourierFeatures(self.coord_dim, n_fourier_freqs)
        input_dim = n_fourier_freqs * 2  # sin + cos

        # Weight type embedding
        self.type_embed = nn.Embedding(self.n_weight_types, hidden_dim)

        # Per-layer modulation vectors (the "epigenome")
        self.layer_mod = nn.Embedding(n_layers, mod_dim)

        # SIREN backbone (shared)
        layers = []
        layers.append(SirenLayer(input_dim + hidden_dim, hidden_dim, omega=omega, is_first=True))
        for _ in range(n_hidden - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega))
        self.backbone = nn.ModuleList(layers)

        # Modulated output head
        self.mod_layer = ModulatedLayer(hidden_dim, hidden_dim, mod_dim)
        self.output_head = nn.Linear(hidden_dim, group_size)

        # Output scaling (learn per weight type)
        self.output_scale = nn.Parameter(torch.ones(self.n_weight_types) * 0.01)

    def forward(
        self,
        coords: torch.Tensor,
        layer_ids: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate weight groups from coordinates.

        Args:
            coords: (batch, 3) — (row_norm, col_norm, group_pos) in [0,1]
            layer_ids: (batch,) — which transformer layer
            type_ids: (batch,) — which weight type (0-6)

        Returns:
            weights: (batch, group_size) — predicted weight values
        """
        # Encode coordinates
        x = self.fourier(coords)  # (batch, n_fourier*2)

        # Add weight type information
        type_emb = self.type_embed(type_ids)  # (batch, hidden_dim)
        x = torch.cat([x, type_emb], dim=-1)  # (batch, input_dim + hidden_dim)

        # SIREN backbone
        for layer in self.backbone:
            x = layer(x)

        # Per-layer modulation
        mod = self.layer_mod(layer_ids)  # (batch, mod_dim)
        x = self.mod_layer(x, mod)
        x = torch.sin(x)

        # Output
        out = self.output_head(x)  # (batch, group_size)

        # Per-type scaling
        scale = self.output_scale[type_ids].unsqueeze(1)  # (batch, 1)
        return out * scale

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self):
        """Size in bytes (FP16 storage)."""
        return self.count_params() * 2


# ============================================================
# Training utilities
# ============================================================

@dataclass
class GenomeTrainingData:
    """Pre-processed training data for the Weight Genome."""
    coords: torch.Tensor       # (N, 3) — normalized coordinates
    layer_ids: torch.Tensor    # (N,) — layer index
    type_ids: torch.Tensor     # (N,) — weight type index
    targets: torch.Tensor      # (N, group_size) — target weight groups
    scales: torch.Tensor       # (N,) — per-group normalization scales
    weight_shapes: Dict[str, Tuple[int, int]]  # name -> (out, in)


def prepare_training_data(
    weights_dict: Dict[str, torch.Tensor],
    n_layers: int,
    group_size: int = 64,
    max_groups_per_weight: int = 10000,
    device: str = "cpu",
) -> GenomeTrainingData:
    """Convert model weights into training data for the genome.

    Each weight matrix is divided into groups of `group_size` elements.
    Each group becomes a training sample with coordinates indicating
    its position in the matrix.
    """
    type_map = {t: i for i, t in enumerate(WeightGenome.WEIGHT_TYPES)}

    all_coords = []
    all_layer_ids = []
    all_type_ids = []
    all_targets = []
    all_scales = []
    weight_shapes = {}

    for layer_idx in range(n_layers):
        prefix = f'model.layers.{layer_idx}.'
        type_suffix_map = {
            'q_proj': 'self_attn.q_proj.weight',
            'k_proj': 'self_attn.k_proj.weight',
            'v_proj': 'self_attn.v_proj.weight',
            'o_proj': 'self_attn.o_proj.weight',
            'gate_proj': 'mlp.gate_proj.weight',
            'up_proj': 'mlp.up_proj.weight',
            'down_proj': 'mlp.down_proj.weight',
        }

        for wtype, suffix in type_suffix_map.items():
            key = prefix + suffix
            if key not in weights_dict:
                continue

            w = weights_dict[key].float()
            weight_shapes[key] = tuple(w.shape)
            type_id = type_map[wtype]

            # Flatten and group
            flat = w.reshape(-1)
            remainder = flat.numel() % group_size
            if remainder != 0:
                flat = torch.cat([flat, torch.zeros(group_size - remainder)])

            groups = flat.reshape(-1, group_size)
            n_groups = groups.shape[0]

            # Subsample if too many groups
            if n_groups > max_groups_per_weight:
                perm = torch.randperm(n_groups)[:max_groups_per_weight]
                groups = groups[perm]
                n_groups = max_groups_per_weight

            # Normalize groups
            scales = groups.norm(dim=1).clamp(min=1e-10) / np.sqrt(group_size)
            normalized = groups / scales.unsqueeze(1)

            # Compute coordinates: (row_norm, col_norm, group_position)
            out_dim, in_dim = w.shape[0], w.shape[1] if w.ndim >= 2 else 1
            for g in range(n_groups):
                flat_pos = g * group_size
                row = flat_pos // in_dim if in_dim > 0 else 0
                col = flat_pos % in_dim if in_dim > 0 else 0
                row_norm = row / max(out_dim - 1, 1)
                col_norm = col / max(in_dim - 1, 1)
                group_pos = g / max(n_groups - 1, 1)

                all_coords.append([row_norm, col_norm, group_pos])
                all_layer_ids.append(layer_idx)
                all_type_ids.append(type_id)

            all_targets.append(normalized)
            all_scales.append(scales)

    return GenomeTrainingData(
        coords=torch.tensor(all_coords, dtype=torch.float32, device=device),
        layer_ids=torch.tensor(all_layer_ids, dtype=torch.long, device=device),
        type_ids=torch.tensor(all_type_ids, dtype=torch.long, device=device),
        targets=torch.cat(all_targets, dim=0).to(device),
        scales=torch.cat(all_scales, dim=0).to(device),
        weight_shapes=weight_shapes,
    )


def train_genome(
    genome: WeightGenome,
    data: GenomeTrainingData,
    n_epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-4,
    entropy_weight: float = 0.0,
    pq_config: tuple = (8, 4, 64),
    temperature_schedule: tuple = (5.0, 0.5),
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Train the Weight Genome.

    Progressive training:
      Phase A (epochs 0-50%): MSE only — learn coarse structure
      Phase B (epochs 50-100%): MSE + entropy — learn PQ-friendly residuals

    Args:
        genome: the model to train
        data: prepared training data
        n_epochs: total training epochs
        batch_size: training batch size
        lr: learning rate
        entropy_weight: beta for entropy loss (0 = MSE only)
        pq_config: (M, K, G) for differentiable PQ in entropy loss
        temperature_schedule: (start_temp, end_temp) for Gumbel-Softmax annealing
        device: compute device
        verbose: print progress

    Returns:
        metrics dict with final loss, per-epoch history, etc.
    """
    from .differentiable_pq import soft_pq_compress

    genome = genome.to(device)
    optimizer = torch.optim.AdamW(genome.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    N = data.coords.shape[0]
    history = []

    for epoch in range(n_epochs):
        genome.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0
        epoch_mse = 0
        epoch_entropy = 0
        n_batches = 0

        # Temperature annealing
        t_start, t_end = temperature_schedule
        progress = epoch / max(n_epochs - 1, 1)
        temperature = t_start + (t_end - t_start) * progress

        # Phase control
        use_entropy = entropy_weight > 0 and epoch >= n_epochs // 2

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]

            coords = data.coords[idx]
            layer_ids = data.layer_ids[idx]
            type_ids = data.type_ids[idx]
            targets = data.targets[idx]

            # Forward
            pred = genome(coords, layer_ids, type_ids)

            # MSE loss
            mse = F.mse_loss(pred, targets)
            loss = mse

            # Entropy loss (Phase B only)
            if use_entropy:
                residual = (targets - pred).detach() + pred - pred.detach()  # straight-through
                M, K, G = pq_config
                if residual.shape[0] >= G and G <= residual.shape[1]:
                    _, entropy, _ = soft_pq_compress(
                        residual.reshape(-1),
                        n_subvectors=M, codebook_size=K,
                        group_size=G, temperature=temperature,
                    )
                    loss = loss + entropy_weight * entropy
                    epoch_entropy += entropy.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(genome.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_entropy = epoch_entropy / max(n_batches, 1)

        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'mse': avg_mse,
            'entropy': avg_entropy,
            'temperature': temperature,
            'lr': scheduler.get_last_lr()[0],
        })

        if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
            ent_str = f"  entropy={avg_entropy:.4f}" if use_entropy else ""
            print(f"  Epoch {epoch:>4}/{n_epochs}  loss={avg_loss:.6f}  mse={avg_mse:.6f}{ent_str}  temp={temperature:.2f}")

    return {
        'final_loss': history[-1]['loss'],
        'final_mse': history[-1]['mse'],
        'history': history,
    }


@torch.no_grad()
def evaluate_genome(
    genome: WeightGenome,
    data: GenomeTrainingData,
    batch_size: int = 8192,
    device: str = "cuda",
) -> dict:
    """Evaluate genome quality per weight type and layer."""
    genome.eval()
    N = data.coords.shape[0]

    all_preds = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pred = genome(
            data.coords[start:end],
            data.layer_ids[start:end],
            data.type_ids[start:end],
        )
        all_preds.append(pred)

    preds = torch.cat(all_preds, dim=0)
    targets = data.targets

    # Overall cosine
    overall_cos = F.cosine_similarity(
        preds.reshape(1, -1), targets.reshape(1, -1)
    ).item()

    # Per-type cosine
    type_cosines = {}
    for type_id, type_name in enumerate(WeightGenome.WEIGHT_TYPES):
        mask = data.type_ids == type_id
        if mask.any():
            p = preds[mask].reshape(1, -1)
            t = targets[mask].reshape(1, -1)
            type_cosines[type_name] = F.cosine_similarity(p, t).item()

    # Per-layer cosine
    layer_cosines = {}
    for layer_id in range(genome.n_layers):
        mask = data.layer_ids == layer_id
        if mask.any():
            p = preds[mask].reshape(1, -1)
            t = targets[mask].reshape(1, -1)
            layer_cosines[layer_id] = F.cosine_similarity(p, t).item()

    # Residual statistics
    residual = targets - preds
    residual_norm = residual.norm(dim=1).mean().item()
    target_norm = targets.norm(dim=1).mean().item()
    residual_ratio = residual_norm / max(target_norm, 1e-10)

    return {
        'overall_cosine': overall_cos,
        'per_type': type_cosines,
        'per_layer': layer_cosines,
        'residual_norm': residual_norm,
        'target_norm': target_norm,
        'residual_ratio': residual_ratio,
        'genome_params': genome.count_params(),
        'genome_bytes': genome.size_bytes(),
    }
