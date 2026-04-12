"""
PARADIGM SHIFT — Three fundamentally new approaches to model compression.

These are NOT variations on "small transformer approximates big transformer."
These are entirely different vehicles.

Paradigm 1: NERF FOR WEIGHTS (Implicit Neural Representation)
  A single MLP encodes ALL weights of the model as a continuous function.
  Input: (layer, matrix_type, row, col) → Output: weight value
  Inspired by: NeRF, SIREN, implicit neural representations
  Key insight: Weight tensors aren't random — they have smooth, structured
  patterns. A tiny MLP can learn these patterns and reconstruct weights
  at any coordinate on demand.

Paradigm 2: PROCEDURAL WEIGHT GENERATION (HyperNetwork)
  A tiny "meta-network" that takes a layer descriptor and GENERATES
  the full weight matrix. Each layer is described by a small code vector.
  Inspired by: HyperNetworks, procedural generation, DNA→protein
  Key insight: Layers are variations on a theme. A generator network
  learns the "theme" and each layer just stores its small variation code.

Paradigm 3: ALGEBRAIC COMPRESSION (Mathematical Structure)
  Find the mathematical relationships between weight matrices.
  Shared eigenbasis + per-layer coefficients + sparse corrections.
  Inspired by: Spectral methods, group theory, functional analysis,
  Kolmogorov complexity, tensor decomposition
  Key insight: Weight matrices live in a low-dimensional manifold.
  Find the manifold's basis, project onto it, store only coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional


# ================================================================
# PARADIGM 1: NERF FOR WEIGHTS
# ================================================================

class SinusoidalEncoding(nn.Module):
    """Positional encoding from NeRF — maps low-dim coordinates to high-dim
    features using sin/cos at exponentially increasing frequencies.

    This is what lets a tiny MLP represent high-frequency detail.
    Without it, MLPs can only learn smooth functions (spectral bias).
    """
    def __init__(self, n_freqs=10, include_input=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        # Frequencies: 2^0, 2^1, ..., 2^(n_freqs-1)
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """x: (..., D) → (..., D * (2*n_freqs + include_input))"""
        enc = []
        if self.include_input:
            enc.append(x)
        for freq in self.freqs:
            enc.append(torch.sin(freq * math.pi * x))
            enc.append(torch.cos(freq * math.pi * x))
        return torch.cat(enc, dim=-1)

    def output_dim(self, input_dim):
        return input_dim * (2 * self.n_freqs + int(self.include_input))


class WeightNeRF(nn.Module):
    """Implicit Neural Representation of ALL model weights.

    A single MLP that maps (layer, matrix_type, row, col) → weight_value.

    The entire model's weights are encoded in this one network.
    At inference time, you reconstruct whatever weights you need by
    querying the appropriate coordinates.

    Architecture inspired by SIREN (sinusoidal activations for better
    high-frequency learning) + NeRF (positional encoding).
    """
    def __init__(self, hidden_dim=512, n_layers=6, n_freqs=8,
                 use_siren=True, omega_0=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_siren = use_siren
        self.omega_0 = omega_0

        # Positional encoding for input coordinates
        self.pos_enc = SinusoidalEncoding(n_freqs=n_freqs)
        # 4 input dims: (layer, matrix_type, row, col) each normalized to [0, 1]
        input_dim = self.pos_enc.output_dim(4)

        # Build MLP with skip connections (like NeRF)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers - 2):
            if i == n_layers // 2 - 1:
                # Skip connection at midpoint
                layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))  # Output: single weight value
        self.layers = nn.ModuleList(layers)
        self.skip_layer = n_layers // 2  # Which layer gets skip connection

        # SIREN initialization (critical for convergence)
        if use_siren:
            self._siren_init()

    def _siren_init(self):
        """SIREN paper initialization — uniform within specific bounds."""
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer: scale by omega_0
                bound = 1.0 / layer.in_features
                nn.init.uniform_(layer.weight, -bound, bound)
            else:
                bound = math.sqrt(6.0 / layer.in_features) / self.omega_0
                nn.init.uniform_(layer.weight, -bound, bound)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, coords):
        """coords: (..., 4) normalized to [0, 1] → (..., 1) weight values"""
        x = self.pos_enc(coords)
        x_skip = x

        for i, layer in enumerate(self.layers[:-1]):
            if i == self.skip_layer:
                x = torch.cat([x, x_skip], dim=-1)
            x = layer(x)
            if self.use_siren:
                x = torch.sin(self.omega_0 * x)
            else:
                x = F.gelu(x)

        return self.layers[-1](x)  # No activation on output

    def reconstruct_matrix(self, layer_idx, matrix_type_idx, rows, cols,
                           n_layers_total, n_matrix_types):
        """Reconstruct a full weight matrix by querying all (row, col) pairs."""
        # Normalize coordinates to [0, 1]
        layer_norm = layer_idx / max(n_layers_total - 1, 1)
        mtype_norm = matrix_type_idx / max(n_matrix_types - 1, 1)

        row_coords = torch.arange(rows, device=next(self.parameters()).device).float() / max(rows - 1, 1)
        col_coords = torch.arange(cols, device=next(self.parameters()).device).float() / max(cols - 1, 1)

        # Create grid of all (row, col) pairs
        row_grid, col_grid = torch.meshgrid(row_coords, col_coords, indexing='ij')

        # Build full coordinate tensor
        n_points = rows * cols
        coords = torch.stack([
            torch.full((n_points,), layer_norm, device=row_grid.device),
            torch.full((n_points,), mtype_norm, device=row_grid.device),
            row_grid.reshape(-1),
            col_grid.reshape(-1),
        ], dim=-1)

        # Query in chunks to avoid OOM
        chunk_size = 65536
        values = []
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            values.append(self(coords[start:end]).squeeze(-1))

        return torch.cat(values).reshape(rows, cols)


class WeightNeRFCompressor:
    """Train a WeightNeRF to encode all weights of a model."""

    def __init__(self, weight_dict, matrix_info, device='cuda'):
        """
        weight_dict: {key: tensor} — all weight matrices
        matrix_info: {key: (layer_idx, matrix_type_idx, n_layers, n_types)}
        """
        self.weight_dict = weight_dict
        self.matrix_info = matrix_info
        self.device = device

        # Pre-compute all training coordinates and target values
        self.all_coords = []
        self.all_values = []

        for key, tensor in weight_dict.items():
            if key not in matrix_info:
                continue
            layer_idx, mtype_idx, n_layers, n_types = matrix_info[key]
            rows, cols = tensor.shape

            # Normalize
            layer_norm = layer_idx / max(n_layers - 1, 1)
            mtype_norm = mtype_idx / max(n_types - 1, 1)

            row_coords = torch.arange(rows).float() / max(rows - 1, 1)
            col_coords = torch.arange(cols).float() / max(cols - 1, 1)
            row_grid, col_grid = torch.meshgrid(row_coords, col_coords, indexing='ij')

            n_points = rows * cols
            coords = torch.stack([
                torch.full((n_points,), layer_norm),
                torch.full((n_points,), mtype_norm),
                row_grid.reshape(-1),
                col_grid.reshape(-1),
            ], dim=-1)

            self.all_coords.append(coords)
            self.all_values.append(tensor.reshape(-1).float())

        self.all_coords = torch.cat(self.all_coords)
        self.all_values = torch.cat(self.all_values)

        # Normalize target values for stable training
        self.value_mean = self.all_values.mean()
        self.value_std = self.all_values.std()
        self.all_values_norm = (self.all_values - self.value_mean) / (self.value_std + 1e-8)

        print(f"Total weight values to encode: {len(self.all_values):,}")
        print(f"Value range: [{self.all_values.min():.4f}, {self.all_values.max():.4f}]")
        print(f"Value mean={self.value_mean:.4f}, std={self.value_std:.4f}")

    def train(self, nerf, n_steps=20000, batch_size=65536, lr=1e-4):
        """Train the NeRF to reconstruct all weights."""
        nerf = nerf.to(self.device)
        opt = torch.optim.Adam(nerf.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

        n_total = len(self.all_coords)

        for step in range(n_steps):
            idx = torch.randint(0, n_total, (batch_size,))
            coords = self.all_coords[idx].to(self.device)
            targets = self.all_values_norm[idx].to(self.device)

            preds = nerf(coords).squeeze(-1)
            loss = F.mse_loss(preds, targets)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(nerf.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 2000 == 0:
                # Compute actual weight reconstruction error
                with torch.no_grad():
                    sample_idx = torch.randint(0, n_total, (min(100000, n_total),))
                    sample_coords = self.all_coords[sample_idx].to(self.device)
                    sample_targets = self.all_values[sample_idx].to(self.device)
                    sample_preds = nerf(sample_coords).squeeze(-1) * self.value_std + self.value_mean
                    mae = (sample_preds - sample_targets).abs().mean()
                    rel_err = mae / (sample_targets.abs().mean() + 1e-8)
                print(f"  Step {step}: loss={loss.item():.6f} MAE={mae.item():.6f} "
                      f"RelErr={rel_err.item():.4f}")

        return nerf


# ================================================================
# PARADIGM 2: PROCEDURAL WEIGHT GENERATION (HyperNetwork)
# ================================================================

class LayerCodebook(nn.Module):
    """Each layer is described by a small learned code vector.

    Like DNA: a compact description that unfolds into a full organism.
    The codebook stores the "genetic code" for each layer.
    """
    def __init__(self, n_layers, code_dim=64):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(n_layers, code_dim) * 0.02)

    def forward(self, layer_idx):
        return self.codes[layer_idx]


class WeightGenerator(nn.Module):
    """Generates weight matrices from layer codes.

    Takes a layer's "DNA" (code vector) + matrix type identifier
    and produces the full weight matrix via chunked generation.

    Inspired by: HyperNetworks (Ha et al.), but with key differences:
    - Shared generator for ALL matrix types (exploits cross-type structure)
    - Chunked generation (generates weight matrix in blocks, not all at once)
    - Residual from a learned "base weight" per matrix type

    The generator learns the PROCESS of creating weights, not the weights themselves.
    """
    def __init__(self, code_dim=64, hidden_dim=512, chunk_size=64,
                 n_matrix_types=11):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        # Matrix type embedding
        self.type_embed = nn.Embedding(n_matrix_types, code_dim)

        # Row/col position encoding
        self.row_enc = SinusoidalEncoding(n_freqs=6)
        self.col_enc = SinusoidalEncoding(n_freqs=6)
        row_dim = self.row_enc.output_dim(1)
        col_dim = self.col_enc.output_dim(1)

        # Generator network: (layer_code + type_embed + row_pos + col_pos) → chunk of weights
        input_dim = code_dim * 2 + row_dim + col_dim
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size),  # Output: one row-chunk of weights
        )

        # Scale parameter (start small for stability)
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def generate_matrix(self, layer_code, matrix_type_idx, rows, cols):
        """Generate a full weight matrix from a layer code.

        Instead of generating all rows*cols values at once (too expensive),
        we generate row by row, with column position encoding.
        """
        device = layer_code.device
        type_emb = self.type_embed(torch.tensor(matrix_type_idx, device=device))

        # For each row, generate weight values across columns
        all_rows = []
        for row_start in range(0, rows, 1):
            row_norm = torch.tensor([row_start / max(rows - 1, 1)], device=device)
            row_enc = self.row_enc(row_norm)

            col_values = []
            for col_start in range(0, cols, self.chunk_size):
                col_end = min(col_start + self.chunk_size, cols)
                actual_chunk = col_end - col_start

                col_norm = torch.tensor([(col_start + actual_chunk // 2) / max(cols - 1, 1)], device=device)
                col_enc = self.col_enc(col_norm)

                inp = torch.cat([layer_code, type_emb, row_enc, col_enc])
                chunk = self.generator(inp) * self.output_scale
                col_values.append(chunk[:actual_chunk])

            all_rows.append(torch.cat(col_values))

        return torch.stack(all_rows)


class ProceduralCompressor(nn.Module):
    """Full procedural compression system.

    Components:
    1. LayerCodebook: small code vector per layer (the "DNA")
    2. WeightGenerator: shared generator that reads DNA → produces weights
    3. At inference: generate weights on-the-fly from codes

    Total storage: codebook (n_layers * code_dim) + generator params
    """
    def __init__(self, n_layers=28, code_dim=64, hidden_dim=512,
                 chunk_size=64, n_matrix_types=11):
        super().__init__()
        self.codebook = LayerCodebook(n_layers, code_dim)
        self.generator = WeightGenerator(code_dim, hidden_dim, chunk_size, n_matrix_types)
        self.n_layers = n_layers
        self.n_matrix_types = n_matrix_types

    def generate_weight(self, layer_idx, matrix_type_idx, rows, cols):
        code = self.codebook(layer_idx)
        return self.generator.generate_matrix(code, matrix_type_idx, rows, cols)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# PARADIGM 3: ALGEBRAIC COMPRESSION
# ================================================================

class AlgebraicCompressor(nn.Module):
    """Compress weights by finding their algebraic structure.

    Key idea: Weight matrices across layers live on a low-dimensional manifold.
    We find the manifold's basis and store only coordinates + sparse corrections.

    Decomposition: W[layer, type] = sum_k(alpha[layer,type,k] * Basis[type,k]) + Sparse[layer,type]

    Where:
    - Basis[type, k] are shared eigenvectors for each matrix type (learned)
    - alpha[layer, type, k] are per-layer coefficients (tiny)
    - Sparse[layer, type] is a sparse correction (top-p% of residual)

    Inspired by:
    - PCA / SVD (but learned, not fixed)
    - Sparse coding (ISTA/FISTA)
    - Manifold learning
    - Tensor decomposition (Tucker, CP)
    - Kolmogorov-Arnold representation theorem

    The basis captures the "archetype" of each matrix type.
    The coefficients capture how each layer deviates.
    The sparse correction captures the irreducible fine detail.
    """
    def __init__(self, n_layers=28, n_basis=16, n_matrix_types=11,
                 sparse_ratio=0.01):
        super().__init__()
        self.n_layers = n_layers
        self.n_basis = n_basis
        self.n_matrix_types = n_matrix_types
        self.sparse_ratio = sparse_ratio

        # These get initialized per matrix type when we analyze the model
        self.basis_dict = nn.ModuleDict()
        self.coeff_dict = nn.ParameterDict()
        self.sparse_indices = {}  # Non-parameter, stored separately
        self.sparse_values = nn.ParameterDict()

        # Inter-layer transition model: predict next layer's coefficients from previous
        # This exploits the sequential structure of transformers
        self.transition = nn.ModuleDict()

    def fit_matrix_type(self, matrix_type_name, weight_stack, rows, cols):
        """Fit the algebraic decomposition for one matrix type across all layers.

        weight_stack: (n_layers, rows, cols) — the same matrix type from every layer
        """
        n_layers = weight_stack.shape[0]
        device = weight_stack.device

        # Step 1: Flatten and find principal components (the "basis")
        flat = weight_stack.reshape(n_layers, -1)  # (n_layers, rows*cols)

        # SVD to find the basis
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)

        # Keep top n_basis components
        basis = Vh[:self.n_basis]  # (n_basis, rows*cols)
        coeffs = U[:, :self.n_basis] * S[:self.n_basis].unsqueeze(0)  # (n_layers, n_basis)

        # Reconstruction and residual
        reconstructed = coeffs @ basis  # (n_layers, rows*cols)
        residual = flat - reconstructed

        # Step 2: Sparse correction — keep top-p% of residual by magnitude
        n_sparse = max(1, int(residual.numel() * self.sparse_ratio))
        residual_flat = residual.reshape(-1)
        _, top_indices = residual_flat.abs().topk(n_sparse)
        sparse_vals = residual_flat[top_indices]

        # Store
        self.basis_dict[matrix_type_name] = nn.Linear(1, 1)  # placeholder
        # Actually store as parameters
        basis_param = nn.Parameter(basis.reshape(self.n_basis, rows, cols), requires_grad=True)
        self.register_parameter(f'basis_{matrix_type_name}', basis_param)

        coeff_param = nn.Parameter(coeffs, requires_grad=True)
        self.coeff_dict[matrix_type_name] = coeff_param

        self.sparse_indices[matrix_type_name] = top_indices.cpu()
        sparse_param = nn.Parameter(sparse_vals, requires_grad=True)
        self.sparse_values[matrix_type_name] = sparse_param

        # Step 3: Inter-layer transition (predict coeffs[i+1] from coeffs[i])
        transition_net = nn.Sequential(
            nn.Linear(self.n_basis, self.n_basis * 2),
            nn.GELU(),
            nn.Linear(self.n_basis * 2, self.n_basis),
        )
        self.transition[matrix_type_name] = transition_net

        # Report
        total_original = weight_stack.numel()
        total_compressed = self.n_basis * rows * cols + n_layers * self.n_basis + n_sparse
        ratio = total_original / total_compressed

        recon_error = residual.pow(2).mean().sqrt()
        sparse_recon = reconstructed.clone().reshape(-1)
        sparse_recon[top_indices] += sparse_vals
        final_error = (flat.reshape(-1) - sparse_recon).pow(2).mean().sqrt()

        print(f"  {matrix_type_name}: {ratio:.1f}x compression, "
              f"RMSE before sparse={recon_error:.6f}, after={final_error:.6f}")

        return ratio

    def reconstruct_matrix(self, matrix_type_name, layer_idx, rows, cols):
        """Reconstruct a weight matrix from basis + coefficients + sparse."""
        basis = getattr(self, f'basis_{matrix_type_name}')  # (n_basis, rows, cols)
        coeffs = self.coeff_dict[matrix_type_name]  # (n_layers, n_basis)

        # Weighted sum of basis matrices
        layer_coeffs = coeffs[layer_idx]  # (n_basis,)
        matrix = torch.einsum('k,krc->rc', layer_coeffs, basis)

        # Add sparse correction
        if matrix_type_name in self.sparse_values:
            sparse_vals = self.sparse_values[matrix_type_name]
            sparse_idx = self.sparse_indices[matrix_type_name].to(matrix.device)

            flat = matrix.reshape(-1)
            # Need to extract only this layer's sparse values
            n_per_layer = rows * cols
            layer_start = layer_idx * n_per_layer
            layer_end = layer_start + n_per_layer

            mask = (sparse_idx >= layer_start) & (sparse_idx < layer_end)
            local_idx = sparse_idx[mask] - layer_start
            local_vals = sparse_vals[mask]

            flat[local_idx] += local_vals
            matrix = flat.reshape(rows, cols)

        return matrix

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# UNIFIED COMPRESSOR — Uses the best paradigm or combines them
# ================================================================

class UnifiedCompressor:
    """Orchestrates compression using all three paradigms and picks the best.

    Or even better — uses different paradigms for different parts of the model:
    - Algebraic for attention weights (highly structured)
    - NeRF for FFN weights (smooth but complex)
    - Procedural for norms (tiny, pattern-based)
    """

    MATRIX_TYPES = {
        'attn_q': 0, 'attn_k': 1, 'attn_v': 2, 'attn_output': 3,
        'ffn_gate': 4, 'ffn_up': 5, 'ffn_down': 6,
        'attn_norm': 7, 'ffn_norm': 8, 'attn_q_norm': 9, 'attn_k_norm': 10,
    }

    def __init__(self, device='cuda'):
        self.device = device

    @staticmethod
    def extract_matrices(gd, n_layers=28):
        """Extract weight matrices organized by type from a weight dict."""
        matrix_stacks = {}
        matrix_shapes = {}

        type_map = {
            'attn_q.weight': 'attn_q', 'attn_k.weight': 'attn_k',
            'attn_v.weight': 'attn_v', 'attn_output.weight': 'attn_output',
            'ffn_gate.weight': 'ffn_gate', 'ffn_up.weight': 'ffn_up',
            'ffn_down.weight': 'ffn_down',
            'attn_norm.weight': 'attn_norm', 'ffn_norm.weight': 'ffn_norm',
            'attn_q_norm.weight': 'attn_q_norm', 'attn_k_norm.weight': 'attn_k_norm',
        }

        for li in range(n_layers):
            for suffix, mtype in type_map.items():
                key = f'blk.{li}.{suffix}'
                if key in gd:
                    w = gd[key].float()
                    if w.ndim == 1:
                        w = w.unsqueeze(0)  # Make 1D weights into (1, D)
                    if mtype not in matrix_stacks:
                        matrix_stacks[mtype] = []
                        matrix_shapes[mtype] = w.shape
                    matrix_stacks[mtype].append(w)

        # Stack into (n_layers, rows, cols) tensors
        for mtype in matrix_stacks:
            matrix_stacks[mtype] = torch.stack(matrix_stacks[mtype])

        return matrix_stacks, matrix_shapes

    @staticmethod
    def build_nerf_training_data(gd, n_layers=28):
        """Build coordinate/value pairs for NeRF training."""
        matrix_stacks, matrix_shapes = UnifiedCompressor.extract_matrices(gd, n_layers)

        matrix_info = {}
        n_types = len(matrix_stacks)
        type_to_idx = {t: i for i, t in enumerate(sorted(matrix_stacks.keys()))}

        weight_dict = {}
        info_dict = {}

        for mtype, stack in matrix_stacks.items():
            for li in range(stack.shape[0]):
                key = f'{mtype}_layer{li}'
                weight_dict[key] = stack[li]
                info_dict[key] = (li, type_to_idx[mtype], n_layers, n_types)

        return weight_dict, info_dict
