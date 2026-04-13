"""
10D HYPERCUBE INDEXING — High-dimensional codebook (#17).

Instead of storing weights directly, index them in a 10D hypercube.
Each weight = a point in 10D space. Nearby points = similar weights.

The hypercube has 2^10 = 1024 vertices. Each vertex stores a
"prototype" weight value. Any weight is interpolated from its
nearest vertices. Store only the 1024 prototypes + the index.

For a 1M-weight matrix: 1M indices (10 bits each) + 1024 prototypes
= 1.25MB + 4KB = massive compression from 4MB.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypercubeCodebook(nn.Module):
    """Quantize weights using a 10D hypercube codebook."""

    def __init__(self, n_dims=10, n_centroids_per_dim=2):
        super().__init__()
        self.n_dims = n_dims
        self.n_centroids = n_centroids_per_dim ** n_dims  # 2^10 = 1024

        # The codebook: 1024 prototype values
        self.codebook = nn.Parameter(torch.randn(self.n_centroids) * 0.01)

        # Learned projection: map weight position to 10D hypercube coordinate
        self.projector = nn.Linear(3, n_dims, bias=False)  # (layer, row, col) -> 10D

    def encode(self, weight_matrix, layer_frac=0.5):
        """Encode a weight matrix into hypercube indices."""
        R, C = weight_matrix.shape
        # Create position grid
        rows = torch.linspace(0, 1, R, device=weight_matrix.device)
        cols = torch.linspace(0, 1, C, device=weight_matrix.device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        layer_grid = torch.full_like(grid_r, layer_frac)
        positions = torch.stack([layer_grid, grid_r, grid_c], dim=-1)  # (R, C, 3)

        # Project to 10D
        coords_10d = torch.sigmoid(self.projector(positions))  # (R, C, 10) in [0,1]

        # Convert to binary index (nearest vertex)
        binary = (coords_10d > 0.5).long()
        indices = torch.zeros(R, C, dtype=torch.long, device=weight_matrix.device)
        for d in range(self.n_dims):
            indices = indices + binary[..., d] * (2 ** d)

        return indices

    def decode(self, indices):
        """Decode hypercube indices back to weight values."""
        return self.codebook[indices]

    def forward(self, weight_matrix, layer_frac=0.5):
        """Encode then decode (for training with straight-through estimator)."""
        indices = self.encode(weight_matrix, layer_frac)
        reconstructed = self.decode(indices)
        # Straight-through: gradients flow through as if no quantization
        return weight_matrix + (reconstructed - weight_matrix).detach()
