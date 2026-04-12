"""
Holographic Boundary Encoding — AdS/CFT-Inspired Weight Compression

In theoretical physics, the holographic principle says a d-dimensional
volume's information can be encoded on its (d-1)-dimensional boundary.
Applied to weight matrices: store only the boundary (first/last rows +
first/last cols) and learn a small network that reconstructs the interior.

If weights have smooth spatial structure (and they do — neighboring elements
correlate), the boundary constrains the interior strongly. The reconstruction
network learns the "bulk physics" — how interior values follow from boundaries.

Storage: 2*(m + n - 2) boundary values + tiny reconstruction network,
instead of m*n interior values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BoundaryData:
    """Boundary-encoded weight representation."""
    boundary: torch.Tensor     # flattened boundary values
    shape: tuple               # original (rows, cols)
    net_state: dict            # reconstruction network weights

    def storage_bytes(self) -> int:
        b = self.boundary.numel() * 2  # FP16
        n = sum(p.numel() * 2 for p in self.net_state.values())
        return b + n

    @property
    def bits_per_weight(self) -> float:
        total = self.shape[0] * self.shape[1]
        return (self.storage_bytes() * 8) / total


class BoundaryEncoder:
    """Encode a weight matrix as its boundary + learned interior reconstruction."""

    def __init__(self, hidden: int = 64, n_epochs: int = 200, lr: float = 3e-3):
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.lr = lr

    @staticmethod
    def _extract_boundary(w: torch.Tensor):
        top = w[0, :]
        bottom = w[-1, :]
        left = w[1:-1, 0]
        right = w[1:-1, -1]
        return torch.cat([top, bottom, left, right])

    def _build_net(self, boundary_dim: int, m: int, n: int):
        interior = (m - 2) * (n - 2)
        return nn.Sequential(
            nn.Linear(boundary_dim + 2, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, 1),
        )

    def encode(self, weight: torch.Tensor) -> BoundaryData:
        w = weight.float()
        m, n = w.shape
        boundary = self._extract_boundary(w)
        interior = w[1:-1, 1:-1].reshape(-1)

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, m - 2), torch.linspace(0, 1, n - 2), indexing='ij'
        ), dim=-1).reshape(-1, 2)

        net = self._build_net(len(boundary), m, n)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        b_expand = boundary.unsqueeze(0).expand(coords.shape[0], -1)
        inp = torch.cat([b_expand, coords], dim=1)

        for _ in range(self.n_epochs):
            pred = net(inp).squeeze(-1)
            loss = F.mse_loss(pred, interior)
            opt.zero_grad(); loss.backward(); opt.step()

        return BoundaryData(
            boundary=boundary.half(), shape=(m, n),
            net_state={k: v.half() for k, v in net.state_dict().items()},
        )

    def decode(self, bd: BoundaryData) -> torch.Tensor:
        m, n = bd.shape
        boundary = bd.boundary.float()
        net = self._build_net(len(boundary), m, n)
        net.load_state_dict({k: v.float() for k, v in bd.net_state.items()})

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, m - 2), torch.linspace(0, 1, n - 2), indexing='ij'
        ), dim=-1).reshape(-1, 2)

        b_expand = boundary.unsqueeze(0).expand(coords.shape[0], -1)
        inp = torch.cat([b_expand, coords], dim=1)

        with torch.no_grad():
            interior = net(inp).squeeze(-1).reshape(m - 2, n - 2)

        w = torch.zeros(m, n)
        w[0, :] = boundary[:n]
        w[-1, :] = boundary[n:2 * n]
        w[1:-1, 0] = boundary[2 * n:2 * n + m - 2]
        w[1:-1, -1] = boundary[2 * n + m - 2:]
        w[1:-1, 1:-1] = interior
        return w
