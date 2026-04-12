"""Mixture-of-Depths for FRR (Raposo et al. 2024) — easy tokens exit early,
hard tokens get more recursion. Avg depth << max depth = compute savings."""

import torch, torch.nn as nn


class DepthRouter(nn.Module):
    """Per-token decision: "is this token done?" at each FRR iteration."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x).squeeze(-1).sigmoid()


class AdaptiveFRR(nn.Module):
    """FRR that stops early for easy tokens, keeps going for hard ones."""
    def __init__(self, dim: int, max_depth: int = 8):
        super().__init__()
        self.max_depth = max_depth
        self.block = nn.TransformerEncoderLayer(dim, nhead=max(1, dim // 64), batch_first=True)
        self.router = DepthRouter(dim)
        self.depth_embeddings = nn.Embedding(max_depth, dim)

    def forward(self, tokens: torch.Tensor, max_depth: int | None = None,
                confidence_threshold: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, depths_per_token)."""
        md = min(max_depth or self.max_depth, self.max_depth)
        B, S, D = tokens.shape
        done = torch.zeros(B, S, device=tokens.device, dtype=torch.bool)
        depths = torch.zeros(B, S, device=tokens.device)
        x = tokens
        for d in range(md):
            x = x + self.depth_embeddings(torch.tensor(d, device=x.device))
            x_new = self.block(x)
            conf = self.router(x_new)
            newly_done = (~done) & (conf > confidence_threshold)
            depths += (~done).float()
            done = done | newly_done
            x = torch.where(done.unsqueeze(-1), x, x_new)
            if done.all():
                break
        return x, depths
