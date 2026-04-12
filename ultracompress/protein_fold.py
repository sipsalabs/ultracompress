"""Protein Fold Compression — Bio-Inspired Weight Encoding

Proteins fold from 1D amino acid sequences into complex 3D structures.
The "compression" is the folding RULES, not the final structure.

Store a short 1D sequence + a tiny MLP that folds it into the full weight matrix.
"""

import torch
import torch.nn as nn


class WeightFolder(nn.Module):
    """Folds a 1D sequence into a 2D weight matrix via learned rules."""

    def __init__(self, rows, cols, seq_len, hidden=32):
        super().__init__()
        self.rows, self.cols, self.seq_len = rows, cols, seq_len
        self.sequence = nn.Parameter(torch.randn(seq_len) * 0.01)
        self.fold_rules = nn.Sequential(
            nn.Linear(seq_len + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def unfold(self):
        r = torch.linspace(-1, 1, self.rows, device=self.sequence.device)
        c = torch.linspace(-1, 1, self.cols, device=self.sequence.device)
        grid = torch.stack(torch.meshgrid(r, c, indexing="ij"), dim=-1)  # (R,C,2)
        seq = self.sequence.expand(self.rows, self.cols, -1)             # (R,C,S)
        inp = torch.cat([seq, grid], dim=-1)                             # (R,C,S+2)
        return self.fold_rules(inp).squeeze(-1)                          # (R,C)

    def compression_ratio(self):
        orig = self.rows * self.cols
        compressed = self.seq_len + sum(p.numel() for p in self.fold_rules.parameters())
        return orig / compressed

    def param_count(self):
        return self.seq_len + sum(p.numel() for p in self.fold_rules.parameters())


class SequenceCompressor:
    """Compress an existing weight matrix into sequence + folding rules."""

    @staticmethod
    def compress(weight, seq_len=64, hidden=32, steps=2000, lr=1e-3):
        rows, cols = weight.shape
        folder = WeightFolder(rows, cols, seq_len, hidden).to(weight.device)
        opt = torch.optim.Adam(folder.parameters(), lr=lr)
        target = weight.detach()
        for i in range(steps):
            loss = (folder.unfold() - target).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        ratio = folder.compression_ratio()
        mse = (folder.unfold() - target).pow(2).mean().item()
        return folder, {"compression_ratio": ratio, "mse": mse}
