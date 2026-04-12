"""
Knowledge Condensation — Compress knowledge, not weights.

Run the model on diverse inputs, collect input->output mappings, and
build a two-tier lookup: a hash-based cache for common patterns (instant)
and a tiny fallback network for rare queries. Like a neural cache —
frequent queries hit the table, rare ones hit a small learned function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class CondensedKnowledge:
    """Two-tier knowledge store: hash cache + fallback net."""
    cache_keys: torch.Tensor     # (n_cached, dim)
    cache_values: torch.Tensor   # (n_cached, out_dim)
    fallback: nn.Module          # tiny network for cache misses
    threshold: float = 0.95      # cosine similarity threshold for cache hit
    stats: dict = field(default_factory=lambda: {"hits": 0, "misses": 0})


class KnowledgeCondenser:
    """Build a two-tier knowledge cache from model behavior."""

    @staticmethod
    def build_cache(model_fn: Callable, sample_inputs: torch.Tensor,
                    cache_size: int = 1024, hidden: int = 64) -> CondensedKnowledge:
        """Collect model outputs, keep the most representative as cache entries."""
        with torch.no_grad():
            outputs = model_fn(sample_inputs)  # (n_samples, out_dim)
        inputs, outputs = sample_inputs.float(), outputs.float()
        # Select cache entries via farthest-point sampling for diversity
        n = inputs.shape[0]
        cache_size = min(cache_size, n)
        selected = [torch.randint(n, (1,)).item()]
        for _ in range(cache_size - 1):
            dists = torch.cdist(inputs[selected], inputs).min(dim=0).values
            selected.append(dists.argmax().item())
        cache_keys = inputs[selected]
        cache_values = outputs[selected]
        # Train tiny fallback network on residuals (what the cache misses)
        in_dim, out_dim = inputs.shape[-1], outputs.shape[-1]
        fallback = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        optim = torch.optim.Adam(fallback.parameters(), lr=1e-3)
        for _ in range(200):
            pred = fallback(inputs)
            loss = F.mse_loss(pred, outputs)
            optim.zero_grad(); loss.backward(); optim.step()
        return CondensedKnowledge(cache_keys=cache_keys, cache_values=cache_values,
                                  fallback=fallback)

    @staticmethod
    def query(knowledge: CondensedKnowledge, x: torch.Tensor) -> torch.Tensor:
        """Two-tier lookup: cache hit -> instant, miss -> fallback net."""
        x_flat = x.float().view(-1, x.shape[-1])
        sims = F.cosine_similarity(x_flat.unsqueeze(1),
                                   knowledge.cache_keys.unsqueeze(0), dim=-1)
        best_sim, best_idx = sims.max(dim=1)
        cache_out = knowledge.cache_values[best_idx]
        net_out = knowledge.fallback(x_flat)
        hit = (best_sim >= knowledge.threshold).unsqueeze(-1)
        knowledge.stats["hits"] += hit.sum().item()
        knowledge.stats["misses"] += (~hit).sum().item()
        return torch.where(hit, cache_out, net_out).view(*x.shape[:-1], -1)
