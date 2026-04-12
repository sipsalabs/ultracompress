"""
Mixture of LoRAs (MoL) adapter for FRR — inspired by Nouriborji et al. 2025.

Instead of static per-layer modulation, use token-conditional LoRA routing.
Each token selects which LoRA expert(s) to use, enabling the shared block
to behave differently for different token types within the same layer.

At inference: merge experts into single adapter (no routing overhead).

This could restore the expressivity lost at 60x compression and push
quality from 63% toward 70%+ T10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoLAdapter(nn.Module):
    """Mixture of LoRA experts with top-k routing.

    For each token, routes to top-k experts and combines their outputs.
    At merge time, can be collapsed into a single LoRA for inference.
    """
    def __init__(self, hidden_dim, n_experts=4, rank=8, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.rank = rank
        self.top_k = top_k

        # Router: token -> expert weights
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)

        # LoRA experts: each has down (hidden -> rank) and up (rank -> hidden)
        self.experts_down = nn.Parameter(torch.randn(n_experts, hidden_dim, rank) * 0.01)
        self.experts_up = nn.Parameter(torch.zeros(n_experts, rank, hidden_dim))

    def forward(self, x):
        """x: (batch, seq, hidden) -> (batch, seq, hidden)"""
        B, S, D = x.shape

        # Route: (B, S, n_experts)
        router_logits = self.router(x.detach())  # detach to avoid router affecting main grad
        weights, indices = router_logits.topk(self.top_k, dim=-1)  # (B, S, top_k)
        weights = F.softmax(weights, dim=-1)  # normalize

        # Compute expert outputs for selected experts
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, S)
            expert_weight = weights[:, :, k]  # (B, S)

            # Gather expert params for each token
            # This is inefficient but clear — can optimize with einsum later
            for e in range(self.n_experts):
                mask = (expert_idx == e)  # (B, S)
                if mask.any():
                    # Apply LoRA: x @ down @ up
                    expert_out = x @ self.experts_down[e] @ self.experts_up[e]
                    output = output + expert_out * (mask.unsqueeze(-1) * expert_weight.unsqueeze(-1))

        return x + output

    def merge(self):
        """Merge all experts into a single LoRA for inference.
        Uses uniform routing weights (average of all experts).
        Returns a simple LoRA (down, up) tuple.
        """
        # Average all experts
        merged_down = self.experts_down.mean(dim=0)  # (hidden, rank)
        merged_up = self.experts_up.mean(dim=0)  # (rank, hidden)
        return merged_down, merged_up

    def extra_repr(self):
        return (f"hidden={self.hidden_dim}, experts={self.n_experts}, "
                f"rank={self.rank}, top_k={self.top_k}, "
                f"params={sum(p.numel() for p in self.parameters()):,}")
