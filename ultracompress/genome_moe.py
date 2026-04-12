"""
MoE Genome — Mixture of Expert genome layers.

Key insight: A single bottleneck can't capture the conditional behavior
of a transformer layer. Different tokens need different transformations.

MoE genome: Multiple tiny experts + a router. Each token activates
top-K experts. This gives much more effective capacity without
proportionally more parameters (inactive experts cost 0 compute).

For 1000T target at 20GB:
  200 layers * 16 experts * ~3.8M params/expert = 12.2B params = ~24GB @ FP16
  With top-2 routing: only 7.6M active params per token per layer
  With INT8 experts: 12GB total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoEGenomeLayer(nn.Module):
    """Mixture-of-Experts genome layer.

    Each expert is a tiny FFN in bottleneck space.
    Router picks top-K experts per token.
    """

    def __init__(self, big_dim, expert_dim, n_experts=8, top_k=2, ff_mult=2):
        super().__init__()
        self.big_dim = big_dim
        self.expert_dim = expert_dim
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: selects which experts to use per token
        self.router = nn.Linear(big_dim, n_experts, bias=False)

        # Experts: each is a down->nonlinear->up path
        self.expert_down = nn.Parameter(torch.randn(n_experts, big_dim, expert_dim) * 0.02)
        self.expert_up = nn.Parameter(torch.zeros(n_experts, expert_dim, big_dim))

        # Shared gate for each expert (SwiGLU-style)
        self.expert_gate = nn.Parameter(torch.randn(n_experts, big_dim, expert_dim) * 0.02)

        # Learnable output scale (start small for stable training)
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Load balancing loss coefficient
        self.balance_coeff = 0.01

    def forward(self, x):
        """x: (B, T, big_dim) -> delta: (B, T, big_dim)"""
        B, T, D = x.shape

        # Router logits
        router_logits = self.router(x)  # (B, T, n_experts)

        # Top-K expert selection
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)  # (B, T, K)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, K)

        # Compute expert outputs for selected experts
        x_flat = x.reshape(-1, D)  # (B*T, D)
        output = torch.zeros_like(x_flat)  # (B*T, D)

        indices_flat = top_k_indices.reshape(-1, self.top_k)  # (B*T, K)
        weights_flat = top_k_weights.reshape(-1, self.top_k)  # (B*T, K)

        for k in range(self.top_k):
            expert_idx = indices_flat[:, k]  # (B*T,)
            weight = weights_flat[:, k:k+1]  # (B*T, 1)

            # Gather expert weights for each token
            # Use einsum for batched expert computation
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                x_e = x_flat[mask]  # (n_tokens, D)
                # SwiGLU: gate * up
                gate = F.silu(x_e @ self.expert_down[e])    # (n_tokens, expert_dim)
                up = x_e @ self.expert_gate[e]               # (n_tokens, expert_dim)
                hidden = gate * up                            # (n_tokens, expert_dim)
                out_e = hidden @ self.expert_up[e]            # (n_tokens, D)
                output[mask] += out_e * weight[mask]

        output = output.reshape(B, T, D)
        return output * self.scale

    def aux_loss(self, x):
        """Load balancing auxiliary loss."""
        router_logits = self.router(x)  # (B, T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        # Want uniform distribution across experts
        avg_probs = router_probs.mean(dim=[0, 1])  # (n_experts,)
        target = torch.ones_like(avg_probs) / self.n_experts
        return F.mse_loss(avg_probs, target) * self.balance_coeff


class MoEGenomeModel(nn.Module):
    """Complete model with MoE genome layers."""

    def __init__(self, vocab_size, big_dim, expert_dim, n_experts, top_k,
                 n_layers, embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.big_dim = big_dim
        self.n_layers = n_layers

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, big_dim)

        if lm_head_weight is not None:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        else:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)

        if norm_weight is not None:
            self.norm = nn.RMSNorm(big_dim)
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        else:
            self.norm = nn.RMSNorm(big_dim)

        self.genome_layers = nn.ModuleList([
            MoEGenomeLayer(big_dim, expert_dim, n_experts, top_k)
            for _ in range(n_layers)
        ])

    def forward(self, token_ids, max_layers=None):
        x = self.embed(token_ids).float()
        n = max_layers or self.n_layers
        for i in range(min(n, len(self.genome_layers))):
            x = x + self.genome_layers[i](x)
        x = self.norm(x)
        return self.lm_head(x)

    def genome_param_count(self):
        return sum(p.numel() for p in self.genome_layers.parameters())

    def save_genome(self, path):
        state = {
            'genome_state': {k: v for k, v in self.state_dict().items()
                           if 'genome_layers' in k},
            'config': {
                'big_dim': self.big_dim,
                'n_layers': self.n_layers,
                'expert_dim': self.genome_layers[0].expert_dim,
                'n_experts': self.genome_layers[0].n_experts,
                'top_k': self.genome_layers[0].top_k,
            },
        }
        torch.save(state, path)
