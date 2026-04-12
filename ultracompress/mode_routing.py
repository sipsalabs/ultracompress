"""
MODE ROUTING — A new approach to the dynamic modulation problem.

The insight: dynamic modulation (controller, 4D) is unstable because
it creates feedback loops. But static modulation limits expressivity.

Solution: maintain K static modulation "modes" (each a gamma/beta pair),
and use a cheap router to SELECT which mode to apply at each layer.

Static weights = stable training.
Dynamic selection = combinatorial expressivity.
4 modes * 28 layers = 4^28 possible configurations = virtually infinite.

This is like Mixture of Experts but for the MODULATION, not the FFN.
The shared block stays the same. Only the operating point changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModeRouter(nn.Module):
    """Routes each layer to one of K static modulation modes.

    Cheap routing: linear projection from hidden state mean to K scores.
    Selects top-1 mode per layer (hard routing) or soft mixture.
    """
    def __init__(self, hidden_dim, n_modes=4, soft=True):
        super().__init__()
        self.n_modes = n_modes
        self.soft = soft
        # Route based on mean hidden state (cheap, stable)
        self.router = nn.Linear(hidden_dim, n_modes, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x):
        """x: (B, S, D) -> weights: (B, n_modes)"""
        # Mean pool across sequence for routing decision
        x_mean = x.mean(dim=1)  # (B, D)
        logits = self.router(x_mean.detach())  # (B, K) — detach for stability
        if self.soft:
            return F.softmax(logits, dim=-1)  # (B, K)
        else:
            # Hard routing: one-hot but with straight-through gradient
            idx = logits.argmax(dim=-1)  # (B,)
            one_hot = F.one_hot(idx, self.n_modes).float()  # (B, K)
            return one_hot + logits.softmax(-1) - logits.softmax(-1).detach()


class ModeRoutedFRR(nn.Module):
    """FRR with mode-routed modulation.

    K static modulation profiles. Router selects which to use per layer.
    Stable (all modes are static). Expressive (combinatorial selection).
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=1, n_modes=4,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.n_modes = n_modes

        # Shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # K static modulation modes
        self.mode_gammas = nn.Parameter(torch.ones(n_modes, hidden_dim))
        self.mode_betas = nn.Parameter(torch.zeros(n_modes, hidden_dim))

        # Per-layer router (selects which mode to use)
        self.routers = nn.ModuleList([
            ModeRouter(hidden_dim, n_modes, soft=True)
            for _ in range(self.total_layers)
        ])

        # Per-layer iteration scaling (static, cheap)
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Embedding and head
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()

        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                # Route: select modulation mode based on current hidden state
                weights = self.routers[layer_idx](x)  # (B, K)

                # Weighted combination of static modes
                # (B, K) @ (K, D) -> (B, D) then broadcast to (B, 1, D)
                gamma = (weights @ self.mode_gammas).unsqueeze(1)  # (B, 1, D)
                beta = (weights @ self.mode_betas).unsqueeze(1)    # (B, 1, D)

                # Apply shared block
                iter_s = self.iter_scale[scale, it]
                x = x + (self.block(x, gamma, beta) - x) * iter_s

                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)
