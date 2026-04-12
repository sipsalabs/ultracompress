"""
Controller Hypernetwork for input-dependent FRR modulation.
Inspired by Ouroboros V2 (2604.02051).

Instead of static per-layer gamma/beta (same modulation regardless of input),
the controller observes the current hidden state and generates modulation
vectors dynamically. This lets the shared block adapt its behavior to what
it's actually processing, not just which layer position it's at.

tiny MLP: hidden_dim -> bottleneck -> hidden_dim * 2 (gamma + beta)
"""
import torch
import torch.nn as nn


class ModulationController(nn.Module):
    """Generate input-dependent gamma/beta from hidden state.

    Given h: (B, S, D), produces gamma, beta: (B, S, D) that depend
    on the actual content being processed.
    """
    def __init__(self, hidden_dim, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim * 2, bias=False),
        )
        # Initialize to identity modulation (gamma=1, beta=0)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, h):
        """h: (B, S, D) -> gamma: (B, S, D), beta: (B, S, D)"""
        out = self.net(h.detach())  # detach to avoid controller affecting main grad flow
        gamma, beta = out.chunk(2, dim=-1)
        gamma = 1.0 + gamma  # residual: start as identity
        return gamma, beta


class ControlledFRR(nn.Module):
    """FRR with input-dependent controller modulation.

    Replaces static per-scale gamma/beta with a learned controller
    that generates modulation from the current hidden state.
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=1, controller_bottleneck=64,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-layer controller (input-dependent modulation)
        self.controllers = nn.ModuleList([
            ModulationController(hidden_dim, controller_bottleneck)
            for _ in range(self.total_layers)
        ])

        # Per-iteration scaling (static, cheap)
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
                # Controller generates input-dependent modulation
                gamma, beta = self.controllers[layer_idx](x)

                # Apply shared block with dynamic modulation
                iter_s = self.iter_scale[scale, it]
                x = x + (self.block(x, gamma, beta) - x) * iter_s

                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def param_summary(self):
        block_p = sum(p.numel() for p in self.block.parameters())
        ctrl_p = sum(p.numel() for p in self.controllers.parameters())
        other_p = self.iter_scale.numel()
        total = block_p + ctrl_p + other_p
        return {
            'block': block_p,
            'controllers': ctrl_p,
            'iter_scale': other_p,
            'total_trainable': total,
        }
