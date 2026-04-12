"""
4D DIMENSIONAL BLOCK — Sip's idea.

Instead of a flat shared block that sees only the current hidden state,
this block operates in a higher-dimensional space where:
- Dim 1-2: token positions + features (standard)
- Dim 3: recursion depth (which pass we're on)
- Dim 4: scale/resolution (coarse-to-fine)

The key insight: the block can form connections ACROSS recursion depths.
Pass 5 can directly reference what pass 2 produced. This gives the
single block "memory" across its recursive applications.

Implementation: maintain a 4D state tensor (batch, seq, depth, hidden)
that accumulates across passes. Cross-depth attention lets later passes
learn from earlier ones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossDepthAttention(nn.Module):
    """Attention across recursion depths.

    Given the accumulated hidden states from all previous passes,
    the current pass can attend to any previous pass's output.
    This gives the shared block "memory" across recursions.
    """
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize near-zero so it starts as a skip connection
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, current, history):
        """
        current: (B, S, D) — current pass hidden state
        history: (B, S, N_prev, D) — accumulated previous pass states
        Returns: (B, S, D) — current + cross-depth attention
        """
        B, S, D = current.shape
        N_prev = history.shape[2]

        # Query from current, Key/Value from history
        q = self.q_proj(current).reshape(B, S, self.n_heads, self.head_dim)
        # Reshape history for attention: (B, S, N_prev, D) -> keys/values
        h_flat = history.reshape(B * S, N_prev, D)
        k = self.k_proj(h_flat).reshape(B, S, N_prev, self.n_heads, self.head_dim)
        v = self.v_proj(h_flat).reshape(B, S, N_prev, self.n_heads, self.head_dim)

        # q: (B, S, H, d) -> (B, H, S, d)
        # k: (B, S, N, H, d) -> (B, H, S, N, d)
        q = q.permute(0, 2, 1, 3)  # (B, H, S, d)
        k = k.permute(0, 3, 1, 2, 4)  # (B, H, S, N, d)
        v = v.permute(0, 3, 1, 2, 4)  # (B, H, S, N, d)

        # Attention: q attends to all N_prev historical states
        # (B, H, S, d) @ (B, H, S, d, N) -> (B, H, S, N)
        attn = torch.einsum('bhsd,bhsnd->bhsn', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        # (B, H, S, N) @ (B, H, S, N, d) -> (B, H, S, d)
        out = torch.einsum('bhsn,bhsnd->bhsd', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)  # (B, S, D)

        return current + self.o_proj(out)


class DimensionalFRR(nn.Module):
    """FRR with 4D dimensional block — cross-depth connections.

    The shared block can "see" all previous recursion passes,
    forming a 4D computation space (tokens x features x depth x scale).
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=1, depth_heads=4,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Standard shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Cross-depth attention (the 4D connection)
        self.cross_depth = CrossDepthAttention(hidden_dim, depth_heads)

        # Per-scale modulation (standard)
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Gate to blend cross-depth info (starts at 0 = pure standard FRR)
        self.depth_gate = nn.Parameter(torch.zeros(self.total_layers))

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
        B, S, D = x.shape

        # History buffer: accumulate hidden states across passes
        history = []

        layer_idx = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                # Standard shared block pass
                iter_s = self.iter_scale[scale, it]
                x_block = x + (self.block(x, gamma, beta) - x) * iter_s

                # Cross-depth attention (4D connection)
                if len(history) > 0:
                    gate = torch.sigmoid(self.depth_gate[layer_idx])
                    h_stack = torch.stack(history, dim=2)  # (B, S, N_prev, D)
                    x_depth = self.cross_depth(x_block, h_stack)
                    x = (1 - gate) * x_block + gate * x_depth
                else:
                    x = x_block

                # Save to history
                history.append(x.detach())  # detach to save memory
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)
