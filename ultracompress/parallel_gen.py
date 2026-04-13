"""
PARALLEL GENERATION — Generate entire sequences at once, not token by token.

Every LLM generates autoregressively: predict token 1, then 2, then 3...
This is O(N) forward passes for N tokens. SLOW.

What if we generated ALL tokens simultaneously?
Input: prompt tokens
Output: ALL continuation tokens in ONE forward pass

The trick: instead of predicting "next token given all previous,"
predict "all tokens given the prompt." The model fills in a
blank canvas all at once, like a diffusion model fills in an image.

This is fundamentally different from transformers.
Transformers MUST be sequential (causal mask).
This model is PARALLEL by design.

Connection to our work:
- Rotation engine: already operates on all positions simultaneously
- HWI: holographic = all info at once
- Process: the computation IS the output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ParallelGenerator(nn.Module):
    """Generate full sequences in one shot.

    Architecture:
    1. Embed prompt tokens + blank tokens (learnable [BLANK] embeddings)
    2. Apply rotation engine to ALL tokens simultaneously
    3. Read out predictions from blank positions
    4. Iteratively refine (like diffusion: start rough, sharpen)

    N refinement steps << N tokens, so this is faster than autoregressive.
    """
    def __init__(self, hidden_dim, n_planes=64, n_refine=8,
                 max_gen=128, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_refine = n_refine
        self.max_gen = max_gen

        # Learnable blank token embedding (positions to fill)
        self.blank_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)

        # Position encoding for generation positions
        self.pos_encode = nn.Parameter(torch.randn(1, max_gen, hidden_dim) * 0.01)

        # Rotation layers for refinement (shared across refinement steps!)
        from .rotation_engine import RotationLayer
        self.rotations = nn.ModuleList([
            RotationLayer(hidden_dim, n_planes) for _ in range(4)
        ])

        # Per-refinement modulation
        self.refine_gamma = nn.Parameter(torch.ones(n_refine, hidden_dim))
        self.refine_beta = nn.Parameter(torch.zeros(n_refine, hidden_dim))

        # Cross-position mixing (causal for prompt, bidirectional for blanks)
        hub_dim = 64
        self.hub_write = nn.Linear(hidden_dim, hub_dim, bias=False)
        self.hub_read = nn.Linear(hub_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.hub_read.weight)

        # Norms
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_refine)])

        # Embedding and head
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, prompt_tokens, n_gen=None):
        """Generate n_gen tokens given prompt, ALL AT ONCE.

        Returns logits for ALL positions (prompt + generated).
        """
        B, P = prompt_tokens.shape
        G = n_gen or self.max_gen

        # Embed prompt
        prompt_emb = self.embed(prompt_tokens).float()  # (B, P, D)

        # Create blank canvas for generation
        blanks = self.blank_embed.expand(B, G, -1) + self.pos_encode[:, :G, :]

        # Concatenate: [prompt | blanks]
        x = torch.cat([prompt_emb, blanks], dim=1)  # (B, P+G, D)

        # Iterative refinement (like diffusion steps)
        for r in range(self.n_refine):
            gamma = self.refine_gamma[r]
            beta = self.refine_beta[r]

            # Modulate
            h = x * gamma + beta

            # Global hub mixing
            hub = self.hub_write(h).mean(dim=1, keepdim=True)
            h = h + self.hub_read(hub)

            # Rotate (use rotation r % 4 — shared block, different step)
            rot_idx = r % len(self.rotations)
            h = self.rotations[rot_idx](h)

            # Activate + residual
            x = x + self.norms[r](F.silu(h) - x) * 0.5

        x = self.out_norm(x)
        return self.lm_head(x)  # (B, P+G, vocab)

    def generate(self, prompt_tokens, n_gen=64):
        """Generate text in ONE forward pass."""
        logits = self.forward(prompt_tokens, n_gen)
        # Take argmax from generation positions
        gen_logits = logits[:, prompt_tokens.shape[1]:, :]
        gen_tokens = gen_logits.argmax(dim=-1)
        return torch.cat([prompt_tokens, gen_tokens], dim=1)
