"""
Prelude/Coda FRR — inspired by Ouroboros V2 (2604.02051, April 2026).

Instead of sharing ALL layers, keep the first few (Prelude) and last few (Coda)
layers untied, and only share the middle layers via FRR recursion.

Rationale: first layers handle tokenization/embedding adaptation, last layers
handle output/prediction specialization. Middle layers do the bulk of
"reasoning" and are more similar to each other — better candidates for sharing.

Ouroboros V2 keeps 17 of 36 layers (47%). We can experiment with different
Prelude/Coda sizes on our 28-layer Qwen3-0.6B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreludeCodaFRR(nn.Module):
    """FRR with untied Prelude (first N layers) and Coda (last M layers).

    Architecture:
      Prelude: first `n_prelude` teacher layers (untied, full quality)
      Recurrent: shared FRR block for middle layers
      Coda: last `n_coda` teacher layers (untied, full quality)
    """

    def __init__(self, hidden_dim, n_heads, total_layers=28,
                 n_prelude=2, n_coda=2,
                 n_scales=4, iters_per_scale=None,
                 vocab_size=151936, ff_mult=1,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.total_layers = total_layers
        self.n_prelude = n_prelude
        self.n_coda = n_coda
        n_recurrent = total_layers - n_prelude - n_coda
        assert n_recurrent > 0, f"No recurrent layers: {total_layers} - {n_prelude} - {n_coda} <= 0"

        self.n_recurrent = n_recurrent
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale or (n_recurrent // n_scales)

        # Prelude: independent transformer layers (loaded from teacher)
        self.prelude_layers = nn.ModuleList()  # Will be populated from teacher weights

        # Recurrent: shared FRR block
        self.shared_block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-recurrent-layer modulation
        self.rec_gamma = nn.Parameter(torch.ones(n_recurrent, hidden_dim))
        self.rec_beta = nn.Parameter(torch.zeros(n_recurrent, hidden_dim))
        self.rec_scale = nn.Parameter(torch.ones(n_recurrent))

        # Coda: independent transformer layers (loaded from teacher)
        self.coda_layers = nn.ModuleList()  # Will be populated from teacher weights

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

    def load_prelude_coda_from_teacher(self, teacher_layers):
        """Load untied layers from teacher.

        teacher_layers: list of nn.Module (the teacher's transformer layers)
        Copies first n_prelude and last n_coda layers.
        These are FROZEN — only the shared block trains.
        """
        # Prelude
        for i in range(self.n_prelude):
            layer = teacher_layers[i]
            self.prelude_layers.append(layer)
            for p in layer.parameters():
                p.requires_grad = False

        # Coda
        for i in range(self.total_layers - self.n_coda, self.total_layers):
            layer = teacher_layers[i]
            self.coda_layers.append(layer)
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, tokens):
        x = self.embed(tokens).float()
        positions = torch.arange(x.shape[1], device=x.device)

        # Prelude: run untied first layers
        for layer in self.prelude_layers:
            x = layer(x, positions)

        # Recurrent: shared block with per-layer modulation
        for i in range(self.n_recurrent):
            gamma = self.rec_gamma[i]
            beta = self.rec_beta[i]
            scale = self.rec_scale[i]
            x = x + (self.shared_block(x, gamma, beta) - x) * scale

        # Coda: run untied last layers
        for layer in self.coda_layers:
            x = layer(x, positions)

        x = self.norm(x)
        return self.lm_head(x)

    def compression_ratio(self, teacher_total_params):
        """Calculate actual compression ratio accounting for Prelude/Coda."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_stored = trainable + frozen  # Prelude/Coda must be stored too
        return teacher_total_params / total_stored
