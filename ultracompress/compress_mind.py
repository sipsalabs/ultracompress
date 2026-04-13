"""
COMPRESSION MIND — Intelligence IS compression. Not a metaphor. Literally.

Hutter Prize proved: optimal text compression = optimal language model.
Solomonoff induction: the shortest program that generates the data IS
the best predictor of the data.

So: understanding = finding the shortest description.
Generating = decompressing that description.
Both directions. Same operation. Same model.

This architecture literally IS a compressor-decompressor:

COMPRESS (understanding):
  text → encoder → tiny latent code (the "meaning")
  The smaller the code, the better the understanding.

DECOMPRESS (generation):
  latent code → decoder → predicted next tokens
  The code contains everything needed to predict.

The SHARED FUNCTION is both compressor and decompressor.
Applied forward = compress (understand).
Applied backward = decompress (generate).

FRR already did this implicitly — 42x compression preserves meaning.
This makes it EXPLICIT — the model's job IS compression.

The key insight: if your latent code is small enough and accurate enough,
you have perfect understanding AND perfect generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CompressionLayer(nn.Module):
    """One step of compression/decompression.

    Compress: reduce sequence to smaller representation
    Decompress: expand representation back to sequence
    Same weights, different direction.
    """
    def __init__(self, hidden_dim, compress_ratio=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compress_ratio = compress_ratio

        # Shared weights for compress AND decompress
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

    def compress(self, x):
        """x: (B, S, D) → (B, S//ratio, D) — squeeze sequence"""
        B, S, D = x.shape
        new_S = max(S // self.compress_ratio, 1)

        h = self.norm(x)
        # Pool adjacent tokens into compressed representations
        if S > new_S:
            h = h[:, :new_S * self.compress_ratio].reshape(B, new_S, self.compress_ratio, D)
            # Weighted pooling via gate
            gate_weights = torch.sigmoid(self.gate(h.mean(dim=2)))  # (B, new_S, D)
            h = (h * gate_weights.unsqueeze(2)).sum(dim=2) / self.compress_ratio

        return F.silu(self.transform(h))

    def decompress(self, x, target_S):
        """x: (B, S_compressed, D) → (B, target_S, D) — expand sequence"""
        B, S, D = x.shape

        h = self.norm(x)
        h = F.silu(self.transform(h))

        # Expand by repeating and differentiating
        if S < target_S:
            h = h.repeat_interleave(self.compress_ratio, dim=1)[:, :target_S]
            # Add position-dependent variation
            pos = torch.arange(target_S, device=x.device).float().unsqueeze(0).unsqueeze(-1) / target_S
            h = h + torch.sin(pos * math.pi) * 0.1  # gentle positional modulation

        return h


class CompressMind(nn.Module):
    """Language model as compressor-decompressor.

    Input tokens → compress through N levels → tiny latent → decompress → predict

    The compression forces the model to find MEANING (can't store noise in tiny code).
    The decompression forces the model to GENERATE from meaning (not memorize).

    Shared weights at each level = FRR-style efficiency.
    """
    def __init__(self, hidden_dim, n_levels=4, compress_ratio=2, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_levels = n_levels
        self.compress_ratio = compress_ratio

        # SHARED compression/decompression layer
        self.layer = CompressionLayer(hidden_dim, compress_ratio)

        # Per-level modulation (like FRR)
        self.compress_gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_levels)
        ])
        self.decompress_gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_levels)
        ])

        # Cross-position at bottleneck (tiny, so even O(n²) is cheap)
        self.bottleneck_attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bottleneck_attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bottleneck_attn_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bottleneck_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Embeddings
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        B, S, D = x.shape

        # COMPRESS: sequence → bottleneck
        sizes = [S]
        residuals = [x]
        for level in range(self.n_levels):
            x = x * self.compress_gammas[level]
            x = self.layer.compress(x)
            sizes.append(x.shape[1])
            residuals.append(x)

        # BOTTLENECK: tiny sequence, full attention is cheap here
        # This is where MEANING lives — the compressed representation
        q = self.bottleneck_attn_q(x)
        k = self.bottleneck_attn_k(x)
        v = self.bottleneck_attn_v(x)
        BN_S = x.shape[1]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if BN_S > 1:
            mask = torch.triu(torch.ones(BN_S, BN_S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.bottleneck_out(attn @ v)

        # DECOMPRESS: bottleneck → sequence
        for level in range(self.n_levels - 1, -1, -1):
            target_S = sizes[level]
            x = x * self.decompress_gammas[level]
            x = self.layer.decompress(x, target_S)
            # Skip connection from compression path
            x = x + residuals[level] * 0.3

        x = self.out_norm(x)
        return self.lm_head(x)
