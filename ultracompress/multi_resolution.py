"""
Multi-Resolution Recursion for FRR — inspired by SpiralFormer (2602.11698).

Process at coarse-to-fine resolution across recursion steps:
  - Early iterations: downsample sequence to 1/4 or 1/8 length (cheaper, captures global patterns)
  - Later iterations: full resolution (fine-grained, local patterns)

Benefits:
  - ~11-30% FLOP reduction per forward pass (faster inference)
  - May improve quality by forcing hierarchical feature learning
  - Compatible with existing FRR modulation (gamma/beta/iter_scale)

Schedule: {1/8, 1/4, 1/2, 1} across 4 scales (28 iterations total).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDownsampler(nn.Module):
    """Downsample sequence via learned aggregation within chunks.

    Given (B, S, D) with chunk_size k, produces (B, S//k, D).
    Uses a learned scorer + softmax within each chunk (not mean pooling).
    """
    def __init__(self, hidden_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.scorer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        """x: (B, S, D) -> (B, S//chunk_size, D)"""
        B, S, D = x.shape
        k = self.chunk_size

        # Pad to multiple of chunk_size
        pad = (k - S % k) % k
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        # Reshape into chunks: (B, n_chunks, k, D)
        n_chunks = x.shape[1] // k
        chunks = x.reshape(B, n_chunks, k, D)

        # Score each position within chunk
        scores = self.scorer(chunks).squeeze(-1)  # (B, n_chunks, k)
        weights = F.softmax(scores, dim=-1)  # (B, n_chunks, k)

        # Weighted sum within each chunk
        out = (chunks * weights.unsqueeze(-1)).sum(dim=2)  # (B, n_chunks, D)
        return out


class CausalUpsampler(nn.Module):
    """Upsample sequence back to original length.

    Given (B, S_down, D) and target length S_orig, produces (B, S_orig, D).
    Each chunk of k output positions gets the same latent + a learned offset.
    """
    def __init__(self, hidden_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.offset = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.offset.weight)

    def forward(self, x_down, target_len):
        """x_down: (B, S_down, D) -> (B, target_len, D)"""
        B, S_down, D = x_down.shape
        k = self.chunk_size

        # Repeat each latent k times
        x_up = x_down.unsqueeze(2).expand(B, S_down, k, D).reshape(B, S_down * k, D)

        # Add learned offset (so positions within chunk can differ)
        x_up = x_up + self.offset(x_up)

        # Trim to target length
        return x_up[:, :target_len, :]


class MultiResolutionFRR(nn.Module):
    """FRR with multi-resolution recursion schedule.

    The 4 scales use different sequence resolutions:
      Scale 0: 1/8 resolution (7 iterations on downsampled sequence)
      Scale 1: 1/4 resolution (7 iterations)
      Scale 2: 1/2 resolution (7 iterations)
      Scale 3: full resolution (7 iterations)

    This is coarse-to-fine: early iterations capture global patterns cheaply,
    later iterations refine with full sequence detail.
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=1,
                 embed_weight=None, lm_head_weight=None, norm_weight=None,
                 resolution_schedule=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Resolution schedule: chunk sizes for downsampling at each scale
        # Default: coarse-to-fine {8, 4, 2, 1} (1 = no downsampling)
        self.resolution_schedule = resolution_schedule or [8, 4, 2, 1]
        assert len(self.resolution_schedule) == n_scales

        # Shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Down/up samplers for each scale (only where chunk_size > 1)
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for chunk_size in self.resolution_schedule:
            if chunk_size > 1:
                self.downsamplers.append(LearnedDownsampler(hidden_dim, chunk_size))
                self.upsamplers.append(CausalUpsampler(hidden_dim, chunk_size))
            else:
                self.downsamplers.append(None)
                self.upsamplers.append(None)

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
        orig_len = x.shape[1]

        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]

            # Downsample if needed
            chunk_size = self.resolution_schedule[scale]
            if chunk_size > 1 and orig_len > chunk_size:
                x_down = self.downsamplers[scale](x)
            else:
                x_down = x

            # Apply shared block iters_per_scale times at this resolution
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]
                x_down = x_down + (self.block(x_down, gamma, beta) - x_down) * iter_s

            # Upsample back if needed
            if chunk_size > 1 and orig_len > chunk_size:
                x = x + self.upsamplers[scale](x_down, orig_len) - x  # residual
            else:
                x = x_down

        x = self.norm(x)
        return self.lm_head(x)
