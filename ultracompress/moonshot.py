"""
MOONSHOT ARCHITECTURES — Inherently 1000x more parameter-efficient.

Not compression. New architectures where intelligence doesn't require
trillions of independent parameters.

Architecture 1: FRACTAL RESIDUAL RECURSION (FRR)
  One shared transformer block applied recursively at multiple scales.
  32 effective layers from 1 set of weights.
  Per-scale modulation gives each "virtual layer" different behavior.

Architecture 2: GENOMIC WEIGHT EXPRESSION (GWE)
  A tiny MLP generates all weights on-the-fly from coordinates.
  10M params can express weights for a 10B-scale model.
  Like DNA encoding an organism.

Architecture 3: HOLOGRAPHIC WEIGHT INTERFERENCE (HWI)
  One shared complex tensor stores all weights via superposition.
  Layer-specific keys reconstruct per-layer weights via interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
# ARCHITECTURE 1: FRACTAL RESIDUAL RECURSION
# ================================================================

class LoRAAdapter(nn.Module):
    """Tiny per-layer adapter. Low-rank residual that specializes the shared block."""
    def __init__(self, hidden_dim, rank=16):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)  # Start as identity

    def forward(self, x):
        return x + self.up(self.down(x))


class GatedRecurrence(nn.Module):
    """Gated recurrence from Ouroboros / "Thinking Deeper" (Chen 2026).

    Essential for stable deep recursion. Without it, recursive application
    makes models WORSE (confirmed by Ouroboros ablation).

    Gate initialized to 88% retention (bias=-2.0), creating a gradient
    highway. Learns to open during training.
    """
    def __init__(self, hidden_dim, init_bias=-2.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, h_new, h_old):
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_new, h_old], dim=-1)))
        return gate * h_new + (1 - gate) * h_old


class FractalBlock(nn.Module):
    """One shared transformer-like block that gets reused at all depths.

    Contains: attention + FFN, but shared across all "virtual layers."
    Scale-conditional modulation makes each application behave differently.
    """
    def __init__(self, hidden_dim, n_heads=8, ff_mult=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Shared attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Shared FFN (SwiGLU)
        ff_dim = hidden_dim * ff_mult
        self.gate = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.up = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)

        # Norms
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, scale_gamma=None, scale_beta=None):
        B, T, D = x.shape

        # Scale-conditional modulation (if provided)
        h = self.norm1(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)

        # Attention
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(out)

        # FFN with modulation
        h = self.norm2(x)
        if scale_gamma is not None:
            h = h * scale_gamma + (scale_beta if scale_beta is not None else 0)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))

        return x


class FractalModel(nn.Module):
    """Fractal Residual Recursion — one block, many virtual layers.

    Schedule: [(scale_0, n_iters), (scale_1, n_iters), ...]
    Total effective layers = sum of all n_iters.
    Total parameters = 1 block + tiny per-scale modulation.
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=8,
                 vocab_size=151936, ff_mult=2,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Shared block (THE core — reused everywhere)
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation (tiny — just gamma + beta vectors)
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))

        # Per-iteration modulation within scale (even tinier)
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Optional: per-layer LoRA adapters (V3 enhancement)
        self.adapters = None  # Set via enable_adapters()

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

    def enable_adapters(self, rank=16):
        """Add per-layer LoRA adapters (V3 enhancement).
        Adds ~32K params per virtual layer for specialization."""
        # Get the device from existing parameters
        dev = next(self.parameters()).device
        self.adapters = nn.ModuleList([
            LoRAAdapter(self.hidden_dim, rank)
            for _ in range(self.total_layers)
        ]).to(dev)  # Move to same device as model
        adapter_params = sum(p.numel() for p in self.adapters.parameters())
        print(f"  Enabled LoRA adapters: {adapter_params:,} extra params ({adapter_params*2/1e6:.1f} MB)")

    def forward(self, tokens, max_layers=None, return_hidden=False):
        x = self.embed(tokens).float()
        total = max_layers or self.total_layers
        hidden_states = [] if return_hidden else None

        layer_count = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                if layer_count >= total:
                    break
                # Apply shared block with scale modulation
                iter_s = self.iter_scale[scale, it]
                x = x + (self.block(x, gamma, beta) - x) * iter_s
                # Apply per-layer LoRA adapter if enabled (V3)
                if self.adapters is not None:
                    x = self.adapters[layer_count](x)
                if return_hidden:
                    hidden_states.append(x)
                layer_count += 1

        x = self.norm(x)
        logits = self.lm_head(x)
        if return_hidden:
            return logits, hidden_states
        return logits

    def fractal_params(self):
        """Just the fractal-specific params (block + modulation + adapters)."""
        total = (sum(p.numel() for p in self.block.parameters()) +
                self.scale_gamma.numel() + self.scale_beta.numel() +
                self.iter_scale.numel())
        if self.adapters is not None:
            total += sum(p.numel() for p in self.adapters.parameters())
        return total


# ================================================================
# ARCHITECTURE 2: GENOMIC WEIGHT EXPRESSION
# ================================================================

class WeightGenome(nn.Module):
    """A tiny MLP that generates weight matrix blocks on-the-fly.

    Input: (layer_fraction, row_fraction, col_fraction, function_type)
    Output: a block_size x block_size weight block

    The genome learns the RULES for generating weights, not the weights themselves.
    """
    def __init__(self, block_size=16, hidden_dim=256, n_genome_layers=4):
        super().__init__()
        self.block_size = block_size

        # Positional encoding for inputs
        self.n_freqs = 6
        input_dim = 4 * (2 * self.n_freqs + 1)  # 4 coords with sin/cos encoding

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_genome_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, block_size * block_size))
        self.net = nn.Sequential(*layers)

    def _encode(self, x):
        """Positional encoding."""
        enc = [x]
        for freq in range(self.n_freqs):
            f = 2.0 ** freq
            enc.append(torch.sin(f * math.pi * x))
            enc.append(torch.cos(f * math.pi * x))
        return torch.cat(enc, dim=-1)

    def generate_block(self, layer_frac, row_frac, col_frac, func_type):
        """Generate one block of a weight matrix."""
        coord = torch.tensor([[layer_frac, row_frac, col_frac, func_type]],
                            device=next(self.parameters()).device)
        encoded = self._encode(coord)
        block = self.net(encoded)
        return block.reshape(self.block_size, self.block_size)


class GWEModel(nn.Module):
    """Genomic Weight Expression language model.

    Uses a tiny genome to generate all transformer weights on-the-fly.
    No stored weights except the genome itself + embeddings.
    """
    def __init__(self, hidden_dim, n_heads, n_layers, vocab_size=151936,
                 genome_hidden=256, block_size=16,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = hidden_dim // n_heads
        self.block_size = block_size

        # THE GENOME — one tiny network generates everything
        self.genome = WeightGenome(block_size=block_size, hidden_dim=genome_hidden)

        # Per-layer norms (cheap, critical)
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(hidden_dim), nn.RMSNorm(hidden_dim)])
            for _ in range(n_layers)
        ])

        # Embeddings
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        self.final_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.final_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def _generate_matrix(self, layer_idx, func_type, rows, cols):
        """Generate a full weight matrix from the genome, block by block."""
        bs = self.block_size
        layer_frac = layer_idx / max(self.n_layers - 1, 1)

        blocks_r = math.ceil(rows / bs)
        blocks_c = math.ceil(cols / bs)

        matrix = torch.zeros(blocks_r * bs, blocks_c * bs,
                           device=next(self.genome.parameters()).device)

        for br in range(blocks_r):
            for bc in range(blocks_c):
                row_frac = br / max(blocks_r - 1, 1)
                col_frac = bc / max(blocks_c - 1, 1)
                block = self.genome.generate_block(layer_frac, row_frac, col_frac, func_type)
                matrix[br*bs:(br+1)*bs, bc*bs:(bc+1)*bs] = block

        return matrix[:rows, :cols]

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        B, T, D = x.shape
        n = max_layers or self.n_layers

        for li in range(min(n, self.n_layers)):
            # Generate weights for this layer on-the-fly
            h = self.norms[li][0](x)

            # Simplified: single matrix transform instead of full attention
            # (Full attention would need Q/K/V/O generation — too slow for testing)
            W = self._generate_matrix(li, 0.0, D, D)  # func_type 0 = "transform"
            attn_out = F.linear(h, W)
            x = x + attn_out * 0.1

            # FFN via genome
            h = self.norms[li][1](x)
            W_ffn = self._generate_matrix(li, 0.5, D, D)  # func_type 0.5 = "FFN"
            x = x + F.silu(F.linear(h, W_ffn)) * 0.1

        x = self.final_norm(x)
        return self.lm_head(x)

    def genome_params(self):
        return sum(p.numel() for p in self.genome.parameters())


# ================================================================
# ARCHITECTURE 3: HOLOGRAPHIC WEIGHT INTERFERENCE
# ================================================================

class HolographicModel(nn.Module):
    """Holographic Weight Interference — one complex hologram, many layer keys.

    A single shared complex-valued tensor (the "hologram") stores ALL layer
    weights via superposition. Each layer reconstructs its own weight matrix
    by interfering with a unique low-rank complex "address key" pair.

    Weight reconstruction:
        W[layer] = real_part(hologram * outer_product(key_a[layer], key_b[layer]))

    Parameters:
        hologram: hidden_dim^2 complex = 2 * hidden_dim^2 real
        keys: n_layers * 2 * rank * hidden_dim complex = n_layers * 4 * rank * hidden_dim real
    """
    def __init__(self, hidden_dim, n_heads, n_layers, rank=16, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = hidden_dim // n_heads
        self.rank = rank

        # THE HOLOGRAM — one shared complex tensor storing all weight information
        # Initialize with small random complex values
        self.hologram_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.hologram_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # Per-layer address keys: key_a and key_b (complex, low-rank)
        # W[layer] = real_part(hologram * outer(key_a, key_b))
        # We have 2 function types per layer: attention-like and FFN-like
        n_funcs = n_layers * 2  # 2 weight matrices per layer
        self.key_a_real = nn.Parameter(torch.randn(n_funcs, rank, hidden_dim) * 0.02)
        self.key_a_imag = nn.Parameter(torch.randn(n_funcs, rank, hidden_dim) * 0.02)
        self.key_b_real = nn.Parameter(torch.randn(n_funcs, rank, hidden_dim) * 0.02)
        self.key_b_imag = nn.Parameter(torch.randn(n_funcs, rank, hidden_dim) * 0.02)

        # Per-layer output scales (tiny, helps training)
        self.layer_scale = nn.Parameter(torch.ones(n_funcs) * 0.1)

        # Per-layer norms (cheap, critical for training)
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.RMSNorm(hidden_dim), nn.RMSNorm(hidden_dim)])
            for _ in range(n_layers)
        ])

        # Embeddings (shared from teacher)
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        self.final_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.final_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def _reconstruct_weight(self, func_idx):
        """Reconstruct a weight matrix from hologram via key interference.

        W = real_part(hologram * sum_r(outer(key_a[r], key_b[r])))

        The outer product of complex keys creates a rank-R interference pattern.
        Elementwise multiply with the hologram, take real part => reconstructed weight.
        """
        # Build complex hologram
        hologram = torch.complex(self.hologram_real, self.hologram_imag)

        # Build complex keys for this function
        ka = torch.complex(self.key_a_real[func_idx], self.key_a_imag[func_idx])  # (rank, D)
        kb = torch.complex(self.key_b_real[func_idx], self.key_b_imag[func_idx])  # (rank, D)

        # Interference pattern: sum of rank-1 outer products
        # pattern shape: (D, D) from sum over rank of (D,1)*(1,D)
        pattern = torch.einsum('rd,re->de', ka, kb.conj())  # (D, D)

        # Reconstruct: elementwise multiply hologram with pattern, take real part
        W = (hologram * pattern).real  # (D, D)

        return W * self.layer_scale[func_idx]

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        B, T, D = x.shape
        n = max_layers or self.n_layers

        for li in range(min(n, self.n_layers)):
            # Attention-like transform
            h = self.norms[li][0](x)
            W_attn = self._reconstruct_weight(li * 2)
            x = x + F.linear(h, W_attn)

            # FFN-like transform
            h = self.norms[li][1](x)
            W_ffn = self._reconstruct_weight(li * 2 + 1)
            x = x + F.silu(F.linear(h, W_ffn))

        x = self.final_norm(x)
        return self.lm_head(x)

    def holographic_params(self):
        """Count hologram + key parameters (excluding shared embed/head/norm)."""
        holo = self.hologram_real.numel() + self.hologram_imag.numel()
        keys = (self.key_a_real.numel() + self.key_a_imag.numel() +
                self.key_b_real.numel() + self.key_b_imag.numel())
        scales = self.layer_scale.numel()
        norms = sum(p.numel() for n in self.norms for m in n for p in m.parameters())
        return holo + keys + scales + norms
