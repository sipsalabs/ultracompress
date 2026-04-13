"""
SPIRAL FRR — Stolen from three 2026 papers, combined with our FRR.

1. Multi-resolution recursion (SpiralFormer): early passes process coarse
   representation, later passes process fine. Different scales = more info.
2. Memory bank (Adaptive Loops paper): small persistent memory that carries
   info across ALL recursion passes. The "process" has state.
3. Monotonic recursion loss (RecursiveVLM): guarantee quality improves
   with each additional pass. No degradation from depth.

Combined with our FRR: ONE shared block, applied at multiple resolutions,
with persistent memory, and loss that ensures each pass improves.

This is Sip's "4D dimensional box" — information flows across scales AND
across recursion depth through the memory bank.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryBank(nn.Module):
    """Small persistent memory that carries info across recursion passes.

    Like working memory in the brain — persists across "thoughts" (passes).
    Each pass can read from and write to memory.

    Total params: 2 * hidden * mem_slots + 2 * hidden * mem_slots = tiny
    """
    def __init__(self, hidden_dim, n_slots=16):
        super().__init__()
        self.n_slots = n_slots
        # Learned initial memory
        self.initial_memory = nn.Parameter(torch.randn(1, n_slots, hidden_dim) * 0.01)
        # Read/write projections
        self.read_proj = nn.Linear(hidden_dim, n_slots, bias=False)
        self.write_gate = nn.Linear(hidden_dim + n_slots, hidden_dim, bias=False)
        nn.init.zeros_(self.write_gate.weight)

    def init_state(self, batch_size, device):
        return self.initial_memory.expand(batch_size, -1, -1).to(device)

    def read(self, x, memory):
        """x: (B, S, D), memory: (B, n_slots, D) -> context: (B, S, D)"""
        # Attention over memory slots
        weights = F.softmax(self.read_proj(x), dim=-1)  # (B, S, n_slots)
        context = weights @ memory  # (B, S, D)
        return context

    def write(self, x, memory):
        """Update memory based on current hidden state."""
        # Average pool sequence to get summary
        summary = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        summary = summary.expand_as(memory)  # (B, n_slots, D)
        # Gated update
        gate_input = torch.cat([memory, self.read_proj(summary)], dim=-1)
        gate = torch.sigmoid(self.write_gate(gate_input))
        new_memory = memory + gate * (summary - memory) * 0.1
        return new_memory


class SpiralFRR(nn.Module):
    """FRR with multi-resolution recursion + memory bank.

    The shared block is applied at 3 resolution levels:
    - Coarse (pool 4x): captures global structure, fast
    - Medium (pool 2x): captures mid-level patterns
    - Fine (full resolution): captures details

    Memory persists across ALL passes at ALL resolutions.
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale, vocab_size,
                 n_mem_slots=16,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # THE SHARED BLOCK (same as FRR)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn_gate = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

        # Per-layer modulation
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(self.total_layers)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(self.total_layers)
        ])
        self.iter_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.total_layers)
        ])

        # MEMORY BANK (persistent across passes)
        self.memory = MemoryBank(hidden_dim, n_mem_slots)

        # Resolution schedule: which passes use which resolution
        # Early passes = coarse, later = fine (like SpiralFormer)
        self.pool_factors = []
        for i in range(self.total_layers):
            progress = i / max(self.total_layers - 1, 1)
            if progress < 0.33:
                self.pool_factors.append(4)  # coarse
            elif progress < 0.66:
                self.pool_factors.append(2)  # medium
            else:
                self.pool_factors.append(1)  # fine

        # Upsample projections for resolution changes
        self.up2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.up2.weight)
        nn.init.eye_(self.up4.weight)

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

    def _apply_block(self, x, gamma, beta):
        """Apply the shared transformer block with modulation."""
        B, S, D = x.shape
        h = self.norm1(x) * gamma + beta

        # Attention
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(out)

        # FFN
        h = self.norm2(x) * gamma + beta
        gate = F.silu(self.ffn_gate(h))
        x = x + self.ffn_down(gate * self.ffn_up(h))
        return x

    def _pool(self, x, factor):
        """Pool sequence to lower resolution."""
        if factor <= 1: return x
        B, S, D = x.shape
        new_S = max(S // factor, 1)
        # Average pool
        x = x[:, :new_S * factor].reshape(B, new_S, factor, D).mean(dim=2)
        return x

    def _unpool(self, x_low, original_S, factor):
        """Upsample back to original resolution."""
        if factor <= 1: return x_low
        # Repeat and project
        x_up = x_low.repeat_interleave(factor, dim=1)[:, :original_S]
        if factor == 2:
            x_up = self.up2(x_up)
        else:
            x_up = self.up4(x_up)
        return x_up

    def forward(self, tokens, return_intermediates=False):
        x = self.embed(tokens).float()
        B, S, D = x.shape
        intermediates = []

        # Initialize memory
        mem = self.memory.init_state(B, x.device)

        layer_count = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                pool_factor = self.pool_factors[layer_count]

                # Read from memory (inject persistent context)
                mem_context = self.memory.read(x, mem)
                x = x + mem_context * 0.1  # gentle injection

                # Pool to current resolution
                x_pooled = self._pool(x, pool_factor)

                # Apply shared block at this resolution
                residual = x_pooled
                x_pooled = self._apply_block(x_pooled, self.gammas[layer_count], self.betas[layer_count])
                x_pooled = residual + (x_pooled - residual) * torch.sigmoid(self.iter_scale[layer_count])

                # Unpool back to full resolution
                if pool_factor > 1:
                    x = x + self._unpool(x_pooled - self._pool(x, pool_factor), S, pool_factor)
                else:
                    x = x_pooled

                # Write to memory (update persistent state)
                mem = self.memory.write(x, mem)

                if return_intermediates:
                    intermediates.append(x)

                layer_count += 1

        x = self.out_norm(x)
        logits = self.lm_head(x)

        if return_intermediates:
            return logits, intermediates
        return logits
