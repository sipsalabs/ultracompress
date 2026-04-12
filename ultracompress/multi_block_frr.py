"""Multi-Block FRR: 2-3 specialized shared blocks for different depth ranges.
Block A (early/syntax), Block B (mid/semantics), Block C (late/prediction).
3 blocks x 10.5M = 31.5M trainable => ~14x compression (vs 42x single-block FRR).
"""
import torch, torch.nn as nn
from .moonshot import FractalBlock, LoRAAdapter, GatedRecurrence


class MultiBlockFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, total_layers=28, n_blocks=3,
                 vocab_size=151936, ff_mult=2, lora_rank=32,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_layers = total_layers
        self.n_blocks = n_blocks

        # Auto-compute ranges: split total_layers into n_blocks roughly equal parts
        boundaries = [round(total_layers * i / n_blocks) for i in range(n_blocks + 1)]
        self.ranges = [(boundaries[i], boundaries[i + 1]) for i in range(n_blocks)]

        # Specialized shared blocks
        self.blocks = nn.ModuleList([FractalBlock(hidden_dim, n_heads, ff_mult) for _ in range(n_blocks)])

        # Per-block modulation (gamma/beta per virtual layer within each block's range)
        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()
        for start, end in self.ranges:
            self.gammas.append(nn.Parameter(torch.ones(end - start, hidden_dim)))
            self.betas.append(nn.Parameter(torch.zeros(end - start, hidden_dim)))

        # Gated recurrence + LoRA per virtual layer
        self.gates = nn.ModuleList([GatedRecurrence(hidden_dim) for _ in range(total_layers)])
        self.adapters = nn.ModuleList([LoRAAdapter(hidden_dim, lora_rank) for _ in range(total_layers)])

        # Embedding and head
        self.embed = (nn.Embedding.from_pretrained(embed_weight, freeze=True)
                      if embed_weight is not None else nn.Embedding(vocab_size, hidden_dim))
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens, max_layers=None):
        x = self.embed(tokens).float()
        total = max_layers or self.total_layers
        layer_idx = 0
        for block_id, (start, end) in enumerate(self.ranges):
            block = self.blocks[block_id]
            for local_i in range(end - start):
                if layer_idx >= total:
                    break
                gamma = self.gammas[block_id][local_i]
                beta = self.betas[block_id][local_i]
                h = block(x, gamma, beta)
                x = self.gates[layer_idx](h, x)
                x = self.adapters[layer_idx](x)
                layer_idx += 1
        return self.lm_head(self.norm(x))

    def trainable_params(self):
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        mod_p = sum(g.numel() for g in self.gammas) + sum(b.numel() for b in self.betas)
        gate_p = sum(p.numel() for g in self.gates for p in g.parameters())
        adapter_p = sum(p.numel() for a in self.adapters for p in a.parameters())
        return block_p + mod_p + gate_p + adapter_p

    def block_summary(self):
        labels = ["early(syntax)", "mid(semantics)", "late(prediction)"]
        for i, (s, e) in enumerate(self.ranges):
            name = labels[i] if i < len(labels) else f"block_{i}"
            bp = sum(p.numel() for p in self.blocks[i].parameters())
            print(f"  Block {i} [{name}] layers {s}-{e-1}: {bp:,} params")
        print(f"  Total trainable: {self.trainable_params():,}")
