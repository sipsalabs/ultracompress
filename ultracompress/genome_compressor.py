"""
Genome Compressor — The core engine for behavioral compression.

Takes a model (local or API) and produces a tiny "genome" that
replicates its behavior. The genome replaces each transformer layer
with a micro-transformer that's 100-1000x smaller.

Key design for scaling:
  - Never needs full model in memory (queries layer by layer)
  - Works with API-accessible models (send tokens, get logits)
  - Trains end-to-end via KL divergence on logit distributions
  - Genome is a standard PyTorch model — runs on any hardware

Usage:
  # Local model
  compressor = GenomeCompressor(model_path="Qwen/Qwen3-0.6B")
  genome = compressor.compress(target_bpw=0.025, n_steps=5000)
  genome.save("qwen3_0.6b_genome.pt")

  # From API (future)
  compressor = GenomeCompressor(api_endpoint="http://model-server/v1")
  genome = compressor.compress(target_bpw=0.025, n_steps=10000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional, Callable


class MicroTransformerLayer(nn.Module):
    """A compressed transformer layer — tiny attention + FFN in bottleneck space.

    Projects big_dim -> small_dim, runs attention+FFN, projects back.
    Replaces a full transformer layer with ~100-1000x fewer parameters.
    """
    def __init__(self, big_dim, small_dim, n_heads, ff_mult=2):
        super().__init__()
        self.big_dim = big_dim
        self.small_dim = small_dim
        self.n_heads = n_heads
        self.head_dim = small_dim // n_heads

        # Down/up projections between full space and compressed space
        self.down = nn.Linear(big_dim, small_dim, bias=False)
        self.up = nn.Linear(small_dim, big_dim, bias=False)

        # Attention in compressed space
        self.qkv = nn.Linear(small_dim, 3 * small_dim, bias=False)
        self.o_proj = nn.Linear(small_dim, small_dim, bias=False)

        # FFN in compressed space
        ff_dim = small_dim * ff_mult
        self.gate = nn.Linear(small_dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(small_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, small_dim, bias=False)

        # Norms
        self.norm1 = nn.RMSNorm(small_dim)
        self.norm2 = nn.RMSNorm(small_dim)

    def forward(self, x):
        """x: (batch, seq_len, big_dim) -> residual delta (batch, seq_len, big_dim)"""
        B, T, _ = x.shape

        # Project down
        h = self.down(x)

        # Attention
        hn = self.norm1(h)
        qkv = self.qkv(hn).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.small_dim)
        h = h + self.o_proj(out)

        # FFN
        hn = self.norm2(h)
        h = h + self.down_proj(F.silu(self.gate(hn)) * self.up_proj(hn))

        # Project back up
        return self.up(h)


class GenomeModel(nn.Module):
    """A complete genome-compressed model.

    Contains: embedding (shared with original) + genome layers + LM head (shared).
    The genome layers replace all transformer layers.
    """
    def __init__(self, vocab_size, big_dim, small_dim, n_heads, n_layers,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.big_dim = big_dim
        self.n_layers = n_layers

        # Embedding and LM head — kept from original model (not compressed)
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, big_dim)

        if lm_head_weight is not None:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        else:
            self.lm_head = nn.Linear(big_dim, vocab_size, bias=False)

        if norm_weight is not None:
            self.norm = nn.RMSNorm(big_dim)
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        else:
            self.norm = nn.RMSNorm(big_dim)

        # Genome layers — the compressed replacements
        self.genome_layers = nn.ModuleList([
            MicroTransformerLayer(big_dim, small_dim, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, token_ids, max_layers=None):
        """Run inference through genome model."""
        x = self.embed(token_ids).float()
        n = max_layers or self.n_layers
        for i in range(min(n, len(self.genome_layers))):
            x = x + self.genome_layers[i](x)
        x = self.norm(x)
        return self.lm_head(x)

    def genome_param_count(self):
        """Count only the genome parameters (not embed/head)."""
        return sum(p.numel() for p in self.genome_layers.parameters())

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def genome_size_bytes(self):
        return self.genome_param_count() * 2  # FP16

    def save_genome(self, path):
        """Save only the genome layers (not embed/head — those come from original)."""
        torch.save({
            'genome_state': self.genome_layers.state_dict(),
            'config': {
                'big_dim': self.big_dim,
                'small_dim': self.genome_layers[0].small_dim,
                'n_heads': self.genome_layers[0].n_heads,
                'n_layers': self.n_layers,
            }
        }, path)

    def load_genome(self, path):
        data = torch.load(path, weights_only=True)
        self.genome_layers.load_state_dict(data['genome_state'])


@dataclass
class CompressionResult:
    genome: GenomeModel
    top1_accuracy: float
    top10_overlap: float
    genome_params: int
    genome_size_mb: float
    original_layer_params: int
    compression_ratio: float
    bpw: float
    training_steps: int
    training_time: float


class GenomeCompressor:
    """Compresses a model into a genome via behavioral distillation."""

    def __init__(self, model_weights: dict = None, model_config=None, device='cuda'):
        self.device = device
        self.weights = model_weights
        self.config = model_config

    def compress(
        self,
        small_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = None,
        n_steps: int = 5000,
        batch_size: int = 8,
        seq_len: int = 32,
        lr: float = 0.001,
        eval_every: int = 1000,
        eval_samples: int = 100,
        teacher_forward: Callable = None,
        verbose: bool = True,
    ) -> CompressionResult:
        """Run compression.

        Args:
            small_dim: bottleneck dimension for micro-transformer layers
            n_heads: attention heads in micro-transformer
            n_layers: number of layers to compress (None = all)
            n_steps: training steps
            teacher_forward: function(token_ids) -> logits. If None, uses internal model.
            verbose: print progress
        """
        from ultracompress.inference import ModelConfig, MiniTransformer

        # Build teacher model
        if teacher_forward is None:
            assert self.weights is not None, "Need model weights or teacher_forward"
            teacher = MiniTransformer(self.config, self.device)
            hf_to_gguf = {
                'self_attn.q_proj.weight': 'attn_q.weight',
                'self_attn.k_proj.weight': 'attn_k.weight',
                'self_attn.v_proj.weight': 'attn_v.weight',
                'self_attn.o_proj.weight': 'attn_output.weight',
                'self_attn.q_norm.weight': 'attn_q_norm.weight',
                'self_attn.k_norm.weight': 'attn_k_norm.weight',
                'input_layernorm.weight': 'attn_norm.weight',
                'post_attention_layernorm.weight': 'ffn_norm.weight',
                'mlp.gate_proj.weight': 'ffn_gate.weight',
                'mlp.up_proj.weight': 'ffn_up.weight',
                'mlp.down_proj.weight': 'ffn_down.weight',
            }
            gd = {}
            gd['token_embd.weight'] = self.weights['model.embed_tokens.weight'].float()
            gd['output_norm.weight'] = self.weights.get('model.norm.weight',
                                                         torch.ones(self.config.hidden_size)).float()
            gd['output.weight'] = self.weights.get('lm_head.weight', gd['token_embd.weight']).float()

            actual_layers = n_layers or self.config.n_layers
            for li in range(actual_layers):
                for h, g in hf_to_gguf.items():
                    k = f'model.layers.{li}.{h}'
                    if k in self.weights:
                        gd[f'blk.{li}.{g}'] = self.weights[k].float()
            teacher.load_weights(gd)

            def teacher_forward(tokens):
                with torch.no_grad():
                    return teacher.forward(tokens, max_layers=actual_layers)

            embed_w = gd['token_embd.weight'].to(self.device)
            head_w = gd['output.weight'].to(self.device)
            norm_w = gd['output_norm.weight'].to(self.device)
        else:
            actual_layers = n_layers
            embed_w = None
            head_w = None
            norm_w = None

        # Count original layer params
        orig_params = sum(
            self.weights[k].numel()
            for k in self.weights
            if any(f'layers.{i}' in k for i in range(actual_layers))
            and 'weight' in k and self.weights[k].ndim >= 2
        ) if self.weights else 0

        # Build genome
        genome = GenomeModel(
            vocab_size=self.config.vocab_size if self.config else 151936,
            big_dim=self.config.hidden_size if self.config else 1024,
            small_dim=small_dim,
            n_heads=n_heads,
            n_layers=actual_layers,
            embed_weight=embed_w,
            lm_head_weight=head_w,
            norm_weight=norm_w,
        ).to(self.device)

        genome_params = genome.genome_param_count()
        bpw = genome_params * 16 / orig_params if orig_params > 0 else 0
        compression = orig_params / genome_params if genome_params > 0 else 0

        if verbose:
            print(f"Genome: {genome_params:,} params ({genome_params*2/1e6:.1f} MB)")
            print(f"Original layers: {orig_params:,} params")
            print(f"Compression: {compression:.0f}x (BPW={bpw:.4f})")
            print()

        # Train
        opt = torch.optim.AdamW(genome.genome_layers.parameters(), lr=lr, weight_decay=0.005)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

        t_start = time.time()
        for step in range(n_steps):
            torch.manual_seed(step)
            tokens = torch.randint(100, 100000, (batch_size, seq_len), device=self.device)

            teacher_logits = teacher_forward(tokens)[:, -1, :]
            student_logits = genome(tokens, max_layers=actual_layers)[:, -1, :]

            loss = F.kl_div(
                F.log_softmax(student_logits / 2, dim=-1),
                F.softmax(teacher_logits / 2, dim=-1),
                reduction='batchmean',
            ) * 4

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(genome.genome_layers.parameters(), 1.0)
            opt.step()
            sched.step()

            if verbose and eval_every and step % eval_every == 0:
                t1, t10 = self._evaluate(genome, teacher_forward, actual_layers, eval_samples)
                elapsed = time.time() - t_start
                print(f"  Step {step:>5}/{n_steps}: loss={loss.item():.3f} "
                      f"Top1={t1*100:.0f}% Top10={t10*100:.0f}% [{elapsed:.0f}s]")
                sys.stdout.flush()

        # Final evaluation
        top1, top10 = self._evaluate(genome, teacher_forward, actual_layers, eval_samples)
        training_time = time.time() - t_start

        if verbose:
            print(f"\nFinal: Top1={top1*100:.0f}% Top10={top10*100:.0f}%")
            print(f"Training time: {training_time:.0f}s")

        return CompressionResult(
            genome=genome,
            top1_accuracy=top1,
            top10_overlap=top10,
            genome_params=genome_params,
            genome_size_mb=genome_params * 2 / 1e6,
            original_layer_params=orig_params,
            compression_ratio=compression,
            bpw=bpw,
            training_steps=n_steps,
            training_time=training_time,
        )

    def _evaluate(self, genome, teacher_forward, n_layers, n_samples):
        t1_matches = 0
        t10_overlaps = []
        for trial in range(n_samples):
            torch.manual_seed(trial * 13 + 5000)
            tokens = torch.randint(100, 50000, (1, 16), device=self.device)
            with torch.no_grad():
                tl = teacher_forward(tokens)
                tp = tl[0, -1].argmax().item()
                tt10 = set(tl[0, -1].topk(10).indices.tolist())

                gl = genome(tokens, max_layers=n_layers)
                gp = gl[0, -1].argmax().item()
                gt10 = set(gl[0, -1].topk(10).indices.tolist())

                if tp == gp:
                    t1_matches += 1
                t10_overlaps.append(len(tt10 & gt10) / 10)

        return t1_matches / n_samples, sum(t10_overlaps) / len(t10_overlaps)
