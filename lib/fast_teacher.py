"""
Fast teacher forward pass using torch.compile.

The MiniTransformer forward pass is bottlenecked by per-layer Python overhead
(28 kernel launches × 30μs overhead = 840μs just for launch latency).
torch.compile fuses the operations into fewer CUDA graphs.

Usage:
    from lib.fast_teacher import FastTeacher
    teacher = FastTeacher('qwen3_1.7b_cache.pt', device='cuda:1')
    logits = teacher(token_ids)  # [B, T, V]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultracompress.inference import ModelConfig, MiniTransformer


class CompiledTeacherLayer(nn.Module):
    """nn.Module wrapper for a single transformer layer (enables torch.compile)."""

    def __init__(self, layer_weights: dict, config: ModelConfig):
        super().__init__()
        h = config.hidden_size
        self.config = config

        # Register weights as buffers (not parameters — no grad needed)
        self.register_buffer('attn_norm_w', layer_weights['attn_norm'].float())
        self.register_buffer('q_w', layer_weights['attn_q'].float())
        self.register_buffer('k_w', layer_weights['attn_k'].float())
        self.register_buffer('v_w', layer_weights['attn_v'].float())
        self.register_buffer('o_w', layer_weights['attn_output'].float())
        self.register_buffer('ffn_norm_w', layer_weights['ffn_norm'].float())
        self.register_buffer('gate_w', layer_weights['ffn_gate'].float())
        self.register_buffer('up_w', layer_weights['ffn_up'].float())
        self.register_buffer('down_w', layer_weights['ffn_down'].float())

        # Optional QKV norms
        if 'attn_q_norm' in layer_weights:
            self.register_buffer('q_norm_w', layer_weights['attn_q_norm'].float())
            self.register_buffer('k_norm_w', layer_weights['attn_k_norm'].float())
        else:
            self.q_norm_w = None
            self.k_norm_w = None

    @staticmethod
    def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + eps) * w).to(x.dtype)

    @staticmethod
    def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
        d = x.shape[-1]
        n_pairs = d // 2
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
        angles = torch.outer(positions.float(), freqs[:n_pairs])
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        while cos_a.dim() < x.dim():
            cos_a = cos_a.unsqueeze(0)
            sin_a = sin_a.unsqueeze(0)
        x_r = x[..., 0::2]
        x_i = x[..., 1::2]
        out_r = x_r * cos_a - x_i * sin_a
        out_i = x_r * sin_a + x_i * cos_a
        return torch.stack([out_r, out_i], dim=-1).reshape(x.shape)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        cfg = self.config

        # Attention
        h = self._rms_norm(x, self.attn_norm_w)
        q = F.linear(h, self.q_w)
        k = F.linear(h, self.k_w)
        v = F.linear(h, self.v_w)

        q = q.reshape(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        if self.q_norm_w is not None:
            q = self._rms_norm(q, self.q_norm_w)
            k = self._rms_norm(k, self.k_norm_w)

        q = self._rope(q, positions, cfg.rope_theta)
        k = self._rope(k, positions, cfg.rope_theta)

        # GQA
        if cfg.n_kv_heads < cfg.n_heads:
            rep = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Use F.scaled_dot_product_attention (Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        out = out.transpose(1, 2).reshape(B, T, C)
        x = x + F.linear(out, self.o_w)

        # FFN
        h = self._rms_norm(x, self.ffn_norm_w)
        x = x + F.linear(F.silu(F.linear(h, self.gate_w)) * F.linear(h, self.up_w), self.down_w)

        return x


class FastTeacher(nn.Module):
    """Compiled teacher model — 2-4x faster than MiniTransformer."""

    def __init__(self, cache_path: str, device: str = 'cuda:0', compile_mode: str = 'reduce-overhead'):
        super().__init__()
        self.device = device

        # Load weights
        wd = torch.load(cache_path, weights_only=True)
        hf_to_gguf = {
            'self_attn.q_proj.weight': 'attn_q',
            'self_attn.k_proj.weight': 'attn_k',
            'self_attn.v_proj.weight': 'attn_v',
            'self_attn.o_proj.weight': 'attn_output',
            'self_attn.q_norm.weight': 'attn_q_norm',
            'self_attn.k_norm.weight': 'attn_k_norm',
            'input_layernorm.weight': 'attn_norm',
            'post_attention_layernorm.weight': 'ffn_norm',
            'mlp.gate_proj.weight': 'ffn_gate',
            'mlp.up_proj.weight': 'ffn_up',
            'mlp.down_proj.weight': 'ffn_down',
        }

        embed = wd['model.embed_tokens.weight'].float()
        norm = wd.get('model.norm.weight', torch.ones(embed.shape[1])).float()
        lm_head = wd.get('lm_head.weight', embed).float()
        hidden = embed.shape[1]
        vocab = embed.shape[0]

        # Detect model size
        n_layers = sum(1 for k in wd if '.self_attn.q_proj.weight' in k)
        n_heads = wd[f'model.layers.0.self_attn.q_proj.weight'].shape[0] // (hidden // 16)
        # For Qwen3: n_heads=16, n_kv_heads=8 (1.7B), n_heads=32, n_kv_heads=8 (8B)
        kv_dim = wd[f'model.layers.0.self_attn.k_proj.weight'].shape[0]
        head_dim = hidden // n_heads
        n_kv_heads = kv_dim // head_dim
        ff_dim = wd[f'model.layers.0.mlp.gate_proj.weight'].shape[0]

        config = ModelConfig(
            n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
            hidden_size=hidden, intermediate_size=ff_dim,
            vocab_size=vocab, head_dim=head_dim,
        )

        # Build layers
        self.register_buffer('embed_w', embed.to(device))
        self.register_buffer('norm_w', norm.to(device))
        self.register_buffer('lm_head_w', lm_head.to(device))

        layers = []
        for i in range(n_layers):
            lw = {}
            for hf_key, our_key in hf_to_gguf.items():
                k = f'model.layers.{i}.{hf_key}'
                if k in wd:
                    lw[our_key] = wd[k].to(device)
            layers.append(CompiledTeacherLayer(lw, config))
        self.layers = nn.ModuleList(layers)
        self.config = config
        del wd

        # Compile the forward method
        if compile_mode:
            print(f"  Compiling teacher forward (mode={compile_mode})...")
            self._compiled_forward = torch.compile(
                self._raw_forward, mode=compile_mode, fullgraph=False
            )
        else:
            self._compiled_forward = self._raw_forward

        print(f"  FastTeacher: {n_layers} layers, {hidden}d, {vocab} vocab, {device}")

    def _raw_forward(self, x: torch.Tensor, positions: torch.Tensor,
                     n_layers: int) -> torch.Tensor:
        for i in range(n_layers):
            x = self.layers[i](x, positions)
        return x

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, max_layers: int | None = None) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=self.device)
        x = F.embedding(token_ids, self.embed_w).float()
        n = min(len(self.layers), max_layers or len(self.layers))
        x = self._compiled_forward(x, positions, n)
        x = CompiledTeacherLayer._rms_norm(x, self.norm_w)
        return F.linear(x, self.lm_head_w)
