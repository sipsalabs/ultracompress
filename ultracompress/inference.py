"""
Inference Harness — Test compressed models end-to-end.

Loads a GGUF model's weights, compresses them, then compares:
  1. Layer-by-layer activation similarity (compressed vs original)
  2. Full forward pass logit comparison
  3. Text generation quality

Uses the model's actual architecture (parsed from GGUF metadata)
to build a minimal transformer forward pass.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture config parsed from GGUF."""
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    hidden_size: int = 4096
    intermediate_size: int = 11008
    vocab_size: int = 32000
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0


def parse_gguf_config(model_path: str) -> ModelConfig:
    """Extract model architecture from GGUF metadata."""
    from gguf import GGUFReader
    reader = GGUFReader(model_path)

    config = ModelConfig()
    meta = {}
    for field in reader.fields:
        name = field
        try:
            val = reader.fields[field]
            if hasattr(val, 'parts') and len(val.parts) > 0:
                raw = val.parts[-1]
                if hasattr(raw, '__len__'):
                    if len(raw) == 1:
                        v = raw[0]
                        meta[name] = int(v) if 'int' in str(type(v)) or 'uint' in str(type(v)) else float(v)
                    else:
                        # Check if it's a string (array of uint8 bytes)
                        try:
                            decoded = bytes(raw).decode('utf-8')
                            meta[name] = decoded
                        except (UnicodeDecodeError, TypeError):
                            meta[name] = list(raw)
                elif hasattr(raw, 'item'):
                    meta[name] = raw.item()
                else:
                    meta[name] = raw
        except Exception:
            pass

    # Dynamic architecture detection from GGUF metadata
    if "general.architecture" in meta:
        arch = meta["general.architecture"]
        if isinstance(arch, (list, tuple)):
            arch = arch[0]
        if isinstance(arch, bytes):
            arch = arch.decode()
        arch = str(arch)
        # Try architecture-specific keys
        for key in meta:
            if key.startswith(arch):
                short_key = key.replace(f"{arch}.", "")
                if "block_count" in short_key:
                    config.n_layers = int(meta[key])
                elif "head_count_kv" in short_key:
                    config.n_kv_heads = int(meta[key])
                elif "head_count" in short_key:
                    config.n_heads = int(meta[key])
                elif "embedding_length" in short_key:
                    config.hidden_size = int(meta[key])
                elif "feed_forward_length" in short_key:
                    config.intermediate_size = int(meta[key])

    config.head_dim = config.hidden_size // config.n_heads

    # Try to get vocab size from token embeddings tensor
    for tensor_info in reader.tensors:
        if "token_embd" in tensor_info.name:
            shape = list(tensor_info.shape)
            if len(shape) >= 2:
                config.vocab_size = int(max(shape))
            break

    return config


class RMSNorm:
    """RMSNorm using raw weight tensor."""
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight.float()
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        # Handle dimension mismatch (e.g., embedding dim != hidden dim)
        w = self.weight
        if x_normed.shape[-1] != w.shape[0]:
            if x_normed.shape[-1] < w.shape[0]:
                x_normed = F.pad(x_normed, (0, w.shape[0] - x_normed.shape[-1]))
            else:
                x_normed = x_normed[..., :w.shape[0]]
        return (x_normed * w).to(x.dtype)


def rope_embed(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Apply rotary position embeddings."""
    d = x.shape[-1]
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    t = positions.float()
    angles = torch.outer(t, freqs)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x_r, x_i = x_reshape[..., 0], x_reshape[..., 1]
    # Ensure angles broadcast correctly with attention head dims
    n_pairs = x_r.shape[-1]
    # Recompute angles to match actual dimension
    actual_freqs = 1.0 / (theta ** (torch.arange(0, n_pairs * 2, 2, device=x.device).float() / (n_pairs * 2)))
    actual_angles = torch.outer(positions.float(), actual_freqs)
    cos_a = torch.cos(actual_angles)
    sin_a = torch.sin(actual_angles)
    # Broadcast for batch dims
    while cos_a.dim() < x_r.dim():
        cos_a = cos_a.unsqueeze(0)
        sin_a = sin_a.unsqueeze(0)
    out_r = x_r * cos_a - x_i * sin_a
    out_i = x_r * sin_a + x_i * cos_a
    return torch.stack([out_r, out_i], dim=-1).reshape(x.shape)


def linear_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a linear layer: x @ W^T"""
    return F.linear(x.float(), weight.float())


class TransformerLayer:
    """Single transformer layer using raw weight tensors."""
    def __init__(self, weights: dict, config: ModelConfig):
        self.config = config
        self.attn_norm = RMSNorm(weights.get("attn_norm", torch.ones(config.hidden_size)))
        self.ffn_norm = RMSNorm(weights.get("ffn_norm", torch.ones(config.hidden_size)))

        self.q_weight = weights.get("attn_q")
        self.k_weight = weights.get("attn_k")
        self.v_weight = weights.get("attn_v")
        self.o_weight = weights.get("attn_output")

        self.gate_weight = weights.get("ffn_gate")
        self.up_weight = weights.get("ffn_up")
        self.down_weight = weights.get("ffn_down")

        # Check for QKV norm (some models have this)
        self.q_norm = None
        self.k_norm = None
        if "attn_q_norm" in weights:
            self.q_norm = RMSNorm(weights["attn_q_norm"])
        if "attn_k_norm" in weights:
            self.k_norm = RMSNorm(weights["attn_k_norm"])

    def attention(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        config = self.config

        # Handle dimension mismatch: project x if weight expects different input dim
        q_in = x if self.q_weight is None or self.q_weight.shape[1] == C else x[..., :self.q_weight.shape[1]]
        k_in = x if self.k_weight is None or self.k_weight.shape[1] == C else x[..., :self.k_weight.shape[1]]
        v_in = x if self.v_weight is None or self.v_weight.shape[1] == C else x[..., :self.v_weight.shape[1]]

        q = linear_forward(q_in, self.q_weight)
        k = linear_forward(k_in, self.k_weight)
        v = linear_forward(v_in, self.v_weight)

        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads

        # Derive head_dim from actual weight output dimensions (handles GQA with different Q/KV dims)
        q_head_dim = q.shape[-1] // n_heads
        kv_head_dim = k.shape[-1] // n_kv_heads

        q = q.reshape(B, T, n_heads, q_head_dim).transpose(1, 2)
        k = k.reshape(B, T, n_kv_heads, kv_head_dim).transpose(1, 2)
        v = v.reshape(B, T, n_kv_heads, kv_head_dim).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        # RoPE (apply to min of q/k dims for compatibility)
        rope_dim = min(q_head_dim, kv_head_dim)
        q_rope = q[..., :rope_dim]
        k_rope = k[..., :rope_dim]
        q_rope = rope_embed(q_rope, positions, config.rope_theta)
        k_rope = rope_embed(k_rope, positions, config.rope_theta)
        q = torch.cat([q_rope, q[..., rope_dim:]], dim=-1) if q_head_dim > rope_dim else q_rope
        k = torch.cat([k_rope, k[..., rope_dim:]], dim=-1) if kv_head_dim > rope_dim else k_rope

        # GQA: repeat k/v heads
        if n_kv_heads < n_heads:
            repeat_factor = n_heads // n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # For attention: if Q and K have different head dims, truncate Q to K dim for dot product
        if q_head_dim != kv_head_dim:
            q_attn = q[..., :kv_head_dim]
        else:
            q_attn = q

        # Scaled dot-product attention
        attn = torch.matmul(q_attn.float(), k.float().transpose(-2, -1)) / np.sqrt(kv_head_dim)

        # Causal mask
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.float())

        out = out.transpose(1, 2).reshape(B, T, -1)
        return linear_forward(out, self.o_weight)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        # Handle dimension mismatch for non-standard architectures
        C = x.shape[-1]
        gate_in = x if self.gate_weight is None or self.gate_weight.shape[1] == C else x[..., :self.gate_weight.shape[1]]
        up_in = x if self.up_weight is None or self.up_weight.shape[1] == C else x[..., :self.up_weight.shape[1]]

        gate = linear_forward(gate_in, self.gate_weight)
        up = linear_forward(up_in, self.up_weight)
        hidden = F.silu(gate) * up
        return linear_forward(hidden, self.down_weight)

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with safe residual
        attn_out = self.attention(self.attn_norm(x), positions)
        if attn_out.shape == x.shape:
            h = x + attn_out
        else:
            # Dimension mismatch — pad or truncate for residual
            h = x.clone()
            min_d = min(attn_out.shape[-1], x.shape[-1])
            h[..., :min_d] = h[..., :min_d] + attn_out[..., :min_d]

        # Pre-norm FFN with safe residual
        ffn_out = self.ffn(self.ffn_norm(h))
        if ffn_out.shape == h.shape:
            out = h + ffn_out
        else:
            out = h.clone()
            min_d = min(ffn_out.shape[-1], h.shape[-1])
            out[..., :min_d] = out[..., :min_d] + ffn_out[..., :min_d]
        return out


class MiniTransformer:
    """Minimal transformer for inference testing."""
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.embed_weight = None
        self.final_norm = None
        self.lm_head = None
        self.layers = []

    def load_weights(self, weight_dict: dict):
        """Load from a flat dict of name -> tensor."""
        self.embed_weight = weight_dict.get("token_embd.weight")
        norm_w = weight_dict.get("output_norm.weight", torch.ones(self.config.hidden_size))
        self.final_norm = RMSNorm(norm_w.to(self.device))
        self.lm_head = weight_dict.get("output.weight", self.embed_weight)

        for i in range(self.config.n_layers):
            prefix = f"blk.{i}."
            layer_weights = {}
            layer_key_map = {
                "attn_q": "attn_q.weight",
                "attn_k": "attn_k.weight",
                "attn_v": "attn_v.weight",
                "attn_output": "attn_output.weight",
                "attn_norm": "attn_norm.weight",
                "attn_q_norm": "attn_q_norm.weight",
                "attn_k_norm": "attn_k_norm.weight",
                "ffn_gate": "ffn_gate.weight",
                "ffn_up": "ffn_up.weight",
                "ffn_down": "ffn_down.weight",
                "ffn_norm": "ffn_norm.weight",
            }
            for our_key, gguf_suffix in layer_key_map.items():
                full_key = prefix + gguf_suffix
                if full_key in weight_dict:
                    layer_weights[our_key] = weight_dict[full_key].to(self.device)

            if "attn_q" in layer_weights:
                self.layers.append(TransformerLayer(layer_weights, self.config))

    def forward(self, token_ids: torch.Tensor, max_layers: int = None) -> torch.Tensor:
        """Run forward pass, return logits."""
        B, T = token_ids.shape
        positions = torch.arange(T, device=self.device)

        x = F.embedding(token_ids, self.embed_weight.to(self.device)).float()

        n_layers = min(len(self.layers), max_layers or len(self.layers))
        for i in range(n_layers):
            x = self.layers[i](x, positions)

        x = self.final_norm(x)
        logits = linear_forward(x, self.lm_head.to(self.device))
        return logits

    def generate(self, token_ids: torch.Tensor, max_new: int = 50, temperature: float = 0.7) -> list:
        """Simple greedy/sampling generation."""
        tokens = token_ids.clone()
        generated = []

        for _ in range(max_new):
            logits = self.forward(tokens)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        return generated


def compare_layer_outputs(
    original_weights: dict,
    compressed_weights: dict,
    config: ModelConfig,
    input_tokens: torch.Tensor,
    device: str = "cuda",
    max_layers: int = 4,
) -> dict:
    """
    Compare layer-by-layer outputs between original and compressed weights.

    Returns per-layer cosine similarity and relative error.
    """
    results = {}

    # Build both models
    orig_model = MiniTransformer(config, device)
    orig_model.load_weights(original_weights)

    comp_model = MiniTransformer(config, device)
    comp_model.load_weights(compressed_weights)

    n_layers = min(len(orig_model.layers), len(comp_model.layers), max_layers)

    # Run forward pass through each layer and compare
    B, T = input_tokens.shape
    positions = torch.arange(T, device=device)

    with torch.no_grad():
        x_orig = F.embedding(input_tokens, orig_model.embed_weight.to(device)).float()
        x_comp = F.embedding(input_tokens, comp_model.embed_weight.to(device)).float()

        for i in range(n_layers):
            x_orig = orig_model.layers[i](x_orig, positions)
            x_comp = comp_model.layers[i](x_comp, positions)

            # Compare
            cos_sim = F.cosine_similarity(
                x_orig.reshape(1, -1), x_comp.reshape(1, -1)
            ).item()
            rel_err = torch.norm(x_orig - x_comp).item() / (torch.norm(x_orig).item() + 1e-10)

            results[f"layer_{i}"] = {
                "cosine_sim": cos_sim,
                "relative_error": rel_err,
            }

    # Compare final logits
    with torch.no_grad():
        x_orig_final = orig_model.final_norm(x_orig)
        x_comp_final = comp_model.final_norm(x_comp)
        logits_orig = linear_forward(x_orig_final, orig_model.lm_head.to(device))
        logits_comp = linear_forward(x_comp_final, comp_model.lm_head.to(device))

        logit_cos = F.cosine_similarity(
            logits_orig.reshape(1, -1), logits_comp.reshape(1, -1)
        ).item()
        results["logits"] = {"cosine_sim": logit_cos}

        # Top-k agreement
        k = 10
        _, top_orig = logits_orig[0, -1].topk(k)
        _, top_comp = logits_comp[0, -1].topk(k)
        top_orig_set = set(top_orig.cpu().tolist())
        top_comp_set = set(top_comp.cpu().tolist())
        overlap = len(top_orig_set & top_comp_set) / k
        results["top10_agreement"] = overlap

        # Check if #1 prediction matches
        results["top1_match"] = (top_orig[0] == top_comp[0]).item()

    return results
