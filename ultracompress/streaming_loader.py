"""Streaming Model Loader — Load models shard by shard.

Never loads the full model into RAM. Processes one shard at a time,
extracts the weights needed, discards the shard.

For a 405B model (810GB), peak RAM usage is ~8GB (one shard).
For a 10T model, same ~8GB per shard.

This is how we compress models too big to download fully.
"""

import os
import torch
from typing import Iterator, Tuple, Dict, Optional
from safetensors.torch import load_file
import json


def get_model_info(model_path: str) -> dict:
    """Get model architecture info without loading weights."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def list_shards(model_path: str) -> list:
    """List all safetensors shard files."""
    files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ])
    return [os.path.join(model_path, f) for f in files]


def stream_layer_weights(
    model_path: str,
    layer_idx: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load weights for a single layer from sharded model.

    Scans shards to find the layer's weights, loads only those.
    Peak RAM: one shard (~4-8GB) + extracted layer weights.
    """
    prefix = f"model.layers.{layer_idx}."
    layer_weights = {}

    for shard_path in list_shards(model_path):
        # Load shard
        shard = load_file(shard_path, device=device)

        # Extract matching weights
        for name, tensor in shard.items():
            if name.startswith(prefix):
                short_name = name[len(prefix):]
                layer_weights[short_name] = tensor.float()

        # Free shard memory
        del shard

        # If we found all expected weights, stop scanning
        if len(layer_weights) >= 7:  # q,k,v,o,gate,up,down + norms
            break

    return layer_weights


def stream_special_weights(
    model_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load embedding, LM head, and final norm from sharded model."""
    special = {}
    needed = {
        'model.embed_tokens.weight',
        'model.norm.weight',
        'lm_head.weight',
    }

    for shard_path in list_shards(model_path):
        shard = load_file(shard_path, device=device)
        for name, tensor in shard.items():
            if name in needed:
                special[name] = tensor.float()
                needed.discard(name)
        del shard
        if not needed:
            break

    return special


def stream_all_layers(
    model_path: str,
    n_layers: int = None,
    device: str = "cpu",
) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
    """Stream layer weights one at a time.

    Yields (layer_idx, weight_dict) for each layer.
    Peak RAM: ~one shard size.
    """
    config = get_model_info(model_path)
    if n_layers is None:
        n_layers = config.get('num_hidden_layers', 32)

    for layer_idx in range(n_layers):
        weights = stream_layer_weights(model_path, layer_idx, device)
        yield layer_idx, weights


def build_teacher_cache_streaming(
    model_path: str,
    n_samples: int = 5000,
    batch_size: int = 16,
    seq_len: int = 32,
    n_layers: int = None,
    device: str = "cuda",
) -> dict:
    """Build teacher output cache by streaming through the model.

    Instead of loading the full model, runs inference layer-by-layer:
    1. Load embedding, generate random input activations
    2. For each layer: load weights, run forward pass, free weights
    3. Apply final norm + LM head
    4. Cache the logits

    Peak RAM: embedding + one layer + activations.
    """
    from ultracompress.inference import ModelConfig, TransformerLayer, RMSNorm

    config = get_model_info(model_path)
    if not config:
        raise ValueError(f"No config.json in {model_path}")

    hidden = config['hidden_size']
    n_heads = config['num_attention_heads']
    n_kv = config.get('num_key_value_heads', n_heads)
    head_dim = hidden // n_heads
    if n_layers is None:
        n_layers = config['num_hidden_layers']
    vocab = config['vocab_size']

    mc = ModelConfig(
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
        hidden_size=hidden, intermediate_size=config['intermediate_size'],
        vocab_size=vocab, head_dim=head_dim,
    )

    # Load special weights
    print("Loading embedding + LM head...")
    special = stream_special_weights(model_path, device="cpu")
    embed = special['model.embed_tokens.weight'].to(device)
    norm_w = special.get('model.norm.weight', torch.ones(hidden)).to(device)
    lm_head = special.get('lm_head.weight', embed).to(device)

    print(f"Streaming {n_layers} layers for {n_samples} samples...")
    all_tokens = []
    all_logits = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        torch.manual_seed(start)
        tokens = torch.randint(100, vocab, (end - start, seq_len), device=device)

        with torch.no_grad():
            x = torch.nn.functional.embedding(tokens, embed).float()
            positions = torch.arange(seq_len, device=device)

            # Stream through each layer
            for layer_idx in range(n_layers):
                layer_w = stream_layer_weights(model_path, layer_idx, device)

                # Build a TransformerLayer and run it
                # Map weight names to what TransformerLayer expects
                tw = {}
                name_map = {
                    'self_attn.q_proj.weight': 'attn_q',
                    'self_attn.k_proj.weight': 'attn_k',
                    'self_attn.v_proj.weight': 'attn_v',
                    'self_attn.o_proj.weight': 'attn_output',
                    'input_layernorm.weight': 'attn_norm',
                    'post_attention_layernorm.weight': 'ffn_norm',
                    'mlp.gate_proj.weight': 'ffn_gate',
                    'mlp.up_proj.weight': 'ffn_up',
                    'mlp.down_proj.weight': 'ffn_down',
                    'self_attn.q_norm.weight': 'attn_q_norm',
                    'self_attn.k_norm.weight': 'attn_k_norm',
                }
                for src, dst in name_map.items():
                    if src in layer_w:
                        tw[dst] = layer_w[src].to(device)

                layer = TransformerLayer(tw, mc)
                x = layer(x, positions)
                del layer_w, tw
                torch.cuda.empty_cache()

            # Final norm + logits
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x_normed = x.float() * torch.rsqrt(variance + 1e-6) * norm_w
            logits = torch.nn.functional.linear(x_normed, lm_head)

        all_tokens.append(tokens.cpu())
        all_logits.append(logits[:, -1, :].cpu())

        if (start // batch_size) % 10 == 0:
            print(f"  {start}/{n_samples}")

    return {
        'tokens': torch.cat(all_tokens),
        'logits': torch.cat(all_logits),
        'n_layers': n_layers,
        'embed': special['model.embed_tokens.weight'],
        'norm': special.get('model.norm.weight', torch.ones(hidden)),
        'head': special.get('lm_head.weight', special['model.embed_tokens.weight']),
        'config': mc,
    }
