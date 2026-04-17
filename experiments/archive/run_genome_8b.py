"""Scale test: Genome compression on Qwen3-8B.

If quality is BETTER on 8B than 0.6B at same genome size,
that proves larger models are easier to compress (more redundancy).
This is the key validation for scaling to 10T+.

WARNING: 8B model needs ~16GB RAM to load. Uses streaming
to load one shard at a time for cache building.
"""
import torch, sys, time
import torch.nn.functional as F
sys.path.insert(0, '.')
from safetensors.torch import load_file
import os

device = 'cuda'

# Qwen3-8B path
model_path = "C:/Users/sip/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

print("=== SCALING TEST: Genome compression on Qwen3-8B ===")
print("If this works better than 0.6B, larger models = easier to compress")
print()

# Check model files
files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
print(f"Model shards: {len(files)}")
for f in files:
    size = os.path.getsize(os.path.join(model_path, f)) / 1e9
    print(f"  {f}: {size:.1f} GB")

# Load config to detect architecture
import json
config_path = os.path.join(model_path, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        model_config = json.load(f)
    print(f"\nArchitecture:")
    print(f"  Hidden: {model_config.get('hidden_size')}")
    print(f"  Layers: {model_config.get('num_hidden_layers')}")
    print(f"  Heads: {model_config.get('num_attention_heads')}")
    print(f"  KV Heads: {model_config.get('num_key_value_heads')}")
    print(f"  Intermediate: {model_config.get('intermediate_size')}")
    print(f"  Vocab: {model_config.get('vocab_size')}")

    n_layers = model_config['num_hidden_layers']
    hidden = model_config['hidden_size']
    n_heads = model_config['num_attention_heads']
    n_kv = model_config.get('num_key_value_heads', n_heads)
    intermediate = model_config['intermediate_size']
    vocab = model_config['vocab_size']

    # Estimate genome size at different sd
    for sd in [64, 128, 256]:
        from ultracompress.genome_compressor import MicroTransformerLayer
        layer = MicroTransformerLayer(hidden, sd, 4)
        layer_params = sum(p.numel() for p in layer.parameters())
        total_genome = layer_params * n_layers
        total_orig = n_layers * (4 * hidden * hidden + 3 * hidden * intermediate)  # rough
        compression = total_orig / total_genome
        print(f"\n  sd={sd}: {total_genome*2/1e6:.1f} MB genome, {compression:.0f}x compression")
        print(f"    At this ratio: 10T genome = {10e12 / compression * 2 / 1e9:.1f} GB")
else:
    print("No config.json found")

print("\nTo run full compression, need to load model layer-by-layer")
print("(16GB model won't fit entirely in 64GB RAM as float32)")
print("This script validates the architecture. Full test needs streaming loader.")
