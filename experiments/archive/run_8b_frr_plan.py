"""8B FRR SCALING TEST — Plan and execute FRR distillation from Qwen3-8B.

Memory budget: 32GB GPU
- Qwen3-8B FP16: ~16GB (loaded shard by shard via streaming)
- FRR model (1024→4096 hidden): ~40-160MB depending on config
- Training overhead: ~5-8GB
- Total: ~24GB — fits with margin

Strategy: Stream teacher layer by layer (never all in memory at once).
For each training step:
1. Load a batch of random tokens
2. Run teacher forward (streaming — load/process/discard each layer)
3. Run FRR forward (all in memory — tiny)
4. Compute KL loss on all positions
5. Backprop through FRR only

The streaming teacher forward is the key innovation for scaling.
"""
import torch, sys, os, time, json, math, gc
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if we can fit in memory
print("8B FRR SCALING TEST — Memory Planning")
print("=" * 60)

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Qwen3-8B not found at {MODEL_PATH}")
    sys.exit(1)

from safetensors import safe_open

# Discover structure
shard_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
print(f"Found {len(shard_files)} shards")

# Get architecture info
with safe_open(os.path.join(MODEL_PATH, shard_files[0]), framework="pt") as f:
    keys = list(f.keys())
    # Find hidden dim from embed
    for k in keys:
        if 'embed_tokens' in k:
            shape = f.get_tensor(k).shape
            vocab_size, hidden_dim = shape
            print(f"Vocab: {vocab_size}, Hidden: {hidden_dim}")
            break

# Count layers
n_layers = 0
for k in keys:
    if 'model.layers.' in k:
        li = int(k.split('model.layers.')[1].split('.')[0])
        n_layers = max(n_layers, li + 1)

# Check all shards for max layer
for sf in shard_files[1:]:
    with safe_open(os.path.join(MODEL_PATH, sf), framework="pt") as f:
        for k in f.keys():
            if 'model.layers.' in k:
                li = int(k.split('model.layers.')[1].split('.')[0])
                n_layers = max(n_layers, li + 1)

print(f"Layers: {n_layers}")
print(f"Hidden dim: {hidden_dim}")

# Memory estimate
shard_sizes = [os.path.getsize(os.path.join(MODEL_PATH, sf)) / 1e9 for sf in shard_files]
total_model_gb = sum(shard_sizes)
print(f"Total model size: {total_model_gb:.1f} GB")
print(f"Largest shard: {max(shard_sizes):.1f} GB")

# FRR model size for 8B (hidden_dim=4096)
for n_heads, ff_mult in [(32, 2), (16, 2)]:
    # One block: QKV (3*D*D) + O (D*D) + gate+up (D*D*ff_mult*2) + down (D*ff_mult*D) + norms
    block_params = 3*hidden_dim*hidden_dim + hidden_dim*hidden_dim + 2*hidden_dim*hidden_dim*ff_mult + hidden_dim*ff_mult*hidden_dim + 2*hidden_dim
    # Scale modulation
    mod_params = n_layers * 2 * hidden_dim + n_layers * (n_layers // 4)  # approx
    total_frr = block_params + mod_params
    frr_gb = total_frr * 4 / 1e9  # FP32 for training
    print(f"\nFRR config (heads={n_heads}, ff={ff_mult}):")
    print(f"  Block params: {block_params:,} ({block_params*4/1e6:.0f} MB FP32)")
    print(f"  + Modulation: {mod_params:,}")
    print(f"  Total FRR: {total_frr:,} ({frr_gb:.2f} GB)")
    print(f"  Compression: {total_model_gb * 1e9 / 2 / total_frr:.0f}x vs FP16 model")

# Memory budget
print(f"\n{'='*60}")
print("MEMORY BUDGET (32GB GPU)")
print(f"{'='*60}")
embed_size_gb = vocab_size * hidden_dim * 2 / 1e9  # FP16
print(f"  Embeddings (FP16): {embed_size_gb:.1f} GB")
print(f"  LM head (FP16): {embed_size_gb:.1f} GB (tied)")
print(f"  One shard (for streaming): {max(shard_sizes):.1f} GB")
print(f"  FRR model (FP32 training): ~0.5 GB")
print(f"  Optimizer states (Adam): ~1.0 GB")
print(f"  Activations (batch=4, seq=32): ~2 GB")
print(f"  Teacher activations cache: ~1 GB")

total_est = embed_size_gb * 2 + max(shard_sizes) + 0.5 + 1.0 + 2.0 + 1.0
print(f"  TOTAL ESTIMATED: {total_est:.1f} GB")
print(f"  GPU AVAILABLE: 32.0 GB")
print(f"  HEADROOM: {32.0 - total_est:.1f} GB")

if total_est < 28:
    print(f"\n  >>> FITS! Can run 8B FRR distillation safely.")
    print(f"  >>> Use batch_size=4, seq_len=32 to stay safe.")
    print(f"  >>> Stream teacher one shard at a time.")
else:
    print(f"\n  >>> TIGHT! Reduce batch_size to 2 or use gradient accumulation.")

print(f"\nTo run: set CUDA_VISIBLE_DEVICES to 8B GPU and launch")
print(f"Next step: implement streaming teacher forward for FRR training")
print(f"{'='*60}")
