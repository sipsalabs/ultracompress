"""8B MODEL COMPRESSIBILITY ANALYSIS — Safe, read-only, memory-conscious.

Loads Qwen3-8B shard by shard, analyzes weight structure.
Key question: Does 8B have MORE redundancy than 0.6B?
If yes, compression should be much easier at scale.

Memory safe: loads one shard at a time, deletes after analysis.
Peak usage: ~4GB (one shard) + analysis overhead.
"""
import torch, sys, os, time, json, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = 'cpu'  # Analysis on CPU for safety — no GPU OOM risk
results = {}

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")

print("=" * 70)
print("QWEN3-8B COMPRESSIBILITY ANALYSIS")
print("Safe mode: CPU only, one shard at a time")
print("=" * 70)

# ============================================================
# Step 1: Discover model structure
# ============================================================
print("\n--- Step 1: Model Structure ---")

from safetensors import safe_open
shard_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
print(f"Found {len(shard_files)} shards")

# Map all tensor names to their shards
tensor_map = {}
total_params = 0
for sf in shard_files:
    path = os.path.join(MODEL_PATH, sf)
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            tensor_map[key] = {'shard': sf, 'shape': list(shape), 'params': np.prod(shape)}
            total_params += np.prod(shape)

print(f"Total tensors: {len(tensor_map)}")
print(f"Total params: {total_params:,} ({total_params*2/1e9:.1f} GB FP16)")

# Identify layer structure
layer_count = 0
for key in tensor_map:
    if 'model.layers.' in key:
        li = int(key.split('model.layers.')[1].split('.')[0])
        layer_count = max(layer_count, li + 1)
print(f"Layers: {layer_count}")

# Identify weight types and their shapes
weight_types = {}
for key in tensor_map:
    if 'model.layers.0.' in key:
        suffix = key.replace('model.layers.0.', '')
        weight_types[suffix] = tensor_map[key]['shape']
        print(f"  {suffix}: {tensor_map[key]['shape']} ({tensor_map[key]['params']:,} params)")

sys.stdout.flush()


# ============================================================
# Step 2: Cross-layer redundancy analysis
# ============================================================
print(f"\n--- Step 2: Cross-Layer Redundancy ---")
print("Loading weight matrices per type across all layers...")
sys.stdout.flush()

redundancy_results = {}

for wtype, expected_shape in weight_types.items():
    if len(expected_shape) < 2:
        continue  # Skip 1D (norms)
    if expected_shape[0] * expected_shape[1] > 50_000_000:
        print(f"  Skipping {wtype} (too large: {expected_shape})")
        continue

    print(f"\n  Analyzing: {wtype} {expected_shape}")
    sys.stdout.flush()

    # Load this weight type from all layers
    layers_data = []
    for li in range(layer_count):
        key = f'model.layers.{li}.{wtype}'
        if key not in tensor_map:
            break
        shard_file = tensor_map[key]['shard']
        path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor(key).float()
            layers_data.append(w)

    if len(layers_data) < 2:
        continue

    stack = torch.stack(layers_data)  # (n_layers, rows, cols)
    n_layers, rows, cols = stack.shape

    # Flatten for cross-layer analysis
    flat = stack.reshape(n_layers, -1)  # (n_layers, rows*cols)

    # A. SVD spectrum across layers
    try:
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
        S_norm = S / S.sum()
        cumulative = torch.cumsum(S_norm, 0)

        # How many components for 90%, 95%, 99% of energy?
        n_90 = (cumulative < 0.90).sum().item() + 1
        n_95 = (cumulative < 0.95).sum().item() + 1
        n_99 = (cumulative < 0.99).sum().item() + 1

        print(f"    SVD: {n_90}/{n_layers} components for 90%, {n_95} for 95%, {n_99} for 99%")

        # Potential compression from SVD alone
        # Original: n_layers * rows * cols
        # Compressed: k * (n_layers + rows * cols) approximately
        for target, n_comp in [('90%', n_90), ('95%', n_95), ('99%', n_99)]:
            compressed_size = n_comp * (n_layers + rows * cols)
            original_size = n_layers * rows * cols
            ratio = original_size / compressed_size
            print(f"    {target} SVD: {ratio:.1f}x compression ({n_comp} components)")

    except Exception as e:
        print(f"    SVD failed: {e}")
        n_90, n_95, n_99 = -1, -1, -1

    # B. Adjacent layer cosine similarity
    cos_sims = []
    for i in range(n_layers - 1):
        cos = torch.nn.functional.cosine_similarity(flat[i:i+1], flat[i+1:i+2]).item()
        cos_sims.append(cos)
    avg_cos = sum(cos_sims) / len(cos_sims)
    min_cos = min(cos_sims)
    max_cos = max(cos_sims)
    print(f"    Adjacent cosine sim: avg={avg_cos:.4f} min={min_cos:.4f} max={max_cos:.4f}")

    # C. Layer-to-layer delta magnitude
    deltas = flat[1:] - flat[:-1]
    delta_norm = deltas.norm(dim=1).mean().item()
    layer_norm = flat.norm(dim=1).mean().item()
    relative_delta = delta_norm / (layer_norm + 1e-8)
    print(f"    Delta/Layer ratio: {relative_delta:.4f} (smaller = more redundant)")

    # D. Value statistics
    print(f"    Values: mean={stack.mean():.6f} std={stack.std():.6f} "
          f"min={stack.min():.4f} max={stack.max():.4f}")

    # E. Sparsity (near-zero values)
    threshold = stack.std() * 0.01
    sparsity = (stack.abs() < threshold).float().mean().item()
    print(f"    Sparsity (<1% of std): {sparsity*100:.1f}%")

    redundancy_results[wtype] = {
        'shape': expected_shape,
        'n_layers': n_layers,
        'svd_90': n_90, 'svd_95': n_95, 'svd_99': n_99,
        'avg_cosine': avg_cos, 'min_cosine': min_cos,
        'relative_delta': relative_delta,
        'mean': stack.mean().item(), 'std': stack.std().item(),
        'sparsity': sparsity,
    }

    # Cleanup
    del stack, flat, layers_data
    gc.collect()
    sys.stdout.flush()


# ============================================================
# Step 3: Norm analysis
# ============================================================
print(f"\n--- Step 3: Norm Weight Analysis ---")

norm_results = {}
for wtype, expected_shape in weight_types.items():
    if len(expected_shape) != 1:
        continue  # Only 1D norms

    layers_data = []
    for li in range(layer_count):
        key = f'model.layers.{li}.{wtype}'
        if key not in tensor_map:
            break
        shard_file = tensor_map[key]['shard']
        path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor(key).float()
            layers_data.append(w)

    if len(layers_data) < 2:
        continue

    stack = torch.stack(layers_data)
    n_layers = stack.shape[0]

    # Cross-layer variance for norms
    layer_means = stack.mean(dim=1)
    layer_stds = stack.std(dim=1)
    cross_layer_var = layer_means.std().item()

    # Adjacent similarity
    cos_sims = []
    for i in range(n_layers - 1):
        cos = torch.nn.functional.cosine_similarity(stack[i:i+1], stack[i+1:i+2]).item()
        cos_sims.append(cos)

    avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0

    print(f"  {wtype}: cross_layer_var={cross_layer_var:.4f} avg_cosine={avg_cos:.4f} "
          f"values=[{stack.min():.3f}, {stack.max():.3f}]")

    norm_results[wtype] = {
        'cross_layer_var': cross_layer_var,
        'avg_cosine': avg_cos,
        'min': stack.min().item(), 'max': stack.max().item(),
    }

    del stack, layers_data
    gc.collect()
    sys.stdout.flush()


# ============================================================
# Step 4: Compare with 0.6B
# ============================================================
print(f"\n--- Step 4: Comparison with 0.6B ---")
print("Loading 0.6B for comparison...")

try:
    wd_06 = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
    hf_to_suffix = {
        'self_attn.q_proj.weight': 'self_attn.q_proj.weight',
        'self_attn.k_proj.weight': 'self_attn.k_proj.weight',
        'mlp.gate_proj.weight': 'mlp.gate_proj.weight',
    }

    for suffix_06 in ['self_attn.q_proj.weight', 'mlp.gate_proj.weight']:
        layers_06 = []
        for li in range(28):
            key = f'model.layers.{li}.{suffix_06}'
            if key in wd_06:
                layers_06.append(wd_06[key].float())
        if len(layers_06) < 2:
            continue
        stack_06 = torch.stack(layers_06)
        flat_06 = stack_06.reshape(len(layers_06), -1)

        U, S, Vh = torch.linalg.svd(flat_06, full_matrices=False)
        S_norm = S / S.sum()
        cum = torch.cumsum(S_norm, 0)
        n90 = (cum < 0.90).sum().item() + 1
        n95 = (cum < 0.95).sum().item() + 1
        n99 = (cum < 0.99).sum().item() + 1

        cos_sims = [torch.nn.functional.cosine_similarity(flat_06[i:i+1], flat_06[i+1:i+2]).item() for i in range(len(layers_06)-1)]

        print(f"  0.6B {suffix_06}:")
        print(f"    SVD: {n90}/{len(layers_06)} for 90%, {n95} for 95%, {n99} for 99%")
        print(f"    Avg cosine: {sum(cos_sims)/len(cos_sims):.4f}")

    del wd_06
    gc.collect()
except Exception as e:
    print(f"  Could not load 0.6B for comparison: {e}")

sys.stdout.flush()


# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*70}")
print("COMPRESSIBILITY SUMMARY")
print(f"{'='*70}")

print(f"\nQwen3-8B: {layer_count} layers, {total_params:,} params")
print(f"\nWeight Matrix Redundancy:")
for wtype, r in sorted(redundancy_results.items(), key=lambda x: x[1].get('avg_cosine', 0), reverse=True):
    print(f"  {wtype:>35}: cosine={r['avg_cosine']:.4f} SVD90={r['svd_90']}/{r['n_layers']} "
          f"delta_ratio={r['relative_delta']:.4f} "
          f"99%_compression={r['n_layers']*np.prod(r['shape']) / (r['svd_99']*(r['n_layers']+np.prod(r['shape']))):.1f}x")

print(f"\nNorm Weights:")
for wtype, r in norm_results.items():
    print(f"  {wtype:>35}: cross_var={r['cross_layer_var']:.4f} cosine={r['avg_cosine']:.4f}")

# Overall compression estimate
avg_svd90_ratio = []
for r in redundancy_results.values():
    if r['svd_90'] > 0:
        original = r['n_layers'] * np.prod(r['shape'])
        compressed = r['svd_90'] * (r['n_layers'] + np.prod(r['shape']))
        avg_svd90_ratio.append(original / compressed)

if avg_svd90_ratio:
    avg_ratio = sum(avg_svd90_ratio) / len(avg_svd90_ratio)
    print(f"\nEstimated compression at 90% SVD energy: {avg_ratio:.1f}x average")
    print(f"  8B at {avg_ratio:.0f}x = {total_params*2/1e9 / avg_ratio:.1f} GB")
    print(f"  If 8B redundancy > 0.6B: compression scales favorably with model size")

# Save
all_results = {
    'model': 'Qwen3-8B',
    'total_params': int(total_params),
    'n_layers': layer_count,
    'weight_redundancy': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                              for kk, vv in v.items()} for k, v in redundancy_results.items()},
    'norm_analysis': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in norm_results.items()},
}
with open('8b_analysis_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to 8b_analysis_results.json")
print(f"{'='*70}")
