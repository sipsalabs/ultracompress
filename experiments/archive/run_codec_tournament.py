"""CODEC TOURNAMENT - 4 pure compression approaches, no retraining.

1. Algebraic V2 (SVD + verbatim norms + low-rank basis)
2. Neural Weight Codec (JPEG for weights)
3. WeightDNA (keyframe + delta encoding)
4. Stacked Pipeline (quantize + SVD + sparse)

All take existing weights in, compressed representation out.
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.codec import AlgebraicV2, NeuralWeightCodec, WeightDNA, StackedPipeline

device = 'cuda'
print("Loading Qwen3-0.6B...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)

hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight', 'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight', 'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight'}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(28):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

config = ModelConfig(n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
                     intermediate_size=3072, vocab_size=151936, head_dim=128)
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)

all_results = {}
pipeline_start = time.time()

# Separate weight matrices from norms
NORM_TYPES = {'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm'}
WEIGHT_TYPES = {'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'}

matrix_stacks = {}
norm_weights = {}

for li in range(28):
    for suffix in WEIGHT_TYPES:
        key = f'blk.{li}.{suffix}.weight'
        if key in gd:
            if suffix not in matrix_stacks:
                matrix_stacks[suffix] = []
            matrix_stacks[suffix].append(gd[key].float())
    for suffix in NORM_TYPES:
        key = f'blk.{li}.{suffix}.weight'
        if key in gd:
            norm_weights[f'{suffix}_layer{li}'] = gd[key].float()

for k in matrix_stacks:
    matrix_stacks[k] = torch.stack(matrix_stacks[k])

matrix_shapes = {k: (v.shape[1], v.shape[2]) for k, v in matrix_stacks.items()}

orig_weight_params = sum(v.numel() for v in matrix_stacks.values())
orig_norm_params = sum(v.numel() for v in norm_weights.values())
print(f"Weight matrices: {orig_weight_params:,} params ({orig_weight_params*4/1e6:.1f} MB)")
print(f"Norm weights: {orig_norm_params:,} params ({orig_norm_params*4/1e3:.1f} KB) -- stored verbatim in ALL approaches")
print()


def eval_model(forward_fn, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = forward_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1/n, sum(t10s)/len(t10s)


def rebuild_and_eval(reconstructed_weights, norm_weights_dict, name):
    """Build MiniTransformer from reconstructed weights and eval."""
    new_gd = dict(gd)  # Start with embed + head + output_norm

    # Add reconstructed weight matrices
    for key, tensor in reconstructed_weights.items():
        # Parse: {type}_layer{N}
        parts = key.rsplit('_layer', 1)
        if len(parts) == 2:
            mtype, li = parts[0], int(parts[1])
            gd_key = f'blk.{li}.{mtype}.weight'
            new_gd[gd_key] = tensor.to(device)

    # Add verbatim norms
    for key, tensor in norm_weights_dict.items():
        parts = key.rsplit('_layer', 1)
        if len(parts) == 2:
            mtype, li = parts[0], int(parts[1])
            gd_key = f'blk.{li}.{mtype}.weight'
            new_gd[gd_key] = tensor.to(device)

    model = MiniTransformer(config, device)
    model.load_weights(new_gd)

    def forward_fn(tokens, _m=model):
        return _m.forward(tokens, max_layers=28)

    t1, t10 = eval_model(forward_fn)
    return t1, t10


# ============================================================
# APPROACH 1: ALGEBRAIC V2
# ============================================================
print("=" * 70)
print("APPROACH 1: ALGEBRAIC V2 (SVD + verbatim norms + low-rank basis)")
print("=" * 70)
sys.stdout.flush()

for n_basis, basis_rank, sparse_ratio, name in [
    (4, 16, 0.005, "AlgV2-4b-r16"),
    (8, 32, 0.01, "AlgV2-8b-r32"),
    (16, 64, 0.02, "AlgV2-16b-r64"),
]:
    print(f"\n--- {name} ---")
    t0 = time.time()
    alg = AlgebraicV2(n_basis=n_basis, basis_rank=basis_rank, sparse_ratio=sparse_ratio)
    result = alg.compress(matrix_stacks, norm_weights)
    recon = alg.decompress(result, matrix_shapes)

    t1, t10 = rebuild_and_eval(recon, norm_weights, name)
    elapsed = time.time() - t0
    comp = result.metadata['total_original'] / result.metadata['total_compressed']
    size_mb = result.metadata['total_compressed'] * 4 / 1e6

    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB {comp:.1f}x Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'compression': comp, 'time': elapsed, 'approach': 'algebraic_v2'}
    sys.stdout.flush()


# ============================================================
# APPROACH 2: NEURAL WEIGHT CODEC
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 2: NEURAL WEIGHT CODEC (JPEG for weights)")
print(f"{'='*70}")
sys.stdout.flush()

for quality, block_size, name in [
    (10, 16, "Codec-Q10"),
    (30, 16, "Codec-Q30"),
    (50, 16, "Codec-Q50"),
    (80, 16, "Codec-Q80"),
]:
    print(f"\n--- {name} ---")
    t0 = time.time()
    codec = NeuralWeightCodec(block_size=block_size, quality=quality)
    compressed, norms = codec.compress_model(matrix_stacks, norm_weights)
    recon, norms = codec.decompress_model(compressed, norms)

    t1, t10 = rebuild_and_eval(recon, norms, name)
    elapsed = time.time() - t0

    total_nonzero = sum(r[1]['n_nonzero'] for layers in compressed.values() for r in layers)
    size_mb = (total_nonzero + orig_norm_params) * 4 / 1e6
    comp = orig_weight_params / total_nonzero

    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB {comp:.1f}x Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'compression': comp, 'time': elapsed, 'approach': 'codec'}
    sys.stdout.flush()


# ============================================================
# APPROACH 3: WEIGHT DNA
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 3: WEIGHT DNA (keyframe + delta encoding)")
print(f"{'='*70}")
sys.stdout.flush()

for delta_rank, name in [
    (8, "DNA-r8"),
    (16, "DNA-r16"),
    (32, "DNA-r32"),
    (64, "DNA-r64"),
]:
    print(f"\n--- {name} ---")
    t0 = time.time()
    dna = WeightDNA(delta_rank=delta_rank)
    compressed, norms = dna.compress(matrix_stacks, norm_weights)
    recon, norms = dna.decompress(compressed, norms)

    t1, t10 = rebuild_and_eval(recon, norms, name)
    elapsed = time.time() - t0

    total_size = sum(r['total_size'] for r in compressed.values()) + orig_norm_params
    size_mb = total_size * 4 / 1e6
    comp = orig_weight_params / (total_size - orig_norm_params)

    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB {comp:.1f}x Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'compression': comp, 'time': elapsed, 'approach': 'dna'}
    sys.stdout.flush()


# ============================================================
# APPROACH 4: STACKED PIPELINE
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 4: STACKED PIPELINE (Quantize + SVD + Sparse)")
print(f"{'='*70}")
sys.stdout.flush()

for quant_bits, svd_rank, sparse_ratio, name in [
    (4, 4, 0.005, "Stack-Q4-r4"),
    (4, 8, 0.01, "Stack-Q4-r8"),
    (3, 8, 0.01, "Stack-Q3-r8"),
    (2, 8, 0.02, "Stack-Q2-r8"),
]:
    print(f"\n--- {name} ---")
    t0 = time.time()
    pipe = StackedPipeline(quant_bits=quant_bits, svd_rank=svd_rank, sparse_ratio=sparse_ratio)
    compressed, norms = pipe.compress(matrix_stacks, norm_weights)
    recon, norms = pipe.decompress(compressed, norms)

    t1, t10 = rebuild_and_eval(recon, norms, name)
    elapsed = time.time() - t0

    # Estimate size
    total_size = 0
    for mtype, data in compressed.items():
        n, r, c = data['shape']
        total_size += data['svd_U'].numel() + data['svd_Vh'].numel()
        total_size += data['sparse_idx'].numel() + data['sparse_val'].numel()
    total_size += orig_norm_params
    size_mb = total_size * 4 / 1e6
    comp = orig_weight_params / (total_size - orig_norm_params)

    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Size={size_mb:.1f}MB {comp:.1f}x Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'size_mb': size_mb, 'compression': comp, 'time': elapsed, 'approach': 'stacked'}
    sys.stdout.flush()


# ============================================================
# FINAL LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"CODEC TOURNAMENT RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Previous bests: Genome=63% top-10 (23.9MB), INT4=~60% top-10")
print()

for approach in ['algebraic_v2', 'codec', 'dna', 'stacked']:
    results = [(n, r) for n, r in all_results.items() if r.get('approach') == approach]
    if results:
        aname = {'algebraic_v2': 'ALGEBRAIC V2', 'codec': 'WEIGHT CODEC', 'dna': 'WEIGHT DNA', 'stacked': 'STACKED PIPELINE'}[approach]
        print(f"  {aname}:")
        for name, r in sorted(results, key=lambda x: x[1]['top10'], reverse=True):
            print(f"    {name:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% Size={r['size_mb']:>7.1f}MB {r['compression']:>6.1f}x")
        print()

print("  OVERALL CHAMPION:")
sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
for i, (name, r) in enumerate(sorted_all[:5]):
    medal = [">>>1st", "   2nd", "   3rd", "   4th", "   5th"][i]
    print(f"  {medal}: {name:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% {r['size_mb']:>7.1f}MB {r['compression']:>6.1f}x [{r['approach']}]")

with open('codec_tournament_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

best_name, best = sorted_all[0]
print(f"\nWinner: {best_name} at {best['top10']*100:.0f}% top-10, {best['size_mb']:.1f}MB")
print(f"{'='*70}")
