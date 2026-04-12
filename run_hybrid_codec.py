"""HYBRID CODEC — DCT compression + genome behavioral correction.

The invention: frequency-domain weight compression + behavioral distillation.
Nobody has combined these two paradigms before.

Tests multiple DCT quality levels with correction training.
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.hybrid_codec import HybridCompressor, FastDCT

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
# Fix teacher GPU transfer
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

all_results = {}
pipeline_start = time.time()

# Separate weights from norms
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

orig_params = sum(v.numel() for v in matrix_stacks.values())
print(f"Weights: {orig_params:,} ({orig_params*4/1e6:.1f} MB)")
print(f"Norms: {sum(v.numel() for v in norm_weights.values()):,} (verbatim)")
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


# ============================================================
# TEST EACH QUALITY LEVEL: DCT only, then DCT + correction
# ============================================================

for quality, correction_rank, correction_steps, name in [
    (80, 16, 10000, "Hybrid-Q80-r16"),   # Conservative DCT + light correction
    (50, 32, 10000, "Hybrid-Q50-r32"),   # Moderate DCT + medium correction
    (30, 32, 15000, "Hybrid-Q30-r32"),   # Aggressive DCT + more correction
    (10, 64, 20000, "Hybrid-Q10-r64"),   # Maximum DCT + heavy correction
]:
    print(f"\n{'='*70}")
    print(f"HYBRID: {name} (DCT Q{quality} + correction rank={correction_rank})")
    print(f"{'='*70}")
    sys.stdout.flush()
    t0 = time.time()

    compressor = HybridCompressor(quality=quality, correction_rank=correction_rank, device=device)

    # Stage 1: DCT compress
    print("Stage 1: DCT compression...")
    compressed, norms, dct_info = compressor.compress_weights(matrix_stacks, norm_weights)

    # Reconstruct DCT-only model
    print("  Reconstructing DCT model...")
    dct_model = compressor.reconstruct_model(compressed, norm_weights, gd, config)

    # Eval DCT-only
    t1_dct, t10_dct = eval_model(lambda t, _m=dct_model: _m.forward(t, max_layers=28))
    print(f"  DCT-only: Top1={t1_dct*100:.0f}% Top10={t10_dct*100:.0f}% at {dct_info['size_mb']:.1f}MB ({dct_info['ratio']:.1f}x)")
    sys.stdout.flush()

    # Stage 2: Train behavioral correction
    print(f"\nStage 2: Training correction ({correction_steps} steps)...")
    corrections = compressor.train_correction(
        dct_model, teacher, config, embed, norm_w, lm_head_w,
        n_layers=28, correction_rank=correction_rank, n_steps=correction_steps, lr=0.001,
    )

    # Eval with correction
    positions = torch.arange(16, device=device)

    def hybrid_forward(tokens, _dct=dct_model, _corr=corrections):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = F.embedding(tokens, embed).float()
        for li in range(28):
            x_dct = _dct.layers[li](x, pos)
            x = _corr[li](x_dct)
        var = x.float().pow(2).mean(-1, keepdim=True)
        xn = x.float() * torch.rsqrt(var + 1e-6) * norm_w
        return F.linear(xn, lm_head_w)

    t1_hybrid, t10_hybrid = eval_model(hybrid_forward)

    correction_params = sum(p.numel() for p in corrections.parameters())
    correction_mb = correction_params * 2 / 1e6  # FP16
    total_mb = dct_info['size_mb'] + correction_mb
    total_ratio = orig_params * 4 / 1e6 / total_mb

    elapsed = time.time() - t0
    print(f"\n  RESULT {name}:")
    print(f"    DCT-only:  Top1={t1_dct*100:.0f}% Top10={t10_dct*100:.0f}% at {dct_info['size_mb']:.1f}MB ({dct_info['ratio']:.1f}x)")
    print(f"    + Correct: Top1={t1_hybrid*100:.0f}% Top10={t10_hybrid*100:.0f}% at {total_mb:.1f}MB ({total_ratio:.1f}x)")
    print(f"    Improvement: Top10 +{(t10_hybrid-t10_dct)*100:.0f}% from correction")
    print(f"    Time: {elapsed:.0f}s")
    sys.stdout.flush()

    all_results[name] = {
        'dct_top1': t1_dct, 'dct_top10': t10_dct, 'dct_size_mb': dct_info['size_mb'], 'dct_ratio': dct_info['ratio'],
        'hybrid_top1': t1_hybrid, 'hybrid_top10': t10_hybrid,
        'total_size_mb': total_mb, 'total_ratio': total_ratio,
        'correction_mb': correction_mb, 'correction_params': correction_params,
        'improvement': t10_hybrid - t10_dct,
        'time': elapsed,
    }

    del dct_model, corrections
    torch.cuda.empty_cache()


# ============================================================
# LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"HYBRID CODEC RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Previous bests: Genome=63% top-10 (23.9MB), INT4=~60% top-10")
print()

print("  DCT-ONLY (no training):")
for name, r in sorted(all_results.items(), key=lambda x: x[1]['dct_top10'], reverse=True):
    print(f"    {name:<25} Top10={r['dct_top10']*100:>4.0f}% at {r['dct_size_mb']:>6.1f}MB ({r['dct_ratio']:>5.1f}x)")

print()
print("  HYBRID (DCT + correction):")
for name, r in sorted(all_results.items(), key=lambda x: x[1]['hybrid_top10'], reverse=True):
    print(f"    {name:<25} Top10={r['hybrid_top10']*100:>4.0f}% at {r['total_size_mb']:>6.1f}MB ({r['total_ratio']:>5.1f}x) [+{r['improvement']*100:.0f}% from correction]")

with open('hybrid_codec_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

best_name = max(all_results, key=lambda k: all_results[k]['hybrid_top10'])
best = all_results[best_name]
print(f"\n{'='*70}")
print(f"CHAMPION: {best_name}")
print(f"  Top10={best['hybrid_top10']*100:.0f}% at {best['total_size_mb']:.1f}MB ({best['total_ratio']:.1f}x)")

if best['hybrid_top10'] >= 0.80:
    print("  >>> 80%+ TOP-10. THE HYBRID WORKS. Scale to 8B NOW.")
elif best['hybrid_top10'] >= 0.65:
    print("  >>> Beats genome (63%). Hybrid approach validated. Push further.")
else:
    print("  >>> Below genome. DCT base quality matters more than correction.")
    print("  >>> Try: higher DCT quality, more correction rank, longer training.")

# Scaling projection
print(f"\n  If this scales to 10T model:")
print(f"    DCT alone: ~{10e12 * 2 / (best['dct_ratio'] * 1e9):.0f} GB")
print(f"    + Correction: ~{10e12 * 2 / (best['dct_ratio'] * 1e9) + 5:.0f} GB (correction is constant-ish)")
print(f"{'='*70}")
