"""PARADIGM TOURNAMENT — Three fundamentally new compression approaches.

Not variations on genome compression. Three entirely different vehicles.
All tested head-to-head on Qwen3-0.6B with the same eval.

Paradigm 1: NeRF for Weights — implicit function encodes all weights
Paradigm 2: Procedural Generation — tiny code + generator → weights
Paradigm 3: Algebraic Compression — shared basis + coefficients + sparse

Winner gets scaled to 8B.
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
from ultracompress.paradigm_shift import (
    WeightNeRF, WeightNeRFCompressor, UnifiedCompressor,
    ProceduralCompressor, AlgebraicCompressor,
)

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

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

all_results = {}
pipeline_start = time.time()


def eval_model(forward_fn, n=100):
    """Standard eval: top-1 and top-10 token agreement vs teacher."""
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


def build_model_from_weights(weight_dict_override, config, device):
    """Build a MiniTransformer using reconstructed weights."""
    new_gd = dict(gd)  # Start with original (embed, norm, head)
    new_gd.update(weight_dict_override)
    model = MiniTransformer(config, device)
    model.load_weights(new_gd)
    return model


# Count original parameters
orig_layer_params = 0
for key, val in gd.items():
    if key.startswith('blk.') and 'weight' in key:
        orig_layer_params += val.numel()
orig_size_mb = orig_layer_params * 4 / 1e6  # FP32
print(f"Original model: {orig_layer_params:,} layer params ({orig_size_mb:.1f} MB FP32)")
print(f"Embed + head: {(gd['token_embd.weight'].numel() + gd['output.weight'].numel()):,} params (shared, not compressed)")


# ============================================================
# PARADIGM 1: NERF FOR WEIGHTS
# ============================================================
print(f"\n{'='*70}")
print("PARADIGM 1: NERF FOR WEIGHTS (Implicit Neural Representation)")
print(f"{'='*70}")
sys.stdout.flush()

t0 = time.time()

# Build training data
weight_dict, info_dict = UnifiedCompressor.build_nerf_training_data(gd, n_layers=28)

# Test multiple NeRF sizes
for hidden_dim, n_layers_nerf, name in [
    (256, 4, "NeRF-small"),
    (512, 6, "NeRF-medium"),
    (1024, 8, "NeRF-large"),
]:
    print(f"\n--- {name} (hidden={hidden_dim}, layers={n_layers_nerf}) ---")
    sys.stdout.flush()

    nerf = WeightNeRF(hidden_dim=hidden_dim, n_layers=n_layers_nerf, n_freqs=8, use_siren=True)
    nerf_params = sum(p.numel() for p in nerf.parameters())
    nerf_size_mb = nerf_params * 4 / 1e6
    compression = orig_layer_params / nerf_params
    print(f"  NeRF params: {nerf_params:,} ({nerf_size_mb:.1f} MB) = {compression:.0f}x compression")
    sys.stdout.flush()

    compressor = WeightNeRFCompressor(weight_dict, info_dict, device=device)
    nerf = compressor.train(nerf, n_steps=20000, batch_size=65536, lr=5e-4)

    # Reconstruct ALL weights and build model
    print("  Reconstructing weights...")
    sys.stdout.flush()
    matrix_stacks, matrix_shapes = UnifiedCompressor.extract_matrices(gd, n_layers=28)
    type_to_idx = {t: i for i, t in enumerate(sorted(matrix_stacks.keys()))}
    n_types = len(type_to_idx)

    reconstructed_gd = {}
    total_mae = 0
    n_matrices = 0
    with torch.no_grad():
        for mtype, stack in matrix_stacks.items():
            for li in range(stack.shape[0]):
                rows, cols = stack[li].shape
                recon = nerf.reconstruct_matrix(
                    li, type_to_idx[mtype], rows, cols, 28, n_types
                )
                # Denormalize
                recon = recon * compressor.value_std + compressor.value_mean
                key = f'blk.{li}.{mtype}.weight'
                if stack[li].shape[0] == 1:
                    reconstructed_gd[key] = recon.squeeze(0).to(device)
                else:
                    reconstructed_gd[key] = recon.to(device)
                total_mae += (recon.cpu() - stack[li]).abs().mean().item()
                n_matrices += 1

    avg_mae = total_mae / n_matrices
    print(f"  Avg MAE: {avg_mae:.6f}")

    # Build model and eval
    recon_model = build_model_from_weights(reconstructed_gd, config, device)
    def nerf_forward(tokens, _m=recon_model):
        return _m.forward(tokens, max_layers=28)

    t1, t10 = eval_model(nerf_forward)
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Size={nerf_size_mb:.1f}MB Compression={compression:.0f}x Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'size_mb': nerf_size_mb,
        'compression': compression, 'params': nerf_params,
        'avg_mae': avg_mae, 'time': elapsed, 'paradigm': 'nerf',
    }
    sys.stdout.flush()

    del nerf, recon_model, reconstructed_gd
    torch.cuda.empty_cache()


# ============================================================
# PARADIGM 2: PROCEDURAL WEIGHT GENERATION
# ============================================================
print(f"\n{'='*70}")
print("PARADIGM 2: PROCEDURAL WEIGHT GENERATION (HyperNetwork)")
print(f"{'='*70}")
sys.stdout.flush()

t0 = time.time()
matrix_stacks, matrix_shapes = UnifiedCompressor.extract_matrices(gd, n_layers=28)
type_list = sorted(matrix_stacks.keys())
type_to_idx = {t: i for i, t in enumerate(type_list)}

for code_dim, hidden_dim, name in [
    (32, 256, "Proc-small"),
    (64, 512, "Proc-medium"),
    (128, 1024, "Proc-large"),
]:
    print(f"\n--- {name} (code={code_dim}, hidden={hidden_dim}) ---")
    sys.stdout.flush()

    proc = ProceduralCompressor(
        n_layers=28, code_dim=code_dim, hidden_dim=hidden_dim,
        chunk_size=64, n_matrix_types=len(type_list),
    ).to(device)
    proc_params = proc.param_count()
    proc_size_mb = proc_params * 4 / 1e6
    compression = orig_layer_params / proc_params
    print(f"  Params: {proc_params:,} ({proc_size_mb:.1f} MB) = {compression:.0f}x compression")
    sys.stdout.flush()

    # Train: for each matrix, minimize reconstruction error
    opt = torch.optim.Adam(proc.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000)

    for step in range(10000):
        # Random layer, random matrix type
        li = torch.randint(0, 28, (1,)).item()
        ti = torch.randint(0, len(type_list), (1,)).item()
        mtype = type_list[ti]

        target = matrix_stacks[mtype][li].to(device)
        rows, cols = target.shape

        # Generate
        pred = proc.generate_weight(li, ti, rows, cols)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(proc.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 2000 == 0:
            print(f"    Step {step}: loss={loss.item():.6f}")
            sys.stdout.flush()

    # Reconstruct and eval
    print("  Reconstructing...")
    sys.stdout.flush()
    reconstructed_gd = {}
    total_mae = 0
    n_matrices = 0
    with torch.no_grad():
        for mtype in type_list:
            ti = type_to_idx[mtype]
            for li in range(28):
                target = matrix_stacks[mtype][li]
                rows, cols = target.shape
                recon = proc.generate_weight(li, ti, rows, cols).cpu()
                key = f'blk.{li}.{mtype}.weight'
                if target.shape[0] == 1:
                    reconstructed_gd[key] = recon.squeeze(0).to(device)
                else:
                    reconstructed_gd[key] = recon.to(device)
                total_mae += (recon - target).abs().mean().item()
                n_matrices += 1

    avg_mae = total_mae / n_matrices
    print(f"  Avg MAE: {avg_mae:.6f}")

    recon_model = build_model_from_weights(reconstructed_gd, config, device)
    def proc_forward(tokens, _m=recon_model):
        return _m.forward(tokens, max_layers=28)

    t1, t10 = eval_model(proc_forward)
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Size={proc_size_mb:.1f}MB Compression={compression:.0f}x Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'size_mb': proc_size_mb,
        'compression': compression, 'params': proc_params,
        'avg_mae': avg_mae, 'time': elapsed, 'paradigm': 'procedural',
    }
    sys.stdout.flush()

    del proc, recon_model, reconstructed_gd
    torch.cuda.empty_cache()


# ============================================================
# PARADIGM 3: ALGEBRAIC COMPRESSION
# ============================================================
print(f"\n{'='*70}")
print("PARADIGM 3: ALGEBRAIC COMPRESSION (Mathematical Structure)")
print(f"{'='*70}")
sys.stdout.flush()

t0 = time.time()

for n_basis, sparse_ratio, name in [
    (4, 0.001, "Alg-4basis"),
    (8, 0.005, "Alg-8basis"),
    (16, 0.01, "Alg-16basis"),
]:
    print(f"\n--- {name} (n_basis={n_basis}, sparse={sparse_ratio*100:.1f}%) ---")
    sys.stdout.flush()

    alg = AlgebraicCompressor(n_layers=28, n_basis=n_basis, sparse_ratio=sparse_ratio)

    # Fit each matrix type
    for mtype, stack in matrix_stacks.items():
        alg.fit_matrix_type(mtype, stack, stack.shape[1], stack.shape[2])

    alg = alg.to(device)
    alg_params = alg.total_params()
    alg_size_mb = alg_params * 4 / 1e6
    compression = orig_layer_params / alg_params
    print(f"  Total params: {alg_params:,} ({alg_size_mb:.1f} MB) = {compression:.0f}x compression")
    sys.stdout.flush()

    # Fine-tune with gradient descent to minimize reconstruction error
    print("  Fine-tuning algebraic decomposition...")
    opt = torch.optim.Adam(alg.parameters(), lr=1e-4)
    for step in range(5000):
        li = torch.randint(0, 28, (1,)).item()
        mtype = type_list[torch.randint(0, len(type_list), (1,)).item()]
        target = matrix_stacks[mtype][li].to(device)
        rows, cols = target.shape

        recon = alg.reconstruct_matrix(mtype, li, rows, cols)
        loss = F.mse_loss(recon, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(alg.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.6f}")
            sys.stdout.flush()

    # Reconstruct and eval
    print("  Reconstructing...")
    sys.stdout.flush()
    reconstructed_gd = {}
    total_mae = 0
    n_matrices = 0
    with torch.no_grad():
        for mtype in type_list:
            for li in range(28):
                target = matrix_stacks[mtype][li]
                rows, cols = target.shape
                recon = alg.reconstruct_matrix(mtype, li, rows, cols).cpu()
                key = f'blk.{li}.{mtype}.weight'
                if target.shape[0] == 1:
                    reconstructed_gd[key] = recon.squeeze(0).to(device)
                else:
                    reconstructed_gd[key] = recon.to(device)
                total_mae += (recon - target).abs().mean().item()
                n_matrices += 1

    avg_mae = total_mae / n_matrices
    print(f"  Avg MAE: {avg_mae:.6f}")

    recon_model = build_model_from_weights(reconstructed_gd, config, device)
    def alg_forward(tokens, _m=recon_model):
        return _m.forward(tokens, max_layers=28)

    t1, t10 = eval_model(alg_forward)
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Size={alg_size_mb:.1f}MB Compression={compression:.0f}x Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'size_mb': alg_size_mb,
        'compression': compression, 'params': alg_params,
        'avg_mae': avg_mae, 'time': elapsed, 'paradigm': 'algebraic',
    }
    sys.stdout.flush()

    del alg, recon_model, reconstructed_gd
    torch.cuda.empty_cache()


# ============================================================
# FINAL LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"PARADIGM TOURNAMENT RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Original model: {orig_size_mb:.0f} MB FP32")
print(f"Previous best (genome): 63% top-10 at 23.9 MB")
print()

# Group by paradigm
for paradigm in ['nerf', 'procedural', 'algebraic']:
    results = [(n, r) for n, r in all_results.items() if r.get('paradigm') == paradigm]
    if results:
        pname = {'nerf': 'NERF FOR WEIGHTS', 'procedural': 'PROCEDURAL GENERATION', 'algebraic': 'ALGEBRAIC COMPRESSION'}[paradigm]
        print(f"  {pname}:")
        for name, r in sorted(results, key=lambda x: x[1].get('top10', 0), reverse=True):
            print(f"    {name:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
                  f"Size={r['size_mb']:>6.1f}MB {r['compression']:>5.0f}x MAE={r['avg_mae']:.6f}")
        print()

# Overall champion
print("  OVERALL LEADERBOARD:")
sorted_all = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_all):
    medal = [">>>1st", "   2nd", "   3rd"][i] if i < 3 else f"   {i+1}th"
    print(f"  {medal}: {name:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
          f"Size={r['size_mb']:>6.1f}MB {r['compression']:>5.0f}x [{r['paradigm']}]")

with open('paradigm_tournament_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to paradigm_tournament_results.json")

# Analysis
print(f"\n{'='*70}")
print("PARADIGM ANALYSIS")
print(f"{'='*70}")

best_name, best = sorted_all[0]
worst_name, worst = sorted_all[-1]

print(f"\nChampion: {best_name} ({best['paradigm']})")
print(f"  Top1={best['top1']*100:.0f}% Top10={best['top10']*100:.0f}% at {best['size_mb']:.1f}MB ({best['compression']:.0f}x)")

if best['top10'] > 0.80:
    print("  >>> PARADIGM SHIFT ACHIEVED. 80%+ top-10. This changes everything.")
elif best['top10'] > 0.63:
    print("  >>> Beats genome approach (63%). This paradigm has more potential.")
elif best['top10'] > 0.50:
    print("  >>> Competitive with genome. Different trade-offs.")
else:
    print("  >>> Below genome baseline. But may scale differently with more params/training.")

# Which paradigm preserves structure best?
for paradigm in ['nerf', 'procedural', 'algebraic']:
    results = [(n, r) for n, r in all_results.items() if r.get('paradigm') == paradigm]
    if results:
        best_p = max(results, key=lambda x: x[1].get('top10', 0))
        print(f"\n  {paradigm.upper()} best: {best_p[0]} at {best_p[1]['top10']*100:.0f}% top-10")
        print(f"    Weight MAE: {best_p[1]['avg_mae']:.6f}")
        if best_p[1]['avg_mae'] < 0.01:
            print(f"    >>> Very accurate weight reconstruction. Quality limited by eval, not reconstruction.")
        elif best_p[1]['avg_mae'] < 0.05:
            print(f"    >>> Good reconstruction. Most weight structure preserved.")
        else:
            print(f"    >>> High reconstruction error. Needs larger model or better training.")

print(f"\n{'='*70}")
print("NEXT STEPS:")
if best['paradigm'] == 'algebraic':
    print("  Algebraic won — the structure IS algebraic. Push n_basis higher.")
    print("  Try: hierarchical basis (global + per-section), adaptive sparse.")
elif best['paradigm'] == 'nerf':
    print("  NeRF won — weights are a smooth function. Push hidden dim / training.")
    print("  Try: multi-resolution hash encoding (InstantNGP), per-type NeRFs.")
elif best['paradigm'] == 'procedural':
    print("  Procedural won — layer codes are powerful. Push code_dim + generator.")
    print("  Try: autoregressive generation, attention in generator.")
print(f"  COMBINE: Use {best['paradigm']} for main weights + genome for residual.")
print(f"{'='*70}")
