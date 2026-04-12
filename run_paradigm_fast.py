"""PARADIGM FAST — Skip NeRF (dead), run Algebraic + Procedural only.

NeRF results: 0% top-1, 0-1% top-10 at all sizes. Dead approach.
Reason: Single MLP can't encode 440M heterogeneous weight values.

Algebraic should dominate — starts from SVD (mathematically optimal).
Procedural is the wild card — layer codes + generator could be powerful.
"""
import torch, sys, os, time, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer, TransformerLayer
from ultracompress.paradigm_shift import (
    ProceduralCompressor, AlgebraicCompressor, UnifiedCompressor,
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

orig_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
orig_size_mb = orig_layer_params * 4 / 1e6
print(f"Original: {orig_layer_params:,} layer params ({orig_size_mb:.1f} MB FP32)")


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


# Extract organized matrices
matrix_stacks, matrix_shapes = UnifiedCompressor.extract_matrices(gd, n_layers=28)
type_list = sorted(matrix_stacks.keys())
type_to_idx = {t: i for i, t in enumerate(type_list)}


# ============================================================
# PARADIGM 3: ALGEBRAIC COMPRESSION (should be the star)
# ============================================================
print(f"\n{'='*70}")
print("PARADIGM 3: ALGEBRAIC COMPRESSION (SVD basis + coefficients + sparse)")
print(f"{'='*70}")
sys.stdout.flush()

for n_basis, sparse_ratio, name in [
    (4, 0.001, "Alg-4b-0.1%sp"),
    (8, 0.005, "Alg-8b-0.5%sp"),
    (16, 0.01, "Alg-16b-1%sp"),
    (24, 0.02, "Alg-24b-2%sp"),  # Extra: push quality with more basis
]:
    print(f"\n--- {name} ---")
    sys.stdout.flush()
    t0 = time.time()

    alg = AlgebraicCompressor(n_layers=28, n_basis=n_basis, sparse_ratio=sparse_ratio)

    for mtype, stack in matrix_stacks.items():
        alg.fit_matrix_type(mtype, stack, stack.shape[1], stack.shape[2])

    alg = alg.to(device)
    alg_params = alg.total_params()
    alg_size_mb = alg_params * 4 / 1e6
    compression = orig_layer_params / alg_params
    print(f"  Total: {alg_params:,} ({alg_size_mb:.1f} MB) = {compression:.1f}x compression")
    sys.stdout.flush()

    # Fine-tune
    print("  Fine-tuning...")
    opt = torch.optim.Adam(alg.parameters(), lr=1e-4)
    for step in range(5000):
        li = torch.randint(0, 28, (1,)).item()
        mtype = type_list[torch.randint(0, len(type_list), (1,)).item()]
        target = matrix_stacks[mtype][li].to(device)
        rows, cols = target.shape
        recon = alg.reconstruct_matrix(mtype, li, rows, cols)
        loss = F.mse_loss(recon, target)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(alg.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"    Step {step}: loss={loss.item():.8f}")
            sys.stdout.flush()

    # Reconstruct all weights
    print("  Reconstructing...")
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
    print(f"  Avg MAE: {avg_mae:.8f}")

    # Build model and eval
    new_gd = dict(gd)
    new_gd.update(reconstructed_gd)
    recon_model = MiniTransformer(config, device)
    recon_model.load_weights(new_gd)

    t1, t10 = eval_model(lambda tokens, _m=recon_model: _m.forward(tokens, max_layers=28))
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Size={alg_size_mb:.1f}MB {compression:.1f}x MAE={avg_mae:.8f} Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'size_mb': alg_size_mb,
        'compression': compression, 'params': alg_params,
        'avg_mae': avg_mae, 'time': elapsed, 'paradigm': 'algebraic',
    }
    sys.stdout.flush()
    del alg, recon_model, reconstructed_gd; torch.cuda.empty_cache()


# ============================================================
# PARADIGM 2: PROCEDURAL WEIGHT GENERATION
# ============================================================
print(f"\n{'='*70}")
print("PARADIGM 2: PROCEDURAL WEIGHT GENERATION (code + generator -> weights)")
print(f"{'='*70}")
sys.stdout.flush()

for code_dim, hidden_dim, n_steps, name in [
    (64, 512, 15000, "Proc-medium"),
    (128, 1024, 15000, "Proc-large"),
]:
    print(f"\n--- {name} (code={code_dim}, hidden={hidden_dim}) ---")
    sys.stdout.flush()
    t0 = time.time()

    proc = ProceduralCompressor(
        n_layers=28, code_dim=code_dim, hidden_dim=hidden_dim,
        chunk_size=64, n_matrix_types=len(type_list),
    ).to(device)
    proc_params = proc.param_count()
    proc_size_mb = proc_params * 4 / 1e6
    compression = orig_layer_params / proc_params
    print(f"  Params: {proc_params:,} ({proc_size_mb:.1f} MB) = {compression:.0f}x compression")
    sys.stdout.flush()

    opt = torch.optim.Adam(proc.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    for step in range(n_steps):
        li = torch.randint(0, 28, (1,)).item()
        ti = torch.randint(0, len(type_list), (1,)).item()
        mtype = type_list[ti]
        target = matrix_stacks[mtype][li].to(device)
        rows, cols = target.shape
        pred = proc.generate_weight(li, ti, rows, cols)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(proc.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 3000 == 0:
            print(f"    Step {step}: loss={loss.item():.6f}")
            sys.stdout.flush()

    # Reconstruct
    print("  Reconstructing...")
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

    new_gd = dict(gd)
    new_gd.update(reconstructed_gd)
    recon_model = MiniTransformer(config, device)
    recon_model.load_weights(new_gd)

    t1, t10 = eval_model(lambda tokens, _m=recon_model: _m.forward(tokens, max_layers=28))
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Size={proc_size_mb:.1f}MB {compression:.0f}x MAE={avg_mae:.6f} Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'size_mb': proc_size_mb,
        'compression': compression, 'params': proc_params,
        'avg_mae': avg_mae, 'time': elapsed, 'paradigm': 'procedural',
    }
    sys.stdout.flush()
    del proc, recon_model, reconstructed_gd; torch.cuda.empty_cache()


# ============================================================
# FINAL LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"PARADIGM SHIFT RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Previous bests: Genome=63% top-10 (23.9MB), Quantized INT4=60%+ top-10 (~220MB)")
print(f"NeRF approach: DEAD (0% top-10 at all sizes)")
print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1].get('top10', 0), reverse=True)
for i, (name, r) in enumerate(sorted_all):
    medal = [">>>1st", "   2nd", "   3rd"][i] if i < 3 else f"   {i+1}th"
    print(f"  {medal}: {name:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
          f"Size={r['size_mb']:>7.1f}MB {r['compression']:>6.1f}x MAE={r['avg_mae']:.6f} [{r['paradigm']}]")

with open('paradigm_fast_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to paradigm_fast_results.json")

best_name, best = sorted_all[0]
print(f"\n{'='*70}")
print(f"CHAMPION: {best_name} ({best['paradigm']})")
print(f"  Top1={best['top1']*100:.0f}% Top10={best['top10']*100:.0f}% at {best['size_mb']:.1f}MB ({best['compression']:.0f}x)")

if best['top10'] >= 0.90:
    print("  >>> 90%+ TOP-10. NEAR-ZERO DEGRADATION ACHIEVED.")
    print("  >>> THIS IS IT. Scale to 8B immediately.")
elif best['top10'] >= 0.70:
    print("  >>> 70%+ top-10. Major improvement over genome (63%). Scale up.")
elif best['top10'] >= 0.50:
    print("  >>> 50%+ top-10. Competitive. Can improve with more basis/training.")
else:
    print("  >>> Below 50%. Weight reconstruction alone isn't enough.")
    print("  >>> Need: algebraic basis + genome behavioral fine-tuning (HYBRID)")

print(f"{'='*70}")
