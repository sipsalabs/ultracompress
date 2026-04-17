"""MOONSHOT: Fractal Residual Recursion — One block, many virtual layers.

THE IDEA: Instead of 28 independent transformer layers (28x the params),
use ONE shared block applied recursively 28 times with per-scale modulation.
That's 1/28th the layer params = ~28x inherent compression.

This tests whether a shared-weight recursive architecture can learn
language modeling when distilled from Qwen3-0.6B.

If FRR at 1/28th params matches genome at full params, we have proof
that weight independence is WASTEFUL, not necessary.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
print("Loading teacher (Qwen3-0.6B)...")
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
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

all_results = {}
pipeline_start = time.time()


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


print("=" * 70)
print("MOONSHOT: FRACTAL RESIDUAL RECURSION")
print("One shared block, recursive application, per-scale modulation")
print("=" * 70)
sys.stdout.flush()

# Test configs: (n_scales, iters_per_scale, n_heads, ff_mult, n_steps, name)
configs = [
    # 4 scales x 7 iters = 28 effective layers (matches Qwen3-0.6B depth)
    (4, 7, 8, 2, 15000, "FRR-4s7i-h8"),
    # 7 scales x 4 iters = 28 effective layers (more scale diversity)
    (7, 4, 8, 2, 15000, "FRR-7s4i-h8"),
    # 4 scales x 7 iters with bigger head count
    (4, 7, 16, 2, 15000, "FRR-4s7i-h16"),
    # Deep fractal: 2 scales x 14 iters (test pure recursion depth)
    (2, 14, 8, 2, 15000, "FRR-2s14i-h8"),
]

for n_scales, iters, n_heads, ff_mult, n_steps, name in configs:
    print(f"\n--- {name} ({n_scales} scales x {iters} iters = {n_scales*iters} eff. layers) ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = FractalModel(
            hidden_dim=1024, n_heads=n_heads, n_scales=n_scales,
            iters_per_scale=iters, vocab_size=151936, ff_mult=ff_mult,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        fractal_p = model.fractal_params()
        # Compare to genome (11.9M params for 28 layers)
        teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
        print(f"  Fractal params: {fractal_p:,} ({fractal_p*2/1e6:.1f} MB)")
        print(f"  vs Teacher layers: {teacher_layer_params:,} ({teacher_layer_params*2/1e6:.1f} MB)")
        print(f"  Compression: {teacher_layer_params/fractal_p:.0f}x")
        sys.stdout.flush()

        # Train with all-position KL + hidden supervision from supertrain findings
        trainable = list(model.block.parameters()) + [model.scale_gamma, model.scale_beta, model.iter_scale]

        opt = torch.optim.AdamW(trainable, lr=0.0005, weight_decay=0.01)
        warmup = 1000

        for step in range(n_steps):
            if step < warmup:
                lr = 0.0005 * step / warmup
            else:
                lr = 0.0005 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)

            student_logits = model(tokens)

            # All-position KL
            B, T, V = student_logits.shape
            loss = F.kl_div(
                F.log_softmax(student_logits.reshape(-1, V) / 2, -1),
                F.softmax(teacher_logits.reshape(-1, V) / 2, -1),
                reduction='batchmean') * 4

            if torch.isnan(loss):
                print(f"    NaN at step {step}, reducing LR and continuing...")
                for pg in opt.param_groups: pg['lr'] *= 0.1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)  # Tighter clipping for stability
            opt.step()

            if step % 3000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% lr={lr:.6f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t))
        elapsed = time.time() - t0
        size_mb = fractal_p * 2 / 1e6
        compression = teacher_layer_params / fractal_p

        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"Size={size_mb:.1f}MB {compression:.0f}x Time={elapsed:.0f}s")
        all_results[name] = {
            'top1': t1, 'top10': t10, 'size_mb': size_mb,
            'compression': compression, 'params': fractal_p,
            'time': elapsed,
        }

    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
    sys.stdout.flush()


# Leaderboard
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"MOONSHOT FRR RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Comparison: Genome=63% top-10 at 23.9MB (37x)")
print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
for i, (n, r) in enumerate(sorted_all):
    print(f"  {n:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
          f"Size={r['size_mb']:>6.1f}MB {r['compression']:>5.0f}x")

with open('moonshot_frr_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

if sorted_all:
    best_n, best = sorted_all[0]
    print(f"\nBest: {best_n} at {best['top10']*100:.0f}% top-10, {best['size_mb']:.1f}MB ({best['compression']:.0f}x)")
    if best['top10'] > 0.63:
        print(">>> BEATS GENOME WITH SHARED WEIGHTS! Weight independence IS wasteful!")
        print(">>> This proves a new architecture paradigm. Scale it up.")
    elif best['top10'] > 0.40:
        print(">>> Competitive. Shared weights work! Needs refinement.")
    else:
        print(">>> Below expectations. Shared block can't capture per-layer specialization.")
        print(">>> Try: more per-scale params, deeper modulation, or GWE approach.")
print(f"{'='*70}")
