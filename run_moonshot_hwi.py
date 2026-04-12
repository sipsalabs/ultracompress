"""MOONSHOT: Holographic Weight Interference — One hologram, many layer keys.

THE IDEA: Store ALL layer weights in a single shared complex-valued tensor
(the "hologram") via superposition. Each layer reconstructs its own weights
by interfering with a unique low-rank complex "address key" pair.

Like holographic memory: the hologram is dense shared storage, layer keys
are tiny addresses. For hidden_dim=1024, rank=16, 28 layers:
  hologram = 1024^2 * 2 = ~2M real params
  keys     = 28 * 2 * 2 * 16 * 1024 = ~1.8M real params
  total    = ~4M params for ALL 28 layers

Compare to teacher's 28 independent layers at ~24M params = 6x compression
from architecture alone, before any quantization.

The key insight: if weight matrices across layers share deep structure
(which we know from genome/fractal experiments), a holographic encoding
should capture that structure in a single dense tensor.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import HolographicModel

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
print("MOONSHOT: HOLOGRAPHIC WEIGHT INTERFERENCE")
print("One shared complex hologram + per-layer low-rank address keys")
print("=" * 70)
sys.stdout.flush()

# Test configs: (rank, n_steps, name)
configs = [
    (16, 15000, "HWI-rank16"),
    (32, 15000, "HWI-rank32"),
    (64, 15000, "HWI-rank64"),
]

for rank, n_steps, name in configs:
    print(f"\n--- {name} (rank={rank}, 28 layers, hidden=1024) ---")
    sys.stdout.flush()
    t0 = time.time()

    try:
        model = HolographicModel(
            hidden_dim=1024, n_heads=16, n_layers=28, rank=rank,
            vocab_size=151936,
            embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w,
        ).to(device)

        holo_p = model.holographic_params()
        teacher_layer_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
        print(f"  Holographic params: {holo_p:,} ({holo_p*2/1e6:.1f} MB)")
        print(f"    hologram: {model.hologram_real.numel()*2:,} real params")
        print(f"    keys:     {model.key_a_real.numel()*4:,} real params")
        print(f"  vs Teacher layers: {teacher_layer_params:,} ({teacher_layer_params*2/1e6:.1f} MB)")
        print(f"  Compression: {teacher_layer_params/holo_p:.0f}x")
        sys.stdout.flush()

        # Collect trainable parameters (hologram + keys + scales + norms)
        trainable = [
            model.hologram_real, model.hologram_imag,
            model.key_a_real, model.key_a_imag,
            model.key_b_real, model.key_b_imag,
            model.layer_scale,
        ]
        for norm_pair in model.norms:
            for norm_mod in norm_pair:
                trainable.extend(norm_mod.parameters())

        opt = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
        warmup = 1000

        for step in range(n_steps):
            if step < warmup:
                lr = 0.001 * step / warmup
            else:
                lr = 0.001 * 0.5 * (1 + math.cos((step - warmup) / (n_steps - warmup) * math.pi))
            for pg in opt.param_groups: pg['lr'] = lr

            tokens = torch.randint(100, 100000, (8, 32), device=device)
            with torch.no_grad():
                teacher_logits = teacher.forward(tokens, max_layers=28)

            student_logits = model(tokens)

            # All-position KL distillation
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
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

            if step % 3000 == 0:
                t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
                print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% lr={lr:.6f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

        t1, t10 = eval_model(lambda t, _m=model: _m(t))
        elapsed = time.time() - t0
        size_mb = holo_p * 2 / 1e6
        compression = teacher_layer_params / holo_p

        print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
              f"Size={size_mb:.1f}MB {compression:.0f}x Time={elapsed:.0f}s")
        all_results[name] = {
            'top1': t1, 'top10': t10, 'size_mb': size_mb,
            'compression': compression, 'params': holo_p,
            'rank': rank, 'time': elapsed,
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
print(f"MOONSHOT HWI RESULTS (Total: {total_time/60:.0f} min)")
print(f"{'='*70}")
print(f"Comparison: Genome=63% top-10 at 23.9MB (37x)")
print()

sorted_all = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
for i, (n, r) in enumerate(sorted_all):
    print(f"  {n:<20} Top1={r['top1']*100:>4.0f}% Top10={r['top10']*100:>4.0f}% "
          f"Size={r['size_mb']:>6.1f}MB {r['compression']:>5.0f}x rank={r['rank']}")

with open('hwi_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

if sorted_all:
    best_n, best = sorted_all[0]
    print(f"\nBest: {best_n} at {best['top10']*100:.0f}% top-10, {best['size_mb']:.1f}MB ({best['compression']:.0f}x)")
    if best['top10'] > 0.63:
        print(">>> BEATS GENOME! Holographic superposition stores weights more efficiently!")
        print(">>> The hologram captures cross-layer structure that independent weights miss.")
    elif best['top10'] > 0.40:
        print(">>> Competitive. Holographic encoding works! Tune rank/LR for more.")
    else:
        print(">>> Below expectations. Holographic reconstruction may need:")
        print(">>>   - More rank, phase-aware training, or multi-hologram approach")
        print(">>>   - Per-layer amplitude modulation on top of phase keys")
print(f"{'='*70}")
