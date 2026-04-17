"""ABLATION STUDY — Test each enhancement INDIVIDUALLY, then combine winners.

Scientific method:
1. Baseline: FRR with standard block (CONTROL)
2. Test each variable ONE AT A TIME
3. Measure effect of each
4. Combine the best ones
5. Test the combination

Variables to test:
A. PHM layers (hypercomplex, 1/4 params)
B. Dendritic neurons (more compute per param)
C. LoRA adapters (per-layer specialization)
D. Hidden supervision (cosine loss on hidden states)
E. Temperature annealing (T=4 -> T=2)

Each gets 10K steps, same hyperparams, same eval. Pure comparison.
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel, FractalBlock

device = 'cuda'
print("Loading teacher...")
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
positions = torch.arange(32, device=device)

STEPS = 10000  # Same for all — fair comparison
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


def cosine_hidden_loss(s, t):
    return (1 - F.cosine_similarity(s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]), dim=-1)).mean()


def train_frr(model, name, use_hidden_sup=False, use_temp_anneal=False,
              lr=0.0005, steps=STEPS):
    """Generic FRR training loop with configurable enhancements."""
    print(f"\n--- {name} ---")
    params = model.fractal_params()
    teacher_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
    compression = teacher_params / params
    print(f"  Params: {params:,} ({params*2/1e6:.1f} MB) = {compression:.0f}x compression")
    sys.stdout.flush()

    # Collect all trainable params
    trainable = []
    for p in model.parameters():
        if p.requires_grad and not any(p is ep for ep in model.embed.parameters()) \
           and not any(p is lp for lp in model.lm_head.parameters()) \
           and not any(p is np_ for np_ in model.norm.parameters()):
            trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    warmup = 1000
    t0 = time.time()

    supervision_layers = [6, 13, 20, 27]

    for step in range(steps):
        if step < warmup:
            cur_lr = lr * step / warmup
        else:
            cur_lr = lr * 0.5 * (1 + math.cos((step - warmup) / (steps - warmup) * math.pi))
        for pg in opt.param_groups: pg['lr'] = cur_lr

        temp = (4.0 - 2.0 * min(1.0, step / (steps * 0.5))) if use_temp_anneal else 2.0

        tokens = torch.randint(100, 100000, (8, 32), device=device)

        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)
            teacher_hiddens = {}
            if use_hidden_sup:
                tx = F.embedding(tokens, embed).float()
                for li in range(28):
                    tx = teacher.layers[li](tx, positions)
                    if li in supervision_layers:
                        teacher_hiddens[li] = tx.clone()

        student_logits = model(tokens)

        B, T, V = student_logits.shape
        kl_loss = F.kl_div(
            F.log_softmax(student_logits.reshape(-1, V) / temp, -1),
            F.softmax(teacher_logits.reshape(-1, V) / temp, -1),
            reduction='batchmean') * (temp ** 2)

        loss = kl_loss

        if use_hidden_sup and teacher_hiddens:
            # Capture student hiddens
            x = model.embed(tokens).float()
            student_hiddens = {}
            layer_count = 0
            for scale in range(model.n_scales):
                gamma = model.scale_gamma[scale]
                beta = model.scale_beta[scale]
                for it in range(model.iters_per_scale):
                    iter_s = model.iter_scale[scale, it]
                    x = x + (model.block(x, gamma, beta) - x) * iter_s
                    if hasattr(model, 'adapters') and model.adapters is not None:
                        x = model.adapters[layer_count](x)
                    if layer_count in supervision_layers:
                        student_hiddens[layer_count] = x
                    layer_count += 1

            h_loss = sum(cosine_hidden_loss(student_hiddens[li], teacher_hiddens[li])
                        for li in supervision_layers if li in student_hiddens) / len(supervision_layers)
            hidden_weight = max(0.05, 0.3 * (1 - step / steps))
            loss = loss + hidden_weight * h_loss

        if torch.isnan(loss):
            for pg in opt.param_groups: pg['lr'] *= 0.1
            continue

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        if step % 2000 == 0:
            t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
            print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    t1, t10 = eval_model(lambda t, _m=model: _m(t))
    elapsed = time.time() - t0
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"Params={params:,} ({params*2/1e6:.1f}MB) {compression:.0f}x Time={elapsed:.0f}s")
    all_results[name] = {
        'top1': t1, 'top10': t10, 'params': params,
        'size_mb': params*2/1e6, 'compression': compression, 'time': elapsed,
    }
    sys.stdout.flush()
    return t1, t10


print("=" * 70)
print("ABLATION STUDY — Test each enhancement individually")
print(f"All tests: {STEPS} steps, same LR, same eval")
print("=" * 70)
sys.stdout.flush()

# ============================================================
# A. CONTROL: Standard FRR (baseline for comparison)
# ============================================================
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    train_frr(model, "A_Control_FRR")
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    del model; torch.cuda.empty_cache(); gc.collect()

# ============================================================
# B. PHM LAYERS (hypercomplex, 1/4 params per linear)
# ============================================================
try:
    from ultracompress.hypercomplex import PHMFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    # Replace the standard block with PHM block
    model.block = PHMFractalBlock(1024, n_heads=8, ff_mult=2, n=4).to(device)
    train_frr(model, "B_PHM_n4")
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
# C. DENDRITIC NEURONS (more compute per param)
# ============================================================
try:
    from ultracompress.dendritic import DendriticFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = DendriticFractalBlock(1024, n_heads=8, ff_mult=2, n_dendrites=4).to(device)
    train_frr(model, "C_Dendritic_4d")
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
# D. LORA ADAPTERS (per-layer specialization)
# ============================================================
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.enable_adapters(rank=16)
    train_frr(model, "D_LoRA_r16")
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
# E. HIDDEN SUPERVISION (cosine loss at hidden layers)
# ============================================================
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    train_frr(model, "E_HiddenSup", use_hidden_sup=True)
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
# F. TEMPERATURE ANNEALING (T=4 -> T=2)
# ============================================================
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2,
                         embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    train_frr(model, "F_TempAnneal", use_temp_anneal=True)
except Exception as e:
    traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()


# ============================================================
# PHASE 2: COMBINE THE WINNERS
# ============================================================
print(f"\n{'='*70}")
print("PHASE 1 RESULTS — Individual Effects")
print(f"{'='*70}")

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)
control_t10 = all_results.get('A_Control_FRR', {}).get('top10', 0)

for name, r in sorted_results:
    delta = (r['top10'] - control_t10) * 100
    marker = "+++" if delta > 2 else "++" if delta > 0 else "--" if delta < -2 else "~"
    print(f"  {marker} {name:<25} Top10={r['top10']*100:>4.0f}% ({delta:>+5.1f}%) "
          f"Params={r['params']:,} ({r['size_mb']:.1f}MB)")

# Find winners (anything that beats or matches control)
winners = [name for name, r in all_results.items()
           if name != 'A_Control_FRR' and r['top10'] >= control_t10 - 0.01]

if winners:
    print(f"\nWINNERS to combine: {winners}")

    # Combine all winners into one model
    print(f"\n{'='*70}")
    print("PHASE 2: COMBINED WINNERS")
    print(f"{'='*70}")

    try:
        model = FractalModel(1024, 8, 4, 7, 151936, 2,
                             embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

        # Apply winner enhancements
        use_hs = False
        use_ta = False
        if any('PHM' in w for w in winners):
            from ultracompress.hypercomplex import PHMFractalBlock
            model.block = PHMFractalBlock(1024, n_heads=8, ff_mult=2, n=4).to(device)
            print("  + PHM layers")
        elif any('Dendritic' in w for w in winners):
            from ultracompress.dendritic import DendriticFractalBlock
            model.block = DendriticFractalBlock(1024, n_heads=8, ff_mult=2, n_dendrites=4).to(device)
            print("  + Dendritic neurons")
        if any('LoRA' in w for w in winners):
            model.enable_adapters(rank=16)
            print("  + LoRA adapters")
        if any('HiddenSup' in w for w in winners):
            use_hs = True
            print("  + Hidden supervision")
        if any('TempAnneal' in w for w in winners):
            use_ta = True
            print("  + Temperature annealing")

        train_frr(model, "COMBINED_WINNERS", use_hidden_sup=use_hs, use_temp_anneal=use_ta,
                  steps=15000)  # More steps for combined

    except Exception as e:
        traceback.print_exc(); print(f"FAILED: {e}")
    finally:
        if 'model' in dir(): del model
        torch.cuda.empty_cache(); gc.collect()
else:
    print("\nNo clear winners over control. FRR base is already optimal at this scale.")


# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"ABLATION STUDY COMPLETE ({total_time/60:.0f} min)")
print(f"{'='*70}")
for name, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {name:<25} Top10={r['top10']*100:.0f}% Params={r['params']:,} ({r['compression']:.0f}x)")

with open('ablation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved to ablation_results.json")
best_name = max(all_results, key=lambda k: all_results[k]['top10'])
best = all_results[best_name]
print(f"\nBEST: {best_name} at {best['top10']*100:.0f}% top-10, {best['size_mb']:.1f}MB ({best['compression']:.0f}x)")
print(f"{'='*70}")
