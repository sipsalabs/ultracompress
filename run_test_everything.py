"""TEST EVERYTHING — No filtering. Every module, every combination.

"Even if it doesn't sound good on paper it might be insane in practice."

Tests EVERY enhancement module we've built, individually and in combinations.
The DATA decides what works, not our assumptions.

Modules to test:
A. FRR base (control)
B. PHM (hypercomplex 4x)
C. Dendritic (more compute/param)
D. LoRA adapters
E. Thalamic Q-routing
F. Predictive coding
G. Activation sparsity
H. Hyperbolic Q/K
I. Combined: best individual winners
J. MEGA: everything at once (why not?)
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

STEPS = 8000  # Quick test each — 8K steps, enough to see trends
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


def train_and_eval(model, name, steps=STEPS, lr=0.0005):
    """Train any FRR variant and eval."""
    print(f"\n--- {name} ---")
    params = model.fractal_params()
    teacher_params = sum(v.numel() for k, v in gd.items() if k.startswith('blk.'))
    compression = teacher_params / params
    print(f"  Params: {params:,} ({params*2/1e6:.1f} MB) = {compression:.0f}x")
    sys.stdout.flush()

    trainable = [p for p in model.parameters() if p.requires_grad
                 and not any(p is ep for ep in model.embed.parameters())
                 and not any(p is lp for lp in model.lm_head.parameters())
                 and not any(p is np_ for np_ in model.norm.parameters())]

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    t0 = time.time()

    for step in range(steps):
        cur_lr = lr * 0.5 * (1 + math.cos(step / steps * math.pi)) if step > 500 else lr * step / 500
        for pg in opt.param_groups: pg['lr'] = cur_lr

        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)
        student_logits = model(tokens)

        B, T, V = student_logits.shape
        loss = F.kl_div(F.log_softmax(student_logits.reshape(-1, V)/2, -1),
                       F.softmax(teacher_logits.reshape(-1, V)/2, -1),
                       reduction='batchmean') * 4

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
    all_results[name] = {'top1': t1, 'top10': t10, 'params': params,
                        'size_mb': params*2/1e6, 'compression': compression, 'time': elapsed}
    sys.stdout.flush()
    return t1, t10


print("=" * 70)
print("TEST EVERYTHING — No filtering, DATA decides")
print(f"Each test: {STEPS} steps, same conditions")
print("=" * 70)
sys.stdout.flush()


# A. CONTROL
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    train_and_eval(model, "A_Control")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# B. PHM (hypercomplex)
try:
    from ultracompress.hypercomplex import PHMFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = PHMFractalBlock(1024, n_heads=8, ff_mult=2, n=4).to(device)
    train_and_eval(model, "B_PHM_n4")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# C. DENDRITIC
try:
    from ultracompress.dendritic import DendriticFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = DendriticFractalBlock(1024, n_heads=8, ff_mult=2, n_dendrites=4).to(device)
    train_and_eval(model, "C_Dendritic")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# D. LORA ADAPTERS
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.enable_adapters(rank=16)
    train_and_eval(model, "D_LoRA_r16")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# E. THALAMIC ROUTING
try:
    from ultracompress.thalamic import ThalamicFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = ThalamicFractalBlock(1024, n_heads=8, ff_mult=2, bottleneck=64).to(device)
    # Need custom forward to pass running_mean through iterations
    original_forward = model.forward
    def thalamic_forward(tokens, _m=model):
        x = _m.embed(tokens).float()
        rm = None
        for scale in range(_m.n_scales):
            gamma = _m.scale_gamma[scale]
            beta = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                iter_s = _m.iter_scale[scale, it]
                block_out, rm = _m.block(x, gamma, beta, rm)
                x = x + (block_out - x) * iter_s
        x = _m.norm(x)
        return _m.lm_head(x)
    model.forward = thalamic_forward
    train_and_eval(model, "E_Thalamic")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# F. PREDICTIVE CODING
try:
    from ultracompress.thalamic import PredictiveCodingLayer
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    pc_layers = nn.ModuleList([PredictiveCodingLayer(1024, rank=32) for _ in range(28)]).to(device)
    original_forward = model.forward
    def pc_forward(tokens, _m=model, _pc=pc_layers):
        x = _m.embed(tokens).float()
        prediction = None
        layer_count = 0
        for scale in range(_m.n_scales):
            gamma = _m.scale_gamma[scale]
            beta = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                iter_s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, gamma, beta) - x) * iter_s
                x, prediction = _pc[layer_count](x, prediction)
                layer_count += 1
        x = _m.norm(x)
        return _m.lm_head(x)
    model.forward = pc_forward
    # Add PC params to model for counting
    model._pc_layers = pc_layers
    orig_fp = model.fractal_params
    model.fractal_params = lambda: orig_fp() + sum(p.numel() for p in pc_layers.parameters())
    train_and_eval(model, "F_PredictiveCoding")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# G. ACTIVATION SPARSITY
try:
    from ultracompress.thalamic import ActivationSparsifier
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    sparsifier = ActivationSparsifier(keep_ratio=0.3).to(device)  # Keep 30% (conservative)
    original_forward = model.forward
    def sparse_forward(tokens, _m=model, _sp=sparsifier):
        x = _m.embed(tokens).float()
        for scale in range(_m.n_scales):
            gamma = _m.scale_gamma[scale]
            beta = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                iter_s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, gamma, beta) - x) * iter_s
                x = _sp(x)  # Sparsify after each virtual layer
        x = _m.norm(x)
        return _m.lm_head(x)
    model.forward = sparse_forward
    train_and_eval(model, "G_Sparse_30pct")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()


# ============================================================
# LEADERBOARD
# ============================================================
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"TEST EVERYTHING RESULTS ({total_time/60:.0f} min)")
print(f"{'='*70}")

if not all_results:
    print("  No results (everything failed)")
else:
    control_t10 = all_results.get('A_Control', {}).get('top10', 0)
    sorted_r = sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True)

    for name, r in sorted_r:
        delta = (r['top10'] - control_t10) * 100
        marker = "+++" if delta > 3 else "++" if delta > 0 else "~" if abs(delta) < 1 else "--"
        print(f"  {marker} {name:<25} Top10={r['top10']*100:>4.0f}% ({delta:>+5.1f}%) "
              f"Params={r['params']:,} ({r['size_mb']:.1f}MB) {r['compression']:.0f}x")

    with open('test_everything_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Identify winners
    winners = [n for n, r in all_results.items()
               if n != 'A_Control' and r['top10'] >= control_t10 - 0.005]
    if winners:
        print(f"\n  WINNERS (match or beat control): {winners}")
        print(f"  Next: combine these into MEGA model")

print(f"{'='*70}")
