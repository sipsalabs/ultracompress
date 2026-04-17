"""TEST THE IMPOSSIBLE — Every wild idea, let data decide.

Tests every module from impossible.py and neuro_advanced.py on FRR.
"Even if it doesn't sound good on paper it might be insane in practice."
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
STEPS = 6000  # Quick — just enough to see if something works

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
if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)
embed = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

all_results = {}


def eval_model(forward_fn, n=50):
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


def quick_train(model, name, steps=STEPS, lr=0.0005):
    print(f"\n--- {name} ---")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {params:,}")
    sys.stdout.flush()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    t0 = time.time()

    for step in range(steps):
        cur_lr = lr * 0.5 * (1 + math.cos(step / steps * math.pi)) if step > 300 else lr * step / 300
        for pg in opt.param_groups: pg['lr'] = cur_lr
        tokens = torch.randint(100, 100000, (8, 32), device=device)
        with torch.no_grad():
            teacher_logits = teacher.forward(tokens, max_layers=28)
        student_logits = model(tokens)
        B, T, V = student_logits.shape
        loss = F.kl_div(F.log_softmax(student_logits.reshape(-1,V)/2,-1),
                       F.softmax(teacher_logits.reshape(-1,V)/2,-1),
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
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'params': params, 'time': elapsed}
    sys.stdout.flush()


print("=" * 70)
print("TEST THE IMPOSSIBLE + NEURO ADVANCED")
print(f"Each: {STEPS} steps, quick eval")
print("=" * 70)

# 1. ZERO PARAM TRANSFORM
try:
    from ultracompress.impossible import ZeroParamTransform
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    zpt = nn.ModuleList([ZeroParamTransform(1024) for _ in range(28)]).to(device)
    def zpt_fwd(tokens, _m=model, _z=zpt):
        x = _m.embed(tokens).float()
        lc = 0
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x = _z[lc](x); lc += 1
        return _m.lm_head(_m.norm(x))
    model.forward = zpt_fwd
    quick_train(model, "ZeroParam")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# 2. SELECTIVE FORGETTER
try:
    from ultracompress.impossible import SelectiveForgetter
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    forgetter = SelectiveForgetter(1024, keep_ratio=0.5).to(device)
    def forget_fwd(tokens, _m=model, _f=forgetter):
        x = _m.embed(tokens).float()
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x = _f(x)
        return _m.lm_head(_m.norm(x))
    model.forward = forget_fwd
    quick_train(model, "Forgetter_50pct")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# 3. HEBBIAN ADAPTER
try:
    from ultracompress.impossible import HebbianAdapter
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    hebb = nn.ModuleList([HebbianAdapter(1024, rank=16) for _ in range(28)]).to(device)
    def hebb_fwd(tokens, _m=model, _h=hebb):
        x = _m.embed(tokens).float()
        lc = 0
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x = _h[lc](x); lc += 1
        return _m.lm_head(_m.norm(x))
    model.forward = hebb_fwd
    quick_train(model, "Hebbian")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# 4. WEIGHT TELEPORTER
try:
    from ultracompress.impossible import WeightTeleporter
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    teleporter = WeightTeleporter(1024, n_waves=32).to(device)
    def tele_fwd(tokens, _m=model, _t=teleporter):
        x = _m.embed(tokens).float()
        lc = 0
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x = _t(x, lc / 27.0); lc += 1
        return _m.lm_head(_m.norm(x))
    model.forward = tele_fwd
    quick_train(model, "Teleporter")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# 5. NEURO FRACTAL BLOCK (phase + astrocyte + oscillatory)
try:
    from ultracompress.neuro_advanced import NeuroFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = NeuroFractalBlock(1024, n_heads=8, ff_mult=2).to(device)
    # Custom forward to pass astro state
    def neuro_fwd(tokens, _m=model):
        x = _m.embed(tokens).float()
        astro_state = None
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                block_out, astro_state = _m.block(x, g, b, astro_state=astro_state)
                x = x + (block_out - x) * s
        return _m.lm_head(_m.norm(x))
    model.forward = neuro_fwd
    quick_train(model, "NeuroFractal")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally: del model; torch.cuda.empty_cache(); gc.collect()

# LEADERBOARD
print(f"\n{'='*70}")
print("IMPOSSIBLE + NEURO RESULTS")
print(f"{'='*70}")
for n, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {n:<25} Top10={r['top10']*100:.0f}% Params={r['params']:,}")
with open('impossible_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"{'='*70}")
