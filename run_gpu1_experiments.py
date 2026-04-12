"""GPU 1 EXPERIMENTS — Run alongside Ollama on GPU 1.

Uses <12GB to leave room for Ollama/overnight.
Runs with torch.compile + FP16 for maximum speed.

Tests the stuff GPU 0 isn't running:
- FRR V3 (LoRA adapters, no hidden supervision)
- FRR + Thalamic routing
- FRR + Predictive coding
- FRR + Activation sparsity
"""
import torch, sys, os, time, json, math, gc, traceback
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
STEPS = 8000
USE_COMPILE = False  # Broken on Windows — Triton compilation fails
USE_FP16 = True

print(f"GPU 1 EXPERIMENTS — torch.compile={USE_COMPILE}, FP16={USE_FP16}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f}GB")

# Load teacher
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

# Check memory after teacher load
mem_used = torch.cuda.memory_allocated() / 1e9
print(f"After teacher load: {mem_used:.1f}GB used")
if mem_used > 12:
    print("WARNING: Already using >12GB, might conflict with Ollama")
    print("Proceeding with smaller batch size...")
    BATCH = 4
else:
    BATCH = 8
sys.stdout.flush()


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
    print(f"\n--- {name} ---")
    params = model.fractal_params()
    compression = sum(v.numel() for k, v in gd.items() if k.startswith('blk.')) / params
    print(f"  Params: {params:,} ({params*2/1e6:.1f} MB) = {compression:.0f}x")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {mem:.1f}GB")
    sys.stdout.flush()

    # Compile for speed
    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile: enabled")
        except:
            print("  torch.compile: failed, using eager")

    trainable = [p for p in model.parameters() if p.requires_grad
                 and id(p) not in {id(ep) for ep in (model.embed.parameters() if hasattr(model, 'embed') else [])}
                 and id(p) not in {id(lp) for lp in (model.lm_head.parameters() if hasattr(model, 'lm_head') else [])}]

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if USE_FP16 else None
    t0 = time.time()

    for step in range(steps):
        cur_lr = lr * 0.5 * (1 + math.cos(step / steps * math.pi)) if step > 500 else lr * step / 500
        for pg in opt.param_groups: pg['lr'] = cur_lr

        tokens = torch.randint(100, 100000, (BATCH, 32), device=device)

        with torch.no_grad():
            if USE_FP16:
                with torch.amp.autocast('cuda'):
                    teacher_logits = teacher.forward(tokens, max_layers=28)
            else:
                teacher_logits = teacher.forward(tokens, max_layers=28)

        if USE_FP16:
            with torch.amp.autocast('cuda'):
                student_logits = model(tokens)
                B, T, V = student_logits.shape
                loss = F.kl_div(F.log_softmax(student_logits.reshape(-1, V)/2, -1),
                               F.softmax(teacher_logits.reshape(-1, V)/2, -1),
                               reduction='batchmean') * 4
        else:
            student_logits = model(tokens)
            B, T, V = student_logits.shape
            loss = F.kl_div(F.log_softmax(student_logits.reshape(-1, V)/2, -1),
                           F.softmax(teacher_logits.reshape(-1, V)/2, -1),
                           reduction='batchmean') * 4

        if torch.isnan(loss):
            for pg in opt.param_groups: pg['lr'] *= 0.1
            continue

        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, 0.5)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()

        if step % 2000 == 0:
            t1_e, t10_e = eval_model(lambda t, _m=model: _m(t))
            sps = (step + 1) / (time.time() - t0)
            print(f"    Step {step}: loss={loss.item():.4f} Top1={t1_e*100:.0f}% Top10={t10_e*100:.0f}% "
                  f"[{sps:.1f} steps/s] ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    t1, t10 = eval_model(lambda t, _m=model: _m(t))
    elapsed = time.time() - t0
    sps = steps / elapsed
    print(f"  RESULT {name}: Top1={t1*100:.0f}% Top10={t10*100:.0f}% "
          f"{params*2/1e6:.1f}MB {compression:.0f}x [{sps:.1f} steps/s] Time={elapsed:.0f}s")
    all_results[name] = {'top1': t1, 'top10': t10, 'params': params,
                        'size_mb': params*2/1e6, 'compression': compression,
                        'time': elapsed, 'steps_per_sec': sps}
    sys.stdout.flush()


print("=" * 70)
print("GPU 1: TESTING BRAIN-INSPIRED ENHANCEMENTS")
print(f"FP16={'ON' if USE_FP16 else 'OFF'}, compile={'ON' if USE_COMPILE else 'OFF'}")
print("=" * 70)
sys.stdout.flush()

# 1. FRR V3 (LoRA, no hidden supervision — predicted best)
try:
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.enable_adapters(rank=16)
    model = model.to(device)  # Move adapters to GPU
    train_and_eval(model, "FRR_V3_LoRA_r16")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# 2. THALAMIC ROUTING
try:
    from ultracompress.thalamic import ThalamicFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = ThalamicFractalBlock(1024, n_heads=8, ff_mult=2, bottleneck=64).to(device)
    def thalamic_fwd(tokens, _m=model):
        x = _m.embed(tokens).float()
        rm = None
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                block_out, rm = _m.block(x, g, b, rm)
                x = x + (block_out - x) * s
        return _m.lm_head(_m.norm(x))
    model.forward = thalamic_fwd
    train_and_eval(model, "FRR_Thalamic")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# 3. PREDICTIVE CODING
try:
    from ultracompress.thalamic import PredictiveCodingLayer
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    pc = nn.ModuleList([PredictiveCodingLayer(1024, 32) for _ in range(28)]).to(device)
    def pc_fwd(tokens, _m=model, _pc=pc):
        x = _m.embed(tokens).float()
        pred = None; lc = 0
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x, pred = _pc[lc](x, pred); lc += 1
        return _m.lm_head(_m.norm(x))
    model.forward = pc_fwd
    orig_fp = model.fractal_params
    model.fractal_params = lambda: orig_fp() + sum(p.numel() for p in pc.parameters())
    train_and_eval(model, "FRR_PredCoding")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# 4. ACTIVATION SPARSITY
try:
    from ultracompress.thalamic import ActivationSparsifier
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    sp = ActivationSparsifier(keep_ratio=0.3).to(device)
    def sparse_fwd(tokens, _m=model, _sp=sp):
        x = _m.embed(tokens).float()
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                x = x + (_m.block(x, g, b) - x) * s
                x = _sp(x)
        return _m.lm_head(_m.norm(x))
    model.forward = sparse_fwd
    train_and_eval(model, "FRR_Sparse30")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()

# 5. LORA + THALAMIC (combo)
try:
    from ultracompress.thalamic import ThalamicFractalBlock
    model = FractalModel(1024, 8, 4, 7, 151936, 2, embed_weight=embed, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    model.block = ThalamicFractalBlock(1024, n_heads=8, ff_mult=2, bottleneck=64).to(device)
    model.enable_adapters(rank=16)
    model = model.to(device)
    def combo_fwd(tokens, _m=model):
        x = _m.embed(tokens).float()
        rm = None; lc = 0
        for scale in range(_m.n_scales):
            g = _m.scale_gamma[scale]; b = _m.scale_beta[scale]
            for it in range(_m.iters_per_scale):
                s = _m.iter_scale[scale, it]
                block_out, rm = _m.block(x, g, b, rm)
                x = x + (block_out - x) * s
                if _m.adapters: x = _m.adapters[lc](x)
                lc += 1
        return _m.lm_head(_m.norm(x))
    model.forward = combo_fwd
    train_and_eval(model, "FRR_LoRA_Thalamic")
except Exception as e: traceback.print_exc(); print(f"FAILED: {e}")
finally:
    if 'model' in dir(): del model
    torch.cuda.empty_cache(); gc.collect()


# LEADERBOARD
total_time = time.time() - pipeline_start
print(f"\n{'='*70}")
print(f"GPU 1 RESULTS ({total_time/60:.0f} min)")
print(f"{'='*70}")
for n, r in sorted(all_results.items(), key=lambda x: x[1]['top10'], reverse=True):
    print(f"  {n:<25} Top10={r['top10']*100:.0f}% {r['size_mb']:.1f}MB {r['compression']:.0f}x [{r['steps_per_sec']:.1f} s/s]")
with open('gpu1_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"{'='*70}")
