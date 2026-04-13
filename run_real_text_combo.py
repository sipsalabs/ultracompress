"""
THE REAL SHOT — Everything proven, combined, on real text.

Why this should break the plateau:
- 500K run PROVED: random tokens cap at 63% T10 (step 50K = 100K = same)
- The bottleneck is the TRAINING SIGNAL, not architecture or steps
- Real text gives meaningful teacher outputs (not noise on random ints)
- Combined loss attacks quality from 3 angles

What we're stacking:
1. Real text inputs (FineWeb-Edu, not random tokens)
2. Combined loss: KL soft + CE hard + attention matching
3. PHM block (proven: same quality, 4x fewer params = 168x)
4. Longer training (50K steps)
5. Evaluate on BOTH T10 and real benchmarks (HellaSwag)
"""
import lib.unbuffered
import torch, sys, os, time, math, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 50000

print("=" * 60)
print("THE REAL SHOT: Real text + combined loss + PHM")
print("500K proved: random tokens cap at 63%. Time to break it.")
print("=" * 60)

# Load teacher
print("Loading teacher...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
               'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
               'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
               'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
               'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
               'mlp.down_proj.weight': 'ffn_down.weight'}
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
embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# Load real text
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_real_batch(batch_size=8, seq_len=64):
    global ds_iter
    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200: continue
                toks = tokenizer.encode(text, max_length=seq_len, truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len:
                    tokens_list.append(toks[:seq_len])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)

def eval_random_tokens(model, n=200):
    """Standard T1/T10 eval on random tokens (for comparison with old results)."""
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            gl = model(t)
            tp = tl[0, -1].argmax().item()
            gp = gl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)

def eval_real_text(model, n=100):
    """T1/T10 on REAL text — the metric that actually matters."""
    t1, t10s = 0, []
    for trial in range(n):
        batch = get_real_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=28)
            gl = model(batch)
            tp = tl[0, -1].argmax().item()
            gp = gl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)

def eval_hellaswag(model_fn, n_samples=200):
    """HellaSwag benchmark — the REAL quality test."""
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
    except:
        print("  (HellaSwag download failed, skipping)")
        return -1
    correct = 0
    total = 0
    for i, sample in enumerate(hs):
        if i >= n_samples: break
        ctx = sample['ctx']
        endings = sample['endings']
        label = int(sample['label'])
        best_score = float('-inf')
        best_idx = 0
        for j, ending in enumerate(endings):
            text = ctx + " " + ending
            tokens = tokenizer.encode(text, max_length=128, truncation=True, return_tensors='pt').to(device)
            if tokens.shape[1] < 2: continue
            with torch.no_grad():
                logits = model_fn(tokens)
            ctx_len = len(tokenizer.encode(ctx, max_length=128, truncation=True))
            if ctx_len >= tokens.shape[1] - 1: continue
            log_probs = F.log_softmax(logits[0, ctx_len-1:-1], dim=-1)
            targets = tokens[0, ctx_len:]
            score = log_probs.gather(1, targets.unsqueeze(1)).mean().item()
            if score > best_score:
                best_score = score
                best_idx = j
        if best_idx == label: correct += 1
        total += 1
    return correct / total if total > 0 else 0


# ═══════════════════════════════════════════════════════════
# CONFIG 1: Random token baseline (for comparison)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  CONFIG 1: Random token baseline (15K steps)")
print(f"{'='*60}")
model_rand = FractalModel(1024, 16, 4, 7, 151936, 1,
                          embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
params = [p for p in model_rand.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15000)
t0 = time.time()
for step in range(15000):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (8, 64), device=device)
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=28)
    sl = model_rand(tokens)
    T = max(2.0, 5.0 * (1 - step / 15000))
    loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                   reduction='batchmean') * T * T
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        rt1, rt10 = eval_random_tokens(model_rand, n=50)
        print(f"    Step {step}: loss={loss.item():.4f} rand_T10={rt10*100:.0f}% ({time.time()-t0:.0f}s)")

rt1, rt10 = eval_random_tokens(model_rand, n=200)
rrt1, rrt10 = eval_real_text(model_rand, n=100)
print(f"  Random token baseline FINAL:")
print(f"    Random T1={rt1*100:.0f}% T10={rt10*100:.0f}%")
print(f"    Real text T1={rrt1*100:.0f}% T10={rrt10*100:.0f}%")
del model_rand; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# CONFIG 2: Real text distillation (THE TEST)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  CONFIG 2: Real text distillation (15K steps)")
print("  Same model, same teacher, REAL text inputs")
print(f"{'='*60}")
model_real = FractalModel(1024, 16, 4, 7, 151936, 1,
                          embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
params = [p for p in model_real.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15000)
t0r = time.time()
for step in range(15000):
    tokens = get_real_batch(8, 64)
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=28)
    sl = model_real(tokens)
    T = max(2.0, 5.0 * (1 - step / 15000))
    loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                   reduction='batchmean') * T * T
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        rt1, rt10 = eval_random_tokens(model_real, n=50)
        rrt1, rrt10 = eval_real_text(model_real, n=50)
        print(f"    Step {step}: loss={loss.item():.4f} rand_T10={rt10*100:.0f}% real_T10={rrt10*100:.0f}% ({time.time()-t0r:.0f}s)")

rt1, rt10 = eval_random_tokens(model_real, n=200)
rrt1, rrt10 = eval_real_text(model_real, n=200)
print(f"  Real text distillation FINAL:")
print(f"    Random T1={rt1*100:.0f}% T10={rt10*100:.0f}%")
print(f"    Real text T1={rrt1*100:.0f}% T10={rrt10*100:.0f}%")
del model_real; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# CONFIG 3: Combined loss on real text (THE FULL SHOT)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  CONFIG 3: Combined loss (KL soft + CE hard) on real text")
print(f"{'='*60}")
model_combo = FractalModel(1024, 16, 4, 7, 151936, 1,
                           embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
params = [p for p in model_combo.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15000)
t0c = time.time()
for step in range(15000):
    tokens = get_real_batch(8, 64)
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=28)
    sl = model_combo(tokens)

    # COMBINED LOSS:
    # 1. KL divergence (soft targets) — learn the distribution
    T = max(2.0, 5.0 * (1 - step / 15000))
    loss_kl = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                       reduction='batchmean') * T * T

    # 2. Cross-entropy on teacher's argmax (hard targets) — nail the top prediction
    hard_targets = tl.argmax(dim=-1)
    loss_ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), hard_targets.reshape(-1))

    # 3. Next-token prediction on actual text (real language signal)
    loss_ntp = F.cross_entropy(sl[:, :-1].reshape(-1, sl.shape[-1]), tokens[:, 1:].reshape(-1))

    # Weighted combination
    loss = 0.5 * loss_kl + 0.3 * loss_ce + 0.2 * loss_ntp

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        rt1, rt10 = eval_random_tokens(model_combo, n=50)
        rrt1, rrt10 = eval_real_text(model_combo, n=50)
        print(f"    Step {step}: loss={loss.item():.4f} rand_T10={rt10*100:.0f}% real_T10={rrt10*100:.0f}% ({time.time()-t0c:.0f}s)")

rt1, rt10 = eval_random_tokens(model_combo, n=200)
rrt1, rrt10 = eval_real_text(model_combo, n=200)
print(f"  Combined loss FINAL:")
print(f"    Random T1={rt1*100:.0f}% T10={rt10*100:.0f}%")
print(f"    Real text T1={rrt1*100:.0f}% T10={rrt10*100:.0f}%")

# HellaSwag on the best model
print(f"\n{'='*60}")
print("  HELLASWAG EVALUATION")
print(f"{'='*60}")
print("  Teacher...")
hs_teacher = eval_hellaswag(lambda t: teacher.forward(t, max_layers=28), n_samples=200)
print(f"  Teacher HellaSwag: {hs_teacher*100:.1f}%")
print("  Combined model...")
hs_combo = eval_hellaswag(lambda t: model_combo(t), n_samples=200)
print(f"  Combined HellaSwag: {hs_combo*100:.1f}%")
if hs_teacher > 0:
    print(f"  Retention: {hs_combo/hs_teacher*100:.1f}%")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Random token plateau (confirmed): 63% T10 at 50K-100K steps")
print(f"  Does real text break it? Check real_T10 numbers above.")
print(f"  Does combined loss help? Check combined vs single-loss.")
print(f"  Does HellaSwag tell a different story? Check retention.")
print("Done!")
