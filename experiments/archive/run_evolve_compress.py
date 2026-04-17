"""
EVOLUTIONARY COMPRESSION — Each generation surpasses the last.

The insight: distillation is a WARM START, not a ceiling.
After distilling, continue on real text → surpass the teacher.
Then use the improved student as teacher for the next generation.

Generation 0: Teacher (Qwen3-0.6B, baseline)
Generation 1: Distill from teacher → continue on FineWeb → eval
Generation 2: Distill from Gen1 → continue on FineWeb → eval
Generation 3: Distill from Gen2 → continue on FineWeb → eval

If each generation improves, this is a compression machine that gets
BETTER over time. The model IMPROVES through compression.

This works BOTH DIRECTIONS:
- Compress: distill existing model into FRR
- Grow: continue training on real text, surpass the teacher
- Repeat: use the improved FRR as teacher for next generation

Nobody has this. Self-improving compression.
"""
import lib.unbuffered
import torch, sys, os, time, math, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
DISTILL_STEPS = 10000    # Phase 1: distill from teacher
CONTINUE_STEPS = 10000   # Phase 2: continue on real text
N_GENERATIONS = 3        # How many compress→grow cycles
USE_COMPILE = False      # torch.compile fails on Windows (Triton)
USE_AMP = True           # BF16 mixed precision

print("=" * 60)
print("EVOLUTIONARY COMPRESSION — Self-improving through generations")
print("=" * 60)

# Load original teacher
print("Loading original teacher (Qwen3-0.6B)...")
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
orig_teacher = MiniTransformer(config, device)
orig_teacher.load_weights(gd)
orig_teacher.embed_weight = orig_teacher.embed_weight.to(device)
if orig_teacher.lm_head is not None: orig_teacher.lm_head = orig_teacher.lm_head.to(device)
embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# Load real text data
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
USE_REAL = False
ds_iter = None
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_iter = iter(ds)
    USE_REAL = True
    print("FineWeb-Edu loaded!")
except:
    print("FineWeb failed. Using random tokens for Phase 2.")

def get_real_batch(batch_size=4, seq_len=64):
    global ds_iter
    if not USE_REAL:
        return torch.randint(100, 50000, (batch_size, seq_len + 1), device=device)
    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200: continue
                toks = tokenizer.encode(text, max_length=seq_len + 1, truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len + 1:
                    tokens_list.append(toks[:seq_len + 1])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)

def eval_vs_original(model_fn, n=200):
    """Always evaluate against the ORIGINAL teacher, not the current generation."""
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = orig_teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model_fn(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)


def eval_perplexity(model_fn, n_batches=50):
    """Evaluate real language modeling quality (not teacher agreement)."""
    total_loss = 0
    count = 0
    for _ in range(n_batches):
        batch = get_real_batch(batch_size=2, seq_len=64)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.no_grad():
            logits = model_fn(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        total_loss += loss.item()
        count += 1
    return math.exp(total_loss / count)


# ═══════════════════════════════════════════════════════════
# EVOLUTIONARY LOOP
# ═══════════════════════════════════════════════════════════

results = []
current_teacher_fn = lambda t: orig_teacher.forward(t, max_layers=28)
prev_model = None

for gen in range(N_GENERATIONS):
    print(f"\n{'='*60}")
    print(f"  GENERATION {gen + 1}/{N_GENERATIONS}")
    print(f"{'='*60}")

    # Build fresh FRR
    model = FractalModel(1024, 16, 4, 7, 151936, 1,
                         embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
    if USE_COMPILE and gen == 0:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  torch.compile enabled (2-3x speedup)")
        except Exception as e:
            print(f"  torch.compile failed: {e}, continuing without")

    # If we have a previous generation, initialize from it
    if prev_model is not None:
        print("  Warm-starting from previous generation's weights...")
        model.load_state_dict(prev_model.state_dict())

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)

    # PHASE 1: Distillation from current teacher
    print(f"\n  Phase 1: Distill from {'original teacher' if gen == 0 else f'Generation {gen} FRR'} ({DISTILL_STEPS} steps)")
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DISTILL_STEPS)
    t0 = time.time()

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    for step in range(DISTILL_STEPS):
        torch.manual_seed(step * 7 + gen * 100000)
        tokens = torch.randint(100, 50000, (8, 48), device=device)  # bigger batch + seq
        with torch.no_grad():
            tl = current_teacher_fn(tokens)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            sl = model(tokens)
            T = max(2.0, 5.0 * (1 - step / DISTILL_STEPS))
            loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                           reduction='batchmean') * T * T
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 5000 == 0:
            t1, t10 = eval_vs_original(model, n=50)
            print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

    t1, t10 = eval_vs_original(model, n=100)
    ppl = eval_perplexity(model) if USE_REAL else -1
    print(f"  Phase 1 DONE: T1={t1*100:.0f}% T10={t10*100:.0f}% PPL={ppl:.1f}")
    phase1_t10 = t10

    # PHASE 2: Continue on real text (break past teacher)
    print(f"\n  Phase 2: Independent learning on FineWeb-Edu ({CONTINUE_STEPS} steps)")
    for p in model.parameters():
        p.requires_grad = True
    all_params = list(model.parameters())
    opt2 = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, CONTINUE_STEPS)

    scaler2 = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    for step in range(CONTINUE_STEPS):
        batch = get_real_batch(batch_size=8, seq_len=64)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt2.zero_grad()
        scaler2.scale(loss).backward()
        scaler2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler2.step(opt2); scaler2.update(); sched2.step()

        if step % 5000 == 0:
            ppl_now = math.exp(min(loss.item(), 20))
            t1, t10 = eval_vs_original(model, n=50)
            print(f"    Step {step}: loss={loss.item():.4f} ppl={ppl_now:.1f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

    t1, t10 = eval_vs_original(model, n=200)
    ppl = eval_perplexity(model) if USE_REAL else -1
    print(f"  Phase 2 DONE: T1={t1*100:.0f}% T10={t10*100:.0f}% PPL={ppl:.1f}")

    improved = t10 > phase1_t10
    results.append({
        'generation': gen + 1,
        'phase1_t10': phase1_t10,
        'phase2_t10': t10,
        'phase2_t1': t1,
        'ppl': ppl,
        'improved': improved,
    })

    if improved:
        print(f"  >>> CEILING BROKEN! Phase 2 surpassed Phase 1: {t10*100:.0f}% > {phase1_t10*100:.0f}%")
    else:
        print(f"  --- Ceiling held. {t10*100:.0f}% <= {phase1_t10*100:.0f}%")

    # Use this generation's model as next teacher
    prev_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    current_teacher_fn = lambda t, m=model: m(t)
    prev_model = model


# ═══════════════════════════════════════════════════════════
# FINAL RESULTS
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"EVOLUTIONARY COMPRESSION RESULTS")
print(f"{'='*60}")
print(f"  {'Gen':<5} {'Distill T10':>12} {'After Real T10':>15} {'PPL':>8} {'Improved?':>10}")
print(f"  {'-'*55}")
for r in results:
    imp = "YES!" if r['improved'] else "no"
    print(f"  {r['generation']:<5} {r['phase1_t10']*100:>11.0f}% {r['phase2_t10']*100:>14.0f}% {r['ppl']:>8.1f} {imp:>10}")

if any(r['improved'] for r in results):
    print(f"\n  EVOLUTION WORKS. FRR improves through compression cycles.")
    print(f"  Self-improving compression is REAL.")
else:
    print(f"\n  Evolution didn't improve. Teacher ceiling is hard.")
    print(f"  Need a different approach to break past it.")

with open('evolve_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to evolve_results.json")
print("Done!")
