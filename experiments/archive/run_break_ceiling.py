"""
BREAK THE CEILING — Distill first, then train from scratch.

Phase 1: Distill from teacher (warm start, gets to ~67% fast)
Phase 2: Switch to real text next-token prediction (break past teacher)

This bridges old and new:
- Starts from existing model (distillation = compress direction)
- Continues learning independently (from-scratch = grow direction)
- The teacher gives a starting point, NOT a ceiling

Nobody has tested this specific two-phase approach on FRR.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

device = 'cuda'
PHASE1_STEPS = 15000  # Distillation warm-start
PHASE2_STEPS = 35000  # From-scratch continuation

print("=" * 60)
print("BREAK THE CEILING: Distill → then learn independently")
print("=" * 60)

# Load teacher for phase 1
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
embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# Load tokenizer + dataset for phase 2
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
USE_REAL = False
ds_iter = None
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_iter = iter(ds)
    USE_REAL = True
    print("FineWeb-Edu loaded for Phase 2!")
except:
    print("FineWeb failed. Phase 2 uses random tokens.")

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
                if len(text) < 200:
                    continue
                toks = tokenizer.encode(text, max_length=seq_len + 1, truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len + 1:
                    tokens_list.append(toks[:seq_len + 1])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)

def eval_vs_teacher(model, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        torch.manual_seed(trial * 13 + 9999)
        t = torch.randint(100, 50000, (1, 16), device=device)
        with torch.no_grad():
            tl = teacher.forward(t, max_layers=28)
            tp = tl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gl = model(t)
            gp = gl[0, -1].argmax().item()
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)

# Build FRR (all params trainable for phase 2)
model = FractalModel(1024, 16, 4, 7, 151936, 1,
                     embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

# PHASE 1: Distillation warm-start
print(f"\n{'='*60}")
print(f"PHASE 1: Distillation warm-start ({PHASE1_STEPS} steps)")
print(f"{'='*60}")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PHASE1_STEPS)
t0 = time.time()
for step in range(PHASE1_STEPS):
    torch.manual_seed(step * 7)
    tokens = torch.randint(100, 50000, (4, 32), device=device)
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=28)
    sl = model(tokens)
    T = max(2.0, 5.0 * (1 - step / PHASE1_STEPS))
    loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1), reduction='batchmean') * T * T
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 5000 == 0:
        t1, t10 = eval_vs_teacher(model, n=50)
        print(f"  Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

t1, t10 = eval_vs_teacher(model, n=100)
print(f"  Phase 1 FINAL: T1={t1*100:.0f}% T10={t10*100:.0f}%")
phase1_t10 = t10

# PHASE 2: Break past the teacher — real text, next-token prediction
print(f"\n{'='*60}")
print(f"PHASE 2: Independent learning ({PHASE2_STEPS} steps)")
print(f"Switching from KL-to-teacher to real next-token prediction")
print(f"{'='*60}")

# Unfreeze ALL params (including embeddings) for independent learning
for p in model.parameters():
    p.requires_grad = True

opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, PHASE2_STEPS)
for step in range(PHASE2_STEPS):
    batch = get_real_batch(batch_size=4, seq_len=64)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step(); sched2.step()
    if step % 5000 == 0:
        ppl = math.exp(min(loss.item(), 20))
        t1, t10 = eval_vs_teacher(model, n=50)
        print(f"  Step {step}: loss={loss.item():.4f} ppl={ppl:.1f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")

t1, t10 = eval_vs_teacher(model, n=200)
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"  Phase 1 (distillation): T10={phase1_t10*100:.0f}%")
print(f"  Phase 2 (independent):  T10={t10*100:.0f}%")
if t10 > phase1_t10:
    print(f"  CEILING BROKEN! Independent learning pushed PAST teacher!")
else:
    print(f"  Ceiling held. Independent learning didn't help.")
print("Done!")
