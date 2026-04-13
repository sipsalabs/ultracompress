"""
1.7B REAL TEXT DISTILLATION — The path to 90%+ HellaSwag.

What we proved tonight:
- Real text +13% T10 over random tokens on 0.6B
- 100K FRR = 83.3% HellaSwag retention (T10 misleadingly says 65%)
- FRR BEATS teacher on WikiText-2 PPL (better generalization!)
- 1.7B > 0.6B by +2% T10 on random tokens

So: 1.7B + real text + 50K steps should push HellaSwag retention to 85-90%+.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 50000

print("=" * 60)
print("1.7B REAL TEXT DISTILLATION — Path to 90%+ HellaSwag")
print("=" * 60)

# Load 1.7B teacher
print("Loading Qwen3-1.7B teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
               'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
               'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
               'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
               'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
               'mlp.down_proj.weight': 'ffn_down.weight'}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
n_layers = 28
for li in range(n_layers):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()

# Detect 1.7B config
hidden = gd['token_embd.weight'].shape[1]
n_heads = 16
head_dim = hidden // n_heads
vocab_size = gd['token_embd.weight'].shape[0]
print(f"  Hidden: {hidden}, Heads: {n_heads}, HeadDim: {head_dim}, Vocab: {vocab_size}")

config = ModelConfig(n_layers=n_layers, n_heads=n_heads, n_kv_heads=8, hidden_size=hidden,
                     intermediate_size=hidden*3, vocab_size=vocab_size, head_dim=head_dim)
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None: teacher.lm_head = teacher.lm_head.to(device)
embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# Real text
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_real_batch(batch_size=4, seq_len=64):
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

# Build FRR for 1.7B
model = FractalModel(hidden, n_heads, 4, 7, vocab_size, 1,
                     embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Estimate teacher layer params from known architecture
total_teacher = n_layers * (4 * hidden * hidden + 3 * hidden * hidden * 3)  # attn + ffn
compression = total_teacher / trainable
print(f"FRR: {trainable:,} ({compression:.1f}x)")

# Train with real text KL distillation
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()
best_t10 = 0

def eval_vs_teacher(n=100):
    t1, t10s = 0, []
    for trial in range(n):
        batch = get_real_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=n_layers)
            gl = model(batch)
            tp = tl[0, -1].argmax().item()
            gp = gl[0, -1].argmax().item()
            tt10 = set(tl[0, -1].topk(10).indices.tolist())
            gt10 = set(gl[0, -1].topk(10).indices.tolist())
            if tp == gp: t1 += 1
            t10s.append(len(tt10 & gt10) / 10)
    return t1 / n, sum(t10s) / len(t10s)

print(f"Training {STEPS} steps with REAL TEXT distillation...")
for step in range(STEPS):
    tokens = get_real_batch(4, 64)
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=n_layers)
    sl = model(tokens)
    T = max(2.0, 5.0 * (1 - step / STEPS))
    loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                   reduction='batchmean') * T * T
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()

    if step % 10000 == 0:
        t1, t10 = eval_vs_teacher(n=100)
        elapsed = time.time() - t0
        new_best = " <-- NEW BEST" if t10 > best_t10 else ""
        if t10 > best_t10: best_t10 = t10
        print(f"  Step {step}/{STEPS}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({elapsed:.0f}s){new_best}")

t1, t10 = eval_vs_teacher(n=200)
print(f"\nFINAL (200 samples): T1={t1*100:.0f}% T10={t10*100:.0f}%")
print(f"Compression: {compression:.1f}x")
print("Done!")
