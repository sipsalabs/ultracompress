"""
FOCUSED LOSS TEST — What if we only learn what MATTERS?

Standard KL distillation matches ALL 151,936 token probabilities.
But 99.99% of tokens are irrelevant at any position.
The model wastes capacity on matching noise in the tail.

Focused loss: only penalize on teacher's top-K predictions.
This concentrates ALL learning on the tokens that actually matter.

If this breaks 60% real T10: the loss function was the bottleneck.
"""
import lib.unbuffered
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 15000

print("=" * 60)
print("FOCUSED LOSS: Only learn what matters (top-K tokens)")
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_batch(batch_size=4, seq_len=64):
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

def eval_real_text(model, n=100):
    t1, t10s = 0, []
    for trial in range(n):
        batch = get_batch(1, 32)
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

def focused_kl_loss(student_logits, teacher_logits, top_k=100, T=2.0):
    """KL divergence on only the teacher's top-K tokens per position."""
    B, S, V = teacher_logits.shape
    # Get teacher's top-K indices
    _, top_indices = teacher_logits.topk(top_k, dim=-1)  # (B, S, K)
    # Gather student and teacher logits for top-K only
    t_topk = teacher_logits.gather(-1, top_indices) / T  # (B, S, K)
    s_topk = student_logits.gather(-1, top_indices) / T  # (B, S, K)
    # KL on the focused distribution
    loss = F.kl_div(F.log_softmax(s_topk, dim=-1), F.softmax(t_topk, dim=-1),
                   reduction='batchmean') * T * T
    return loss

def train_and_eval(name, model, loss_fn, steps=STEPS):
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Trainable: {trainable:,}")
    print(f"{'='*60}")
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps):
        tokens = get_batch(4, 64)
        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=28)
        sl = model(tokens)
        T = max(2.0, 5.0 * (1 - step / steps))
        loss = loss_fn(sl, tl, T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 5000 == 0:
            t1, t10 = eval_real_text(model, n=50)
            print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")
    t1, t10 = eval_real_text(model, n=200)
    print(f"  FINAL: T1={t1*100:.0f}% T10={t10*100:.0f}%")
    return t1, t10

# Standard KL (baseline)
def standard_kl(sl, tl, T):
    return F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                   reduction='batchmean') * T * T

frr1 = FractalModel(1024, 16, 4, 7, 151936, 1,
                    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
t1_std, t10_std = train_and_eval("Standard KL (real text)", frr1, standard_kl)
del frr1; torch.cuda.empty_cache()

# Focused KL top-100
frr2 = FractalModel(1024, 16, 4, 7, 151936, 1,
                    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
t1_f100, t10_f100 = train_and_eval("Focused KL top-100 (real text)", frr2,
                                    lambda sl, tl, T: focused_kl_loss(sl, tl, 100, T))
del frr2; torch.cuda.empty_cache()

# Focused KL top-10
frr3 = FractalModel(1024, 16, 4, 7, 151936, 1,
                    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
t1_f10, t10_f10 = train_and_eval("Focused KL top-10 (real text)", frr3,
                                  lambda sl, tl, T: focused_kl_loss(sl, tl, 10, T))

print(f"\n{'='*60}")
print(f"RESULTS: Does focused loss break the barrier?")
print(f"{'='*60}")
print(f"  Standard KL (all 151K tokens): T1={t1_std*100:.0f}% T10={t10_std*100:.0f}%")
print(f"  Focused KL top-100:            T1={t1_f100*100:.0f}% T10={t10_f100*100:.0f}%")
print(f"  Focused KL top-10:             T1={t1_f10*100:.0f}% T10={t10_f10*100:.0f}%")
print("Done!")
