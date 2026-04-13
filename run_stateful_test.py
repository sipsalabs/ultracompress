"""
STATEFUL FRR TEST — Does memory across applications break the 67% barrier?

Head-to-head: Standard FRR vs Stateful FRR.
Same teacher, same training, same eval.
If stateful > 57% T10 at 15K steps (standard's ceiling), the barrier is broken.
"""
import lib.unbuffered
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from ultracompress.stateful_frr import StatefulFRR
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 15000

print("=" * 60)
print("STATEFUL FRR: Does memory break the barrier?")
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

# Use REAL TEXT (proven better than random tokens)
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


def eval_vs_teacher(model, n=100):
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


def train_and_eval(name, model, steps=STEPS):
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
        loss = F.kl_div(F.log_softmax(sl / T, dim=-1), F.softmax(tl / T, dim=-1),
                       reduction='batchmean') * T * T
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 5000 == 0:
            t1, t10 = eval_vs_teacher(model, n=50)
            print(f"    Step {step}: loss={loss.item():.4f} T1={t1*100:.0f}% T10={t10*100:.0f}% ({time.time()-t0:.0f}s)")
    t1, t10 = eval_vs_teacher(model, n=200)
    print(f"  FINAL: T1={t1*100:.0f}% T10={t10*100:.0f}%")
    return t1, t10, trainable


# Standard FRR (baseline)
frr = FractalModel(1024, 16, 4, 7, 151936, 1,
                   embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
t1_std, t10_std, p_std = train_and_eval("Standard FRR (real text)", frr)
del frr; torch.cuda.empty_cache()

# Stateful FRR (the test)
stateful = StatefulFRR(1024, 16, 4, 7, 151936, state_dim=64,
                       embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
t1_state, t10_state, p_state = train_and_eval("Stateful FRR (real text, state_dim=64)", stateful)

print(f"\n{'='*60}")
print(f"RESULTS: Does state break the barrier?")
print(f"{'='*60}")
print(f"  Standard FRR: T1={t1_std*100:.0f}% T10={t10_std*100:.0f}% ({p_std:,} params)")
print(f"  Stateful FRR: T1={t1_state*100:.0f}% T10={t10_state*100:.0f}% ({p_state:,} params)")
diff = t10_state - t10_std
if diff > 0.03:
    print(f"  >>> STATE BREAKS BARRIER! +{diff*100:.0f}% T10")
elif diff > 0:
    print(f"  State helps slightly: +{diff*100:.0f}% T10")
else:
    print(f"  State didn't help. The barrier is elsewhere.")
print("Done!")
