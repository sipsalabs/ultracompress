"""
Confidence-Weighted FRR (CWF) — direct attack on the proxy-ceiling finding.

Ground-truth diagnostic revealed:
  - Teacher confident (entropy<2): 3.6% of positions, 62.8% GT top-1
  - Teacher unsure (entropy>5):   72% of positions, 9.6% GT top-1 (near random)
  - KL gradient is weighted by teacher prob mass: ~3.6% of signal comes from
    positions where teacher actually knows, 72% from noise. Student trained
    almost entirely on positions where teacher is guessing.

HYPOTHESIS: reweighting the per-POSITION loss by (1/teacher_entropy) concentrates
learning on positions where teacher is competent → student learns ACTUAL
capability instead of approximating teacher-guessing.

If this breaks the 68% proxy ceiling, it validates the diagnosis. If not, 68%
is a real architectural ceiling.

This is the most consequential test of the session. NOT a new architecture —
same FRR base, just the LOSS.

Base:    checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt  (already 65.6% T10)
Loss:    L = Σ_i w_i · KL(s_i || t_i)   with  w_i = 1 / (H(t_i) + 1)
         normalized so Σw_i = N (same gradient scale as plain KL)
Steps:   5000 continued fine-tune
LR:      1e-4 (gentle, we're nudging)
"""
import lib.unbuffered
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

DEVICE = 'cuda:0'
TOTAL_STEPS = 5_000
LR = 1e-4
KL_TEMP = 2.0
BATCH_SIZE = 4
SEQ_LEN = 64
EVAL_INTERVAL = 500
PATIENCE = 2000
N_TEACHER_LAYERS = 28

BASE_CKPT = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
OUT_DIR = 'checkpoints_1.7b_cwf'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("Confidence-Weighted FRR — test of proxy-ceiling hypothesis")
print(f"  base: {BASE_CKPT}")
print(f"  loss: w_i = 1/(H(t_i)+1) · KL(s||t) per position")
print("=" * 72)

# ── Teacher ──
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd

hidden = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=hidden, intermediate_size=hidden*3,
                  vocab_size=vocab_size, head_dim=hidden//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Student (FRR from checkpoint) ──
print(f"\nLoading student {BASE_CKPT}...")
ckpt = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
# inspect to get config
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    ff_mult = ckpt.get('ff_mult', 1)
else:
    sd = ckpt
    ff_mult = 1
print(f"  ff_mult={ff_mult}, loaded {len(sd)} tensors")

student = FractalModel(
    hidden_dim=hidden, n_heads=16,
    n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=ff_mult,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load state dict (may have minor key differences)
missing, unexpected = student.load_state_dict(sd, strict=False)
print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
if missing[:3]: print(f"  missing sample: {missing[:3]}")
if unexpected[:3]: print(f"  unexpected sample: {unexpected[:3]}")

# Inject embedding/norm/lm_head if model uses them shared with teacher
# (FractalModel likely has its own — keep as-is since checkpoint set them)
total = sum(p.numel() for p in student.parameters())
teacher_layer_p = N_TEACHER_LAYERS * (4*hidden*hidden + 3*hidden*hidden*3)
print(f"  student params = {total:,} ({total/1e6:.2f}M)  compression = {teacher_layer_p/total:.0f}x")

# ── Data ──
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
data_ptr = [0]
def batch():
    e = data_ptr[0] + BATCH_SIZE * SEQ_LEN
    if e > all_tokens.numel() - 1:
        data_ptr[0] = 0; e = BATCH_SIZE * SEQ_LEN
    toks = all_tokens[data_ptr[0]:e].long().reshape(BATCH_SIZE, SEQ_LEN).to(DEVICE)
    data_ptr[0] = e
    return toks

@torch.no_grad()
def eval_all(n=50):
    student.eval(); t1=t10=0; nt=0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            t1 += int(st[0] == tt[0])
            t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
            nt += 1
    student.train()
    return t1/nt, t10/nt

@torch.no_grad()
def eval_last(n=200):
    student.eval(); t1=t10=0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        t1 += int(st[0] == tt[0])
        t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
    student.train()
    return t1/n, t10/n

# Stratified eval: T10 on LOW-entropy positions only (where teacher actually knows)
@torch.no_grad()
def eval_stratified(n=100):
    student.eval()
    buckets = {'conf(<2)': [], 'mid(2-5)': [], 'unsure(>5)': []}
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        t_prob = F.softmax(tl[0], dim=-1)
        ent = -(t_prob * (t_prob.clamp_min(1e-12)).log()).sum(-1) / 0.6931  # bits
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            ov = len(set(tt.tolist()) & set(st.tolist())) / 10
            e = ent[pos].item()
            if e < 2: buckets['conf(<2)'].append(ov)
            elif e < 5: buckets['mid(2-5)'].append(ov)
            else: buckets['unsure(>5)'].append(ov)
    student.train()
    return {k: (sum(v)/len(v) if v else 0.0, len(v)) for k, v in buckets.items()}

print("\nBaseline:")
t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
strat = eval_stratified(100)
print(f"  all-pos T10={t10*100:.1f}%  last-tok T10={t10l*100:.1f}%")
for k, (v, n) in strat.items():
    print(f"  stratified {k}:  T10={v*100:.1f}%  (n={n})")

# ── Training ──
opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS, eta_min=LR*0.1)

print(f"\n{'='*72}\nTRAINING: confidence-weighted KL\n{'='*72}")
best = t10l; best_step = -1; since_best = 0
t0 = time.time()
losses = []

for step in range(TOTAL_STEPS):
    toks = batch()
    with torch.no_grad():
        t_logits = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        # Per-position entropy in bits
        t_soft_prob = F.softmax(t_logits, dim=-1)
        t_entropy = -(t_soft_prob * (t_soft_prob.clamp_min(1e-12)).log()).sum(-1) / 0.6931
        # Weights: inverse entropy, normalize to N
        w = 1.0 / (t_entropy + 1.0)       # (B, T)
        w = w * (w.numel() / w.sum())     # normalize: mean(w)=1
        w = w.unsqueeze(-1)               # (B, T, 1)
        t_soft = F.log_softmax(t_logits / KL_TEMP, -1)

    s_logits = student(toks)
    s_logsoft = F.log_softmax(s_logits / KL_TEMP, -1)

    # Per-position KL
    kl_per_pos = (t_soft.exp() * (t_soft - s_logsoft)).sum(-1)    # (B, T)
    loss = (w.squeeze(-1) * kl_per_pos).mean() * (KL_TEMP ** 2)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_all(50); t1l, t10l = eval_last(200)
        elapsed = time.time() - t0
        avg = sum(losses[-100:]) / min(len(losses), 100)
        mark = ""
        if t10l > best:
            best = t10l; best_step = step; since_best = 0
            mark = " *** NEW BEST ***"
            torch.save(student.state_dict(), f'{OUT_DIR}/cwf_best.pt')
        else:
            since_best += EVAL_INTERVAL
        print(f"  Step {step:>5d}/{TOTAL_STEPS}  wKL={avg:.4f}  all-T10={t10*100:.1f}%  "
              f"last-T10={t10l*100:.1f}%  best={best*100:.1f}%@{best_step}  ({elapsed:.0f}s){mark}")
        if step % 1000 == 0 and step > 0:
            strat = eval_stratified(100)
            for k, (v, n) in strat.items():
                print(f"      stratified {k}:  T10={v*100:.1f}%  (n={n})")
        if since_best >= PATIENCE:
            print(f"  Early stop @ step {step}")
            break

# Final stratified
print("\nFinal stratified eval (300 samples):")
strat = eval_stratified(300)
for k, (v, n) in strat.items():
    print(f"  {k}:  T10={v*100:.2f}%  (n={n})")

print(f"\n{'='*72}")
print(f"CWF DONE  best last-T10 = {best*100:.2f}% @ step {best_step}")
print(f"  baseline FRR 100K: 65.6%   pure_kl record (28x): 68.23%")
print(f"  if >68%: confidence weighting breaks proxy ceiling → validates diagnosis")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
print(f"{'='*72}")
