"""
Hard-Masked KL (HMK) — from scratch, 50K steps.

Ground-truth diagnostic showed:
  - 72% of positions have teacher entropy > 5 bits (teacher is guessing)
  - 3.6% of positions have entropy < 2 bits (teacher knows)
  - Standard KL distributes gradient by teacher prob-mass ≈ by confidence
    BUT still lets noise positions dilute the signal

CWF (previous): reweight by 1/(H+1). Gave +1.5pp conf-T1 on fine-tune.
HMK (this):     HARD MASK positions with H(t) > 5 bits. Zero gradient there.
                Forces student to ONLY learn from teacher-confident positions.

Training from scratch (random init) instead of fine-tuning, because:
  - Existing FRR-100K already converged to proxy-ceiling distribution
  - Can't unlearn the "match the noise" behavior from KL fine-tune
  - Fresh init + masked signal = learns ONLY real capability

Config:
  - FRR (same arch as baseline: 1 shared block, 4 scales × 7 iters = 28 layers)
  - 50K steps, cosine LR 5e-4 -> 5e-5
  - KL T=2.0
  - Hard mask: positions with teacher_entropy >= 5 bits contribute 0 loss
  - Auto-save every 2500 steps, save best by last-tok T10
  - Eval every 2500 steps (100 samples) to not lose too much time to eval
  - Final eval stratified on 500 samples
"""
import lib.unbuffered
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

DEVICE = 'cuda:0'
TOTAL_STEPS = 50_000
LR = 5e-4
KL_TEMP = 2.0
BATCH_SIZE = 4
SEQ_LEN = 64
ENTROPY_MASK_BITS = 5.0      # mask positions with H(t) >= 5 bits
EVAL_INTERVAL = 2500
CKPT_INTERVAL = 2500
N_TEACHER_LAYERS = 28

OUT_DIR = 'checkpoints_1.7b_hmk'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 72)
print("HMK — Hard-Masked KL from scratch, 50K steps")
print(f"  mask: zero gradient where teacher H >= {ENTROPY_MASK_BITS} bits")
print(f"  expected kept positions: ~28% (72% masked, matching GT diagnostic)")
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

# ── Student: fresh FRR init ──
print("\nInitializing fresh FRR student...")
student = FractalModel(
    hidden_dim=hidden, n_heads=16,
    n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)
total = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"  trainable params = {total:,} ({total/1e6:.2f}M)")

# Resume if checkpoint exists
resume_step = 0
resume_path = f'{OUT_DIR}/hmk_latest.pt'
if os.path.exists(resume_path):
    try:
        ck = torch.load(resume_path, weights_only=False, map_location=DEVICE)
        if isinstance(ck, dict) and 'state_dict' in ck:
            student.load_state_dict(ck['state_dict'], strict=False)
            resume_step = ck.get('step', 0)
            print(f"  RESUMED from step {resume_step}")
    except Exception as e:
        print(f"  resume failed: {e}  (starting fresh)")

# ── Data ──
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
data_ptr = [resume_step * BATCH_SIZE * SEQ_LEN % (all_tokens.numel() - BATCH_SIZE * SEQ_LEN)]
def batch():
    e = data_ptr[0] + BATCH_SIZE * SEQ_LEN
    if e > all_tokens.numel() - 1:
        data_ptr[0] = 0; e = BATCH_SIZE * SEQ_LEN
    toks = all_tokens[data_ptr[0]:e].long().reshape(BATCH_SIZE, SEQ_LEN).to(DEVICE)
    data_ptr[0] = e
    return toks

@torch.no_grad()
def eval_suite(n_all=50, n_last=200):
    student.eval()
    t1=t10=0; nt=0
    for _ in range(n_all):
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
    all_t1 = t1/nt; all_t10 = t10/nt

    lt1=lt10=0
    for _ in range(n_last):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        lt1 += int(st[0] == tt[0])
        lt10 += len(set(tt.tolist()) & set(st.tolist())) / 10
    student.train()
    return all_t1, all_t10, lt1/n_last, lt10/n_last

# ── Optimizer ──
opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS, eta_min=LR * 0.1)
for _ in range(resume_step):
    sched.step()

print(f"\n{'='*72}\nTRAINING: hard-masked KL (mask H >= {ENTROPY_MASK_BITS} bits)\n{'='*72}")
print(f"  starting at step {resume_step}")
if resume_step == 0:
    a1, a10, l1, l10 = eval_suite(30, 50)
    print(f"  Baseline (random init): all-T10={a10*100:.1f}%  last-T10={l10*100:.1f}%")

best = 0.0; best_step = -1
t0 = time.time()
losses = []
mask_rates = []

for step in range(resume_step, TOTAL_STEPS):
    toks = batch()
    with torch.no_grad():
        t_logits = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        t_prob = F.softmax(t_logits, dim=-1)
        t_entropy_bits = -(t_prob * (t_prob.clamp_min(1e-12)).log()).sum(-1) / 0.6931   # (B, T)
        # Hard mask: keep only entropy < threshold
        keep_mask = (t_entropy_bits < ENTROPY_MASK_BITS).float()   # (B, T)
        n_kept = keep_mask.sum().item()
        total_pos = keep_mask.numel()
        mask_rates.append(n_kept / total_pos)
        t_soft = F.log_softmax(t_logits / KL_TEMP, -1)

    s_logits = student(toks)
    s_logsoft = F.log_softmax(s_logits / KL_TEMP, -1)

    kl_per_pos = (t_soft.exp() * (t_soft - s_logsoft)).sum(-1)   # (B, T)
    masked_kl = kl_per_pos * keep_mask
    # Normalize by actual kept positions; if batch is all-masked, skip
    if n_kept > 0:
        loss = masked_kl.sum() / n_kept * (KL_TEMP ** 2)
    else:
        # extremely rare: just use regular KL
        loss = kl_per_pos.mean() * (KL_TEMP ** 2)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item())

    if step % 100 == 0 and step > resume_step:
        # terse heartbeat
        elapsed = time.time() - t0
        kr = sum(mask_rates[-100:]) / min(len(mask_rates), 100)
        avg = sum(losses[-100:]) / min(len(losses), 100)
        if step % 500 == 0:
            print(f"  [hb] step={step}  loss={avg:.4f}  keep={kr*100:.1f}%  ({elapsed:.0f}s)")

    if step % EVAL_INTERVAL == 0 and step > resume_step:
        a1, a10, l1, l10 = eval_suite(50, 200)
        elapsed = time.time() - t0
        avg = sum(losses[-100:]) / min(len(losses), 100)
        kr = sum(mask_rates[-500:]) / min(len(mask_rates), 500)
        mark = ""
        if l10 > best:
            best = l10; best_step = step
            mark = " *** NEW BEST ***"
            torch.save({'state_dict': student.state_dict(), 'step': step, 'ff_mult': 1},
                       f'{OUT_DIR}/hmk_best.pt')
        print(f"  EVAL step={step:>6d}  KL={avg:.4f}  keep={kr*100:.1f}%  "
              f"all-T10={a10*100:.1f}%  last-T10={l10*100:.1f}%  "
              f"best={best*100:.1f}%@{best_step}  ({elapsed/60:.1f}min){mark}", flush=True)

    if step % CKPT_INTERVAL == 0 and step > resume_step:
        torch.save({'state_dict': student.state_dict(), 'step': step, 'ff_mult': 1},
                   f'{OUT_DIR}/hmk_latest.pt')

# Final
torch.save({'state_dict': student.state_dict(), 'step': TOTAL_STEPS, 'ff_mult': 1},
           f'{OUT_DIR}/hmk_final.pt')

a1, a10, l1, l10 = eval_suite(100, 500)
print(f"\n{'='*72}")
print(f"HMK DONE  {TOTAL_STEPS} steps")
print(f"  final: all-T10={a10*100:.2f}%  last-T10={l10*100:.2f}%")
print(f"  best:  last-T10={best*100:.2f}% @ step {best_step}")
print(f"  avg kept: {sum(mask_rates)/len(mask_rates)*100:.1f}%")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
print(f"  record (pure_kl 28x): 68.23%")
print(f"{'='*72}")
