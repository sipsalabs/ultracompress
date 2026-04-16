"""
HMK+GT hybrid: Ground-truth-aware training.
- Where teacher entropy < 5 bits: KL to teacher (teacher knows, mimic it)
- Where teacher entropy >= 5 bits: cross-entropy on REAL next token (teacher unsure, learn from data)

Rationale: teacher is only 14.92% top-1 on ground truth. On high-entropy positions
the teacher is guessing, so mimicking it caps us at the teacher's confused distribution.
Using the real next token there teaches capabilities BEYOND the teacher.

Fresh FRR init, 50K steps. Launches AFTER hmk finishes (watchdog).
"""
import lib.unbuffered
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

DEVICE = 'cuda:0'
SEQ_LEN = 64
BATCH = 4
STEPS = 50000
LR_MAX = 5e-4
LR_MIN = 5e-5
KL_TEMP = 2.0
ENT_THRESHOLD = 5.0  # bits; >= this -> use GT-CE instead of KL
EVAL_EVERY = 2500
CKPT_EVERY = 2500
N_TEACHER_LAYERS = 28
CKPT_DIR = 'checkpoints_1.7b_hmkgt'
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST = f'{CKPT_DIR}/hmkgt_latest.pt'
BEST = f'{CKPT_DIR}/hmkgt_best.pt'

print("=" * 72)
print("HMK+GT hybrid: teacher-confident -> KL, teacher-unsure -> real-token CE")
print(f"  threshold={ENT_THRESHOLD}b  steps={STEPS}  lr={LR_MAX}")
print("=" * 72)

# ========== Teacher ==========
print("Loading teacher...")
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

# ========== Data ==========
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
print(f"Data: {all_tokens.numel()/1e6:.1f}M tokens")

# ========== Student (fresh init) ==========
student = FractalModel(
    hidden_dim=hidden, n_heads=16,
    n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
teacher_params = N_TEACHER_LAYERS * 4 * hidden * hidden
print(f"Student trainable: {trainable/1e6:.2f}M  compression: {teacher_params/trainable:.1f}x")

opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR_MAX, weight_decay=1e-4, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler('cuda')

# ========== Resume ==========
step0 = 0; best = 0.0
if os.path.exists(LATEST):
    print(f"Resuming {LATEST}")
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    step0 = ck['step']; best = ck.get('best', 0.0)
    print(f"  resumed step={step0} best={best:.4f}")

# ========== Eval ==========
@torch.no_grad()
def evaluate(n=100):
    student.eval()
    all_t10 = last_t10 = conf_t10 = 0
    conf_n = 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        t_prob = F.softmax(tl[0], dim=-1)
        ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(-1) / 0.6931
        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            h = len(tt & st) / 10
            all_t10 += h
            if ent[pos].item() < 2:
                conf_t10 += h; conf_n += 1
        tt = set(tl[0, -1].topk(10).indices.tolist())
        st = set(sl[0, -1].topk(10).indices.tolist())
        last_t10 += len(tt & st) / 10
    student.train()
    return all_t10/(n*SEQ_LEN), last_t10/n, (conf_t10/max(conf_n,1) if conf_n else 0), conf_n

# ========== Train ==========
print(f"\nTraining from step {step0}...")
t0 = time.time()
for step in range(step0, STEPS):
    # lr schedule
    prog = step / STEPS
    lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups: g['lr'] = lr

    # sample
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (BATCH,))
    toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts]).to(DEVICE)
    # targets = next token (for GT-CE)
    next_toks = torch.stack([all_tokens[s+1:s+SEQ_LEN+1].long() for s in starts]).to(DEVICE)

    with torch.no_grad():
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)  # [B,S,V]
        t_prob = F.softmax(tl, dim=-1)
        ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(-1) / 0.6931  # bits [B,S]
        unsure_mask = (ent >= ENT_THRESHOLD)  # True -> use GT-CE
        conf_mask = ~unsure_mask              # True -> use KL

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        sl = student(toks)  # [B,S,V]

        # KL path (on teacher-confident positions)
        s_log = F.log_softmax(sl / KL_TEMP, dim=-1)
        t_soft = F.softmax(tl / KL_TEMP, dim=-1)
        kl_per = (t_soft * (t_soft.clamp_min(1e-12).log() - s_log)).sum(-1) * (KL_TEMP ** 2)  # [B,S]
        kl_loss = (kl_per * conf_mask.float()).sum() / conf_mask.float().sum().clamp_min(1)

        # GT-CE path (on teacher-unsure positions)
        ce_per = F.cross_entropy(sl.reshape(-1, vocab_size), next_toks.reshape(-1), reduction='none').reshape(BATCH, SEQ_LEN)
        ce_loss = (ce_per * unsure_mask.float()).sum() / unsure_mask.float().sum().clamp_min(1)

        # Combined: weight each by fraction of positions it covers (natural)
        conf_frac = conf_mask.float().mean()
        loss = conf_frac * kl_loss + (1 - conf_frac) * ce_loss

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update()

    if step % 100 == 0:
        cf = conf_mask.float().mean().item()
        print(f"  [hbgt] step={step:5d}  loss={loss.item():.4f}  kl={kl_loss.item():.3f}  ce={ce_loss.item():.3f}  conf={cf*100:.1f}%  ({int(time.time()-t0)}s)")

    if step > 0 and step % EVAL_EVERY == 0:
        a10, l10, c10, cn = evaluate(100)
        flag = ''
        if l10 > best:
            best = l10
            torch.save({'state_dict': student.state_dict(), 'step': step, 'best': best,
                        'ff_mult': 1}, BEST)
            flag = ' *** NEW BEST ***'
        print(f"  EVAL step={step:5d}  all-T10={a10*100:.1f}%  last-T10={l10*100:.1f}%  conf-T10={c10*100:.1f}%(n={cn})  best={best*100:.1f}%{flag}")

    if step > 0 and step % CKPT_EVERY == 0:
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step+1, 'best': best, 'ff_mult': 1}, LATEST)

# Final
torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
            'step': STEPS, 'best': best, 'ff_mult': 1}, LATEST)
print(f"\nDONE. best last-T10={best*100:.2f}%")
