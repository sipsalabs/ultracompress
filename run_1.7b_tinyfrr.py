"""
TinyFRR compression-ratio sweep.
Projects teacher-width (2048) -> h_inner -> 2048, runs FractalModel at h_inner.
Trains pure KL from scratch, 8K steps each. Records best last-T10.

Builds the Pareto curve: compression ratio vs last-T10.
Known points:
  FRR-2048 (29.4M, 28x):   68.5% (from prior 100K run)
  FlowODE-286x (5.4M):     57% (abandoned)
  FlowODE-970x (1.6M):     60%

This sweep will fill: 1024, 768, 512, 384 inner dims.
"""
import lib.unbuffered
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, required=True, help='Inner hidden dim')
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--tag', type=str, default=None)
args = ap.parse_args()

DEVICE = 'cuda:0'
SEQ_LEN = 64
BATCH = 4
STEPS = args.steps
LR_MAX = 5e-4
LR_MIN = 5e-5
KL_TEMP = 2.0
EVAL_EVERY = 1000
CKPT_EVERY = 1000
N_TEACHER_LAYERS = 28
H_INNER = args.h
TAG = args.tag or f'h{H_INNER}'
CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
os.makedirs(CKPT_DIR, exist_ok=True)
LATEST = f'{CKPT_DIR}/latest.pt'
BEST = f'{CKPT_DIR}/best.pt'

print(f"TinyFRR sweep: inner={H_INNER}  steps={STEPS}  tag={TAG}")

# ========== Teacher ==========
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
H_OUTER = gd['token_embd.weight'].shape[1]  # 2048
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=H_OUTER, intermediate_size=H_OUTER*3,
                  vocab_size=vocab_size, head_dim=H_OUTER//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w_outer = gd['token_embd.weight'].to(DEVICE)
norm_w_outer = gd['output_norm.weight'].to(DEVICE)
lm_head_w_outer = gd['output.weight'].to(DEVICE)
del gd

# Choose n_heads for inner dim: need H_INNER divisible by n_heads
candidate_heads = [16, 8, 12, 4]
n_heads_inner = next((h for h in candidate_heads if H_INNER % h == 0), 4)
print(f"  H_OUTER={H_OUTER}  H_INNER={H_INNER}  n_heads_inner={n_heads_inner}")

# ========== TinyFRR wrapper ==========
class TinyFRR(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        # Kaiming init for projections
        nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_out.weight, a=math.sqrt(5))
        # Inner FRR operates at h_inner. Embeddings are external; we don't share teacher embeds here
        # but we DO use teacher lm_head (via h_outer space).
        # Build frr without embed/lm_head tied; use as hidden-space processor.
        # Trick: pass dummy embed/lm_head at h_inner that won't be used since we bypass them.
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7,
            vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        # Freeze unused inner embed/lm_head (we bypass them in forward)
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        # Use teacher embed/lm_head at outer dim (not registered as parameters)
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x_outer = self.embed(tokens).float()  # [B,S,H_outer]
        x = self.proj_in(x_outer)  # [B,S,H_inner]
        # Run inner FRR block by block (replicate forward loop)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        # Project back
        x_outer = self.proj_out(x)  # [B,S,H_outer]
        x_outer = self.norm_outer(x_outer)
        logits = F.linear(x_outer, self.lm_head_w)
        return logits

student = TinyFRR(H_OUTER, H_INNER, n_heads_inner, vocab_size,
                  embed_w_outer, lm_head_w_outer, norm_w_outer).to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
teacher_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER
print(f"  trainable = {trainable/1e6:.2f}M  total = {total/1e6:.2f}M  teacher = {teacher_params/1e6:.1f}M  compression = {teacher_params/trainable:.1f}x")

# ========== Data ==========
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)

# ========== Train ==========
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR_MAX, weight_decay=1e-4, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler('cuda')

step0 = 0; best = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    step0 = ck['step']; best = ck.get('best', 0.0)
    print(f"  Resumed step={step0} best={best:.4f}")

@torch.no_grad()
def evaluate(n=100):
    student.eval()
    all_t10 = last_t10 = 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            all_t10 += len(tt & st) / 10
        tt = set(tl[0, -1].topk(10).indices.tolist())
        st = set(sl[0, -1].topk(10).indices.tolist())
        last_t10 += len(tt & st) / 10
    student.train()
    return all_t10/(n*SEQ_LEN), last_t10/n

t0 = time.time()
for step in range(step0, STEPS):
    prog = step / STEPS
    lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups: g['lr'] = lr

    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
    toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts]).to(DEVICE)
    with torch.no_grad():
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        sl = student(toks)
        s_log = F.log_softmax(sl / KL_TEMP, dim=-1)
        t_soft = F.softmax(tl / KL_TEMP, dim=-1)
        loss = (t_soft * (t_soft.clamp_min(1e-12).log() - s_log)).sum(-1).mean() * (KL_TEMP ** 2)

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update()

    if step % 100 == 0:
        print(f"  [{TAG}] step={step:5d}  loss={loss.item():.4f}  ({int(time.time()-t0)}s)")

    if step > 0 and step % EVAL_EVERY == 0:
        a10, l10 = evaluate(100)
        flag = ''
        if l10 > best:
            best = l10
            torch.save({'state_dict': student.state_dict(), 'step': step, 'best': best,
                        'h_inner': H_INNER, 'n_heads_inner': n_heads_inner, 'trainable': trainable}, BEST)
            flag = ' *** NEW BEST ***'
        print(f"  EVAL step={step:5d}  all-T10={a10*100:.1f}%  last-T10={l10*100:.1f}%  best={best*100:.1f}%{flag}")

    if step > 0 and step % CKPT_EVERY == 0:
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step+1, 'best': best}, LATEST)

# Final eval + save
a10, l10 = evaluate(200)
if l10 > best:
    best = l10
    torch.save({'state_dict': student.state_dict(), 'step': STEPS, 'best': best,
                'h_inner': H_INNER, 'n_heads_inner': n_heads_inner, 'trainable': trainable}, BEST)
print(f"\nDONE {TAG}: best last-T10 = {best*100:.2f}%  trainable={trainable/1e6:.2f}M  compression={teacher_params/trainable:.0f}x")
# Append to sweep results
with open('tinyfrr_sweep_results.txt', 'a') as f:
    f.write(f"{TAG}\th_inner={H_INNER}\ttrainable={trainable}\tcompression={teacher_params/trainable:.1f}x\tbest_last_T10={best*100:.2f}%\tsteps={STEPS}\n")
