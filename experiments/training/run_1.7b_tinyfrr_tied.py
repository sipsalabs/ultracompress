"""
TinyFRR-Tied: proj_out is proj_in.T (tied). Halves projection parameters.
For h=128: trainable drops from 640K to ~380K, compression increases ~1.7x.
Also experiments with SHARED proj across layers (already true for a single block).

If quality stays high, we can push deeper: smaller h with same param budget.
"""
import lib.unbuffered
import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, required=True)
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--tag', type=str, default=None)
args = ap.parse_args()

STEPS = args.steps
BATCH = 4
SEQ_LEN = 64
LR = 5e-4
H_INNER = args.h
TAG = args.tag or f'h{H_INNER}_tied'
N_TEACHER_LAYERS = 28
DEVICE = 'cuda:0'
TEMP = 2.0

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"TinyFRR-Tied: inner={H_INNER}  steps={STEPS}  tag={TAG}")

# ==================== Teacher ====================
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
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()
del wd
H_OUTER = gd['token_embd.weight'].shape[1]
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

candidate_heads = [16, 8, 12, 4]
n_heads_inner = next((h for h in candidate_heads if H_INNER % h == 0), 4)
print(f"  H_OUTER={H_OUTER}  H_INNER={H_INNER}  n_heads_inner={n_heads_inner}")


# ==================== Student with tied projections ====================
class TinyFRRTied(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        # Single projection matrix; use W for proj_in and W.T for proj_out
        self.proj = nn.Parameter(torch.empty(h_inner, h_outer))
        nn.init.kaiming_uniform_(self.proj, a=math.sqrt(5))
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7,
            vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x_outer = self.embed(tokens).float()
        # proj_in: x @ proj.T  (out-dim h_inner)
        x = F.linear(x_outer, self.proj)  # [B,S,h_inner]
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        # proj_out: uses transpose (tied)
        x_outer = F.linear(x, self.proj.t().contiguous())  # [B,S,h_outer]
        x_outer = self.norm_outer(x_outer)
        return F.linear(x_outer, self.lm_head_w)


student = TinyFRRTied(H_OUTER, H_INNER, n_heads_inner, vocab_size,
                     embed_w_outer, lm_head_w_outer, norm_w_outer).to(DEVICE)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
teacher_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER
print(f"  trainable = {trainable/1e6:.2f}M  total = {total/1e6:.2f}M  teacher = {teacher_params/1e6:.1f}M  compression = {teacher_params/trainable:.1f}x")

# ==================== Data / Opt ====================
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=5e-5)
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_t10 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_t10 = ck.get('best_t10', 0.0)
    print(f"  resumed from step {start_step}  best_t10={best_t10:.3f}")


@torch.no_grad()
def quick_eval(n=100):
    student.eval()
    all_t10 = 0.0; last_t10 = 0.0; n_tok = 0
    eval_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (n,))
    for s in eval_starts:
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = student(toks)
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            h10 = len(set(tt.tolist()) & set(st.tolist())) / 10
            all_t10 += h10; n_tok += 1
            if pos == SEQ_LEN - 1: last_t10 += h10
    student.train()
    return all_t10/n_tok, last_t10/n


# ==================== Train ====================
student.train()
t0 = time.time()
for step in range(start_step, STEPS):
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
    toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts]).to(DEVICE)
    with torch.no_grad():
        t_logits = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        s_logits = student(toks)
        loss = F.kl_div(F.log_softmax(s_logits / TEMP, dim=-1),
                        F.softmax(t_logits / TEMP, dim=-1),
                        reduction='batchmean') * (TEMP ** 2)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update(); sched.step()

    if step % 100 == 0:
        print(f"  [{TAG}] step={step:5d}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if step > 0 and step % 1000 == 0:
        all10, last10 = quick_eval()
        new_best = last10 > best_t10
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:5d}  all-T10={all10*100:.1f}%  last-T10={last10*100:.1f}%  best={max(best_t10, last10)*100:.1f}%{marker}", flush=True)
        if new_best:
            best_t10 = last10
            torch.save({'state_dict': student.state_dict(), 'step': step,
                        'best_t10': best_t10, 'h_inner': H_INNER,
                        'n_heads_inner': n_heads_inner, 'trainable': trainable,
                        'tied': True}, BEST)
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best_t10': best_t10, 'h_inner': H_INNER,
                    'n_heads_inner': n_heads_inner, 'trainable': trainable,
                    'tied': True}, LATEST)

all10, last10 = quick_eval()
if last10 > best_t10:
    best_t10 = last10
    torch.save({'state_dict': student.state_dict(), 'step': STEPS,
                'best_t10': best_t10, 'h_inner': H_INNER,
                'n_heads_inner': n_heads_inner, 'trainable': trainable,
                'tied': True}, BEST)

print(f"\nDONE {TAG}: best last-T10 = {best_t10*100:.2f}%  trainable={trainable/1e6:.2f}M  compression={teacher_params/trainable:.0f}x")
