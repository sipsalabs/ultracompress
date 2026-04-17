"""
HQ2 — Hidden-state matching + logit distillation.

Key addition vs HQ: match teacher's post-final-norm 2048-dim LATENT
(the vector that gets projected to vocab by lm_head). This is the real
"move away from tokens" signal — student learns teacher's representation,
not just its output distribution.

Loss = fkl + rkl_w * rkl + latent_w * cosine_MSE(student_latent, teacher_latent)

Warm-starts from h{H}_hq if present, else h{H}_long, else h{H}.
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
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--latent_w', type=float, default=1.0)
ap.add_argument('--rkl_w', type=float, default=0.3)
args = ap.parse_args()

STEPS = args.steps
BATCH = 4
ACCUM = 2                      # effective batch 8
SEQ_LEN = 64
LR_MAX = 3e-4
LR_MIN = 1e-5
WARMUP = 1000
H_INNER = args.h
TAG = args.tag or f'h{H_INNER}_hq2'
N_TEACHER_LAYERS = 28
DEVICE = args.device
T_START = 2.0
T_END = 1.0
RKL_W = args.rkl_w
LATENT_W = args.latent_w

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"TinyFRR-HQ2: inner={H_INNER}  steps={STEPS}  tag={TAG}  device={DEVICE}")
print(f"  T: {T_START}->{T_END}  rkl_w={RKL_W}  latent_w={LATENT_W}  accum={ACCUM}")

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


class TinyFRR(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_out.weight, a=math.sqrt(5))
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

    def forward(self, tokens, return_latent=False):
        x_outer = self.embed(tokens).float()
        x = self.proj_in(x_outer)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x_outer = self.proj_out(x)
        latent = self.norm_outer(x_outer)          # post-norm, pre-lm_head
        logits = F.linear(latent, self.lm_head_w)
        if return_latent:
            return logits, latent
        return logits


# Warm-start: prefer hq > long > vanilla
WARM_CANDIDATES = [
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_hq/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_long/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}/best.pt',
]
WARM_PATH = next((p for p in WARM_CANDIDATES if os.path.exists(p)), None)

student = TinyFRR(H_OUTER, H_INNER, n_heads_inner, vocab_size,
                  embed_w_outer, lm_head_w_outer, norm_w_outer).to(DEVICE)
if WARM_PATH:
    warm = torch.load(WARM_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = student.load_state_dict(warm['state_dict'], strict=False)
    print(f"  WARM START from {WARM_PATH}  missing={len(missing)} unexpected={len(unexpected)}")
else:
    print(f"  no warm-start checkpoint found; training from scratch")

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
teacher_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER
print(f"  trainable = {trainable/1e6:.2f}M  compression = {teacher_params/trainable:.1f}x")

# ==================== Data / Opt ====================
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.01)

def lr_lambda(step):
    if step < WARMUP: return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN / LR_MAX) + (1 - LR_MIN / LR_MAX) * cos

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_ppl_ratio = 999.0
best_top1 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_ppl_ratio = ck.get('best_ppl_ratio', 999.0)
    best_top1 = ck.get('best_top1', 0.0)
    print(f"  resumed step={start_step}  best_ppl_ratio={best_ppl_ratio:.3f}  best_top1={best_top1:.3f}")


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end: return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


@torch.no_grad()
def teacher_latent(tokens):
    """Teacher post-final-norm latent (the vector that feeds lm_head)."""
    _, hs = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS, return_hidden=True)
    x = hs[-1]
    return teacher.final_norm(x)


@torch.no_grad()
def quick_eval(n=80):
    student.eval()
    all_t10 = last_t10 = top1 = 0.0
    teacher_nll = student_nll = 0.0
    n_tok = 0
    eval_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (n,))
    for s in eval_starts:
        s = int(s.item())
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]; tgt = toks[0, 1:SEQ_LEN+1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = student(inp)
        t_logp = F.log_softmax(tl[0].float(), dim=-1)
        s_logp = F.log_softmax(sl[0].float(), dim=-1)
        teacher_nll += -t_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        student_nll += -s_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        t_top1 = tl[0].argmax(-1); s_top1 = sl[0].argmax(-1)
        top1 += (t_top1 == s_top1).float().mean().item()
        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            h10 = len(tt & st) / 10
            all_t10 += h10; n_tok += 1
            if pos == SEQ_LEN - 1: last_t10 += h10
    student.train()
    t_ppl = math.exp(teacher_nll / n_tok)
    s_ppl = math.exp(student_nll / n_tok)
    return all_t10/n_tok, last_t10/n, top1/n, s_ppl/t_ppl


# ==================== Train ====================
student.train()
t0 = time.time()
opt.zero_grad()
for step in range(start_step, STEPS):
    T = current_temp(step)
    accum_loss = accum_fkl = accum_rkl = accum_lat = 0.0
    for micro in range(ACCUM):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
        toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts]).to(DEVICE)

        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
            t_latent = teacher.final_norm(t_hs[-1]).float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits, s_latent = student(toks, return_latent=True)

            # Logit KLs
            t_logp = F.log_softmax(t_logits / T, dim=-1)
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            t_prob = t_logp.exp(); s_prob = s_logp.exp()
            fkl = (t_prob * (t_logp - s_logp)).sum(-1).mean() * (T ** 2)
            rkl = (s_prob * (s_logp - t_logp)).sum(-1).mean() * (T ** 2)

            # Hidden-state matching: cosine + MSE (scale-invariant + magnitude)
            tl = t_latent.float(); sl = s_latent.float()
            cos = 1.0 - F.cosine_similarity(sl, tl, dim=-1).mean()
            mse = F.mse_loss(sl, tl) / (tl.float().pow(2).mean() + 1e-6)
            latent_loss = cos + 0.1 * mse

            loss = fkl + RKL_W * rkl + LATENT_W * latent_loss
            loss = loss / ACCUM

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_fkl += fkl.item() / ACCUM
        accum_rkl += rkl.item() / ACCUM
        accum_lat += latent_loss.item() / ACCUM

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    opt.zero_grad()

    if step % 100 == 0:
        lr = opt.param_groups[0]['lr']
        print(f"  [{TAG}] step={step:6d}  loss={accum_loss:.4f}  fkl={accum_fkl:.4f}  rkl={accum_rkl:.4f}  lat={accum_lat:.4f}  T={T:.2f}  lr={lr:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if step > 0 and step % 2000 == 0:
        all10, last10, top1, ppl_ratio = quick_eval()
        quality = 100.0 / ppl_ratio
        new_best = ppl_ratio < best_ppl_ratio
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  all-T10={all10*100:.1f}%  last-T10={last10*100:.1f}%  top1={top1*100:.1f}%  ppl-ratio={ppl_ratio:.3f}  quality={quality:.1f}%{marker}", flush=True)
        if new_best:
            best_ppl_ratio = ppl_ratio; best_top1 = top1
            torch.save({'state_dict': student.state_dict(), 'step': step,
                        'best_ppl_ratio': best_ppl_ratio, 'best_top1': best_top1,
                        'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
                        'trainable': trainable}, BEST)
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best_ppl_ratio': best_ppl_ratio,
                    'best_top1': best_top1, 'h_inner': H_INNER,
                    'n_heads_inner': n_heads_inner, 'trainable': trainable}, LATEST)

all10, last10, top1, ppl_ratio = quick_eval()
if ppl_ratio < best_ppl_ratio:
    best_ppl_ratio = ppl_ratio; best_top1 = top1
    torch.save({'state_dict': student.state_dict(), 'step': STEPS,
                'best_ppl_ratio': best_ppl_ratio, 'best_top1': best_top1,
                'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
                'trainable': trainable}, BEST)

print(f"\nDONE {TAG}: best_top1={best_top1*100:.2f}%  best_ppl_ratio={best_ppl_ratio:.3f}  quality={100/best_ppl_ratio:.1f}%  trainable={trainable/1e6:.2f}M  compression={teacher_params/trainable:.0f}x")
