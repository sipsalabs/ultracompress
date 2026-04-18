"""
HQ4 CEILING BREAK — flip HQ3's objective to escape T1=54% / T10=68% plateau.

Diagnosis of HQ3 plateau:
  - Confidence weighting (w=1/(1+H)) upweighted easy tokens; student already saturated there.
  - T10 gains live in high-entropy (hard) positions; those were DOWNweighted.
  - Latent MSE pulled toward mean-seeking regime the CE signal couldn't escape.

HQ4 fixes:
  1. INVERTED confidence weighting: w ∝ (1 + entropy) → force gradient into hard positions.
  2. LATENT RAMP-DOWN: latent_w 1.0 -> 0.1 between step 20K..50K, release the attractor.
  3. Entropy-weighted fKL: KL also focuses where teacher is uncertain.
  4. Warm-start from HQ3 best.pt (keep the T1=54% floor, only add ceiling).
  5. CE+margin ramp starts at 0.5 (we're already warm) to preserve T1.
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
ap.add_argument('--h', type=int, default=256)
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--latent_w', type=float, default=1.0)
ap.add_argument('--latent_w_final', type=float, default=0.1)
ap.add_argument('--rkl_w', type=float, default=0.3)
ap.add_argument('--ce_w', type=float, default=0.5)
ap.add_argument('--margin_w', type=float, default=0.3)
ap.add_argument('--entropy_power', type=float, default=1.0,
                help='exponent on (1+entropy) weight; higher = stronger hard-token focus')
args = ap.parse_args()

STEPS = args.steps
BATCH = 4
ACCUM = 2
SEQ_LEN = 128
LR_MAX = 2e-4  # slightly lower (warm-start from HQ3 best)
LR_MIN = 1e-5
WARMUP = 500
H_INNER = args.h
TAG = args.tag or f'hq4_h{H_INNER}'
N_TEACHER_LAYERS = 28
DEVICE = args.device
T_START = 2.0
T_END = 1.0
RKL_W = args.rkl_w
LATENT_W_START = args.latent_w
LATENT_W_END = args.latent_w_final
CE_W = args.ce_w
MARGIN_W = args.margin_w
MARGIN_TARGET = 1.0
ENT_POW = args.entropy_power

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"TinyFRR-HQ4 CEILING BREAK: inner={H_INNER}  steps={STEPS}  tag={TAG}  device={DEVICE}")
print(f"  T: {T_START}->{T_END}  rkl_w={RKL_W}  latent_w: {LATENT_W_START}->{LATENT_W_END}")
print(f"  ce_w={CE_W}  margin_w={MARGIN_W}  entropy_power={ENT_POW}")
print(f"  SEQ_LEN={SEQ_LEN}  BATCH={BATCH}  ACCUM={ACCUM}")

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
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd
H_OUTER = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=H_OUTER, intermediate_size=H_OUTER * 3,
                  vocab_size=vocab_size, head_dim=H_OUTER // 16)
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


class TinyFRR_HQ4(nn.Module):
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
        for p in self.inner.embed.parameters():
            p.requires_grad = False
        for p in self.inner.lm_head.parameters():
            p.requires_grad = False
        for p in self.inner.norm.parameters():
            p.requires_grad = False
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
        latent = self.norm_outer(x_outer)
        logits = F.linear(latent, self.lm_head_w)
        if return_latent:
            return logits, latent
        return logits


# ==================== Warm-start from HQ3 best ====================
WARM_CANDIDATES = [
    f'checkpoints_1.7b_tinyfrr_hq3_h{H_INNER}/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_hq2/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_long/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}/best.pt',
]

student = TinyFRR_HQ4(H_OUTER, H_INNER, n_heads_inner, vocab_size,
                      embed_w_outer, lm_head_w_outer, norm_w_outer).to(DEVICE)

warm_loaded = False
WARM_PATH = next((p for p in WARM_CANDIDATES if os.path.exists(p)), None)
if WARM_PATH:
    warm = torch.load(WARM_PATH, map_location=DEVICE, weights_only=False)
    sd = warm.get('state_dict', warm)
    missing, unexpected = student.load_state_dict(sd, strict=False)
    print(f"  WARM START from {WARM_PATH}  missing={len(missing)} unexpected={len(unexpected)}")
    warm_loaded = True
if not warm_loaded:
    print(f"  no warm-start checkpoint found; training from scratch")

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
teacher_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER
print(f"  trainable = {trainable/1e6:.2f}M  compression = {teacher_params/trainable:.1f}x")

# ==================== Data / Opt ====================
DATA_500M = 'fineweb_edu_500M_tokens.pt'
DATA_100M = 'fineweb_edu_100M_tokens.pt'
data_path = DATA_500M if os.path.exists(DATA_500M) else DATA_100M
all_tokens = torch.load(data_path, weights_only=True)
print(f"  Using {data_path}: {all_tokens.numel()/1e6:.0f}M tokens")

opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.01)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN / LR_MAX) + (1 - LR_MIN / LR_MAX) * cos


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_quality = 0.0
best_top1 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_quality = ck.get('best_quality', 0.0)
    best_top1 = ck.get('best_top1', 0.0)
    print(f"  resumed step={start_step}  best_quality={best_quality:.3f}  best_top1={best_top1:.3f}")


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end:
        return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


def ce_schedule(step):
    # Start higher since warm from HQ3 — 0.5 → 1.0 across 20%..60%
    ramp_start = int(STEPS * 0.2)
    ramp_end = int(STEPS * 0.6)
    if step < ramp_start:
        return 0.5
    if step >= ramp_end:
        return 1.0
    prog = (step - ramp_start) / (ramp_end - ramp_start)
    return 0.5 + 0.5 * prog


def latent_schedule(step):
    # 1.0 until step 20K, linear ramp to 0.1 by step 50K, then 0.1
    ramp_start = int(STEPS * 0.25)   # 20K
    ramp_end = int(STEPS * 0.625)    # 50K
    if step < ramp_start:
        return LATENT_W_START
    if step >= ramp_end:
        return LATENT_W_END
    prog = (step - ramp_start) / (ramp_end - ramp_start)
    return LATENT_W_START + (LATENT_W_END - LATENT_W_START) * prog


@torch.no_grad()
def quick_eval(n=100):
    student.eval()
    all_t10 = last_t10 = top1 = 0.0
    teacher_nll = student_nll = 0.0
    n_tok = 0
    eval_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (n,))
    for s in eval_starts:
        s = int(s.item())
        toks = all_tokens[s:s + SEQ_LEN + 1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]
        tgt = toks[0, 1:SEQ_LEN + 1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = student(inp)
        t_logp = F.log_softmax(tl[0].float(), dim=-1)
        s_logp = F.log_softmax(sl[0].float(), dim=-1)
        teacher_nll += -t_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        student_nll += -s_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        t_top1 = tl[0].argmax(-1)
        s_top1 = sl[0].argmax(-1)
        top1 += (t_top1 == s_top1).float().mean().item()
        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            h10 = len(tt & st) / 10
            all_t10 += h10
            n_tok += 1
            if pos == SEQ_LEN - 1:
                last_t10 += h10
    student.train()
    t_ppl = math.exp(teacher_nll / n_tok)
    s_ppl = math.exp(student_nll / n_tok)
    return all_t10 / n_tok, last_t10 / n, top1 / n, s_ppl / t_ppl


# ==================== Train ====================
student.train()
t0 = time.time()
opt.zero_grad()

for step in range(start_step, STEPS):
    T = current_temp(step)
    ce_ramp = ce_schedule(step)
    lat_w = latent_schedule(step)

    accum_loss = accum_fkl = accum_rkl = accum_lat = accum_ce = accum_mrg = 0.0
    for micro in range(ACCUM):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
        toks = torch.stack([all_tokens[s:s + SEQ_LEN].long() for s in starts]).to(DEVICE)

        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
            t_latent = teacher.final_norm(t_hs[-1]).float()
            t_logits_f = t_logits.float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits, s_latent = student(toks, return_latent=True)

            # Teacher probs/entropy at training temperature
            t_logp = F.log_softmax(t_logits_f / T, dim=-1)
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            t_prob = t_logp.exp()
            t_entropy = -(t_prob * t_logp).sum(-1)  # [B, S]

            # *** INVERTED confidence weighting: upweight hard/high-entropy positions ***
            hard_weight = (1.0 + t_entropy) ** ENT_POW
            hard_weight = hard_weight / hard_weight.mean()  # normalize to mean=1

            # ---- Loss 1: Entropy-weighted Forward KL (T10 signal on HARD tokens) ----
            fkl_per = (t_prob * (t_logp - s_logp)).sum(-1)  # [B, S]
            fkl = (fkl_per * hard_weight).mean() * (T ** 2)

            # ---- Loss 2: Reverse KL (unweighted to keep mode-seeking pressure) ----
            s_prob = s_logp.exp()
            rkl_per = (s_prob * (s_logp - t_logp)).sum(-1)
            rkl = rkl_per.mean() * (T ** 2)

            # ---- Loss 3: Latent matching (ramp DOWN over training) ----
            tl = t_latent.float()
            sl = s_latent.float()
            cos = 1.0 - F.cosine_similarity(sl, tl, dim=-1).mean()
            mse = F.mse_loss(sl, tl) / (tl.float().pow(2).mean() + 1e-6)
            latent_loss = cos + 0.1 * mse

            # ---- Loss 4: CE on teacher argmax — unweighted (T1 on ALL tokens) ----
            t_argmax = t_logits_f.argmax(dim=-1)  # [B, S]
            ce_loss = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                t_argmax.view(-1),
                reduction='mean',
            )

            # ---- Loss 5: Margin ranking — also focused on HARD tokens ----
            s_logits_f = s_logits.float()
            s_at_t1 = s_logits_f.gather(-1, t_argmax.unsqueeze(-1)).squeeze(-1)
            s_mask = s_logits_f.clone()
            s_mask.scatter_(-1, t_argmax.unsqueeze(-1), float('-inf'))
            s_runner_up = s_mask.max(dim=-1).values
            margin_raw = F.relu(MARGIN_TARGET - (s_at_t1 - s_runner_up))
            margin_loss = (margin_raw * hard_weight).mean()

            # ---- Combined loss ----
            loss = (fkl
                    + RKL_W * rkl
                    + lat_w * latent_loss
                    + CE_W * ce_ramp * ce_loss
                    + MARGIN_W * ce_ramp * margin_loss)
            loss = loss / ACCUM

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_fkl += fkl.item() / ACCUM
        accum_rkl += rkl.item() / ACCUM
        accum_lat += latent_loss.item() / ACCUM
        accum_ce += ce_loss.item() / ACCUM
        accum_mrg += margin_loss.item() / ACCUM

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad()

    if step % 100 == 0:
        lr = opt.param_groups[0]['lr']
        print(f"  [{TAG}] step={step:6d}  loss={accum_loss:.4f}  fkl={accum_fkl:.4f}  "
              f"rkl={accum_rkl:.4f}  lat={accum_lat:.4f}  ce={accum_ce:.4f}  mrg={accum_mrg:.4f}  "
              f"T={T:.2f}  lat_w={lat_w:.2f}  ce_ramp={ce_ramp:.2f}  lr={lr:.5f}  ({time.time()-t0:.0f}s)",
              flush=True)

    if step > 0 and step % 2000 == 0:
        all10, last10, top1, ppl_ratio = quick_eval()
        quality = (all10 * 100 + top1 * 100 + 100.0 / ppl_ratio) / 3.0
        new_best = quality > best_quality
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  all-T10={all10*100:.1f}%  last-T10={last10*100:.1f}%  "
              f"T1={top1*100:.1f}%  ppl-ratio={ppl_ratio:.3f}  quality={quality:.1f}%{marker}",
              flush=True)
        if new_best:
            best_quality = quality
            best_top1 = top1
            torch.save({
                'state_dict': student.state_dict(), 'step': step,
                'best_quality': best_quality, 'best_top1': best_top1,
                'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
                'trainable': trainable,
            }, BEST)
        torch.save({
            'state_dict': student.state_dict(), 'opt': opt.state_dict(),
            'step': step, 'best_quality': best_quality,
            'best_top1': best_top1, 'h_inner': H_INNER,
            'n_heads_inner': n_heads_inner, 'trainable': trainable,
        }, LATEST)

# Final eval
all10, last10, top1, ppl_ratio = quick_eval()
quality = (all10 * 100 + top1 * 100 + 100.0 / ppl_ratio) / 3.0
if quality > best_quality:
    best_quality = quality
    best_top1 = top1
    torch.save({
        'state_dict': student.state_dict(), 'step': STEPS,
        'best_quality': best_quality, 'best_top1': best_top1,
        'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
        'trainable': trainable,
    }, BEST)

print(f"\nDONE {TAG}: best_top1={best_top1*100:.2f}%  best_quality={best_quality:.1f}%  "
      f"trainable={trainable/1e6:.2f}M  compression={teacher_params/trainable:.0f}x")
