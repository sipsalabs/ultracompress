"""
HQ8 — Progressive Intra-Iteration Distillation (PIID).

Novel training objective for FRR. Instead of only supervising the final
student output against the final teacher output, we supervise at 4
intermediate iteration checkpoints within the student's recursive body
against the teacher's corresponding intermediate layer hidden states.

Why this is novel:

  Prior layer-wise distillation (TinyBERT, MobileBERT, DistilBERT) adds
  intermediate supervision to a student that has distinct per-layer
  weights. FRR has ONE shared block iterated N times. Naive layer-wise
  matching would give contradictory gradients to the same weights.

  Our key insight: use the SAME proj_out head at each checkpoint. The
  constraint that a single projection must decode 4 different intermediate
  representations into 4 different teacher-layer outputs acts as a strong
  *structural regularizer*. It forces the shared block to produce
  progressively refined representations that all live on a single
  decodeable manifold.

  Combined with a curriculum (intermediate loss >> final loss early,
  reversed late), this gives the student 4x more gradient signal per
  forward pass without adding any parameters.

Checkpoint mapping (FRR 4 scales x 7 iters = 28 inner iterations, teacher
28 layers):
  scale 0 end (inner iter  7) -> teacher layer  7
  scale 1 end (inner iter 14) -> teacher layer 14
  scale 2 end (inner iter 21) -> teacher layer 21
  scale 3 end (inner iter 28) -> teacher layer 28 (= final, unchanged)

Loss schedule:
  steps 0..20%      : w_final=0.3,  w_intermediate=1.0
  steps 20..60%     : linear transition
  steps 60..100%    : w_final=1.0,  w_intermediate=0.2

Usage:
  python run_frr_hq8.py --teacher_cache qwen3_1.7b_cache.pt \
      --h 256 --steps 80000 --tag hq8_h256 --device cuda:1 \
      --warm_from checkpoints_1.7b_tinyfrr_hq5_h256/best.pt

Warm-starting from HQ5 is recommended: HQ8 refines an already-good model
with richer supervision rather than trying to learn everything from scratch.
"""
import sys
import os
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.moonshot import FractalModel
from scaling.teacher_loader import load_qwen3_teacher

ap = argparse.ArgumentParser()
ap.add_argument('--teacher_cache', type=str, required=True)
ap.add_argument('--h', type=int, default=256)
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--tag', type=str, required=True)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--accum', type=int, default=2)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--lr', type=float, default=1e-4,
                help='Lower than HQ5 default since we warm-start')
ap.add_argument('--rkl_w', type=float, default=0.3)
ap.add_argument('--ce_w', type=float, default=0.5)
ap.add_argument('--margin_w', type=float, default=0.3)
ap.add_argument('--entropy_power', type=float, default=1.5)
ap.add_argument('--w_inter_start', type=float, default=1.0,
                help='weight on intermediate losses at start of training')
ap.add_argument('--w_inter_end', type=float, default=0.2,
                help='weight on intermediate losses at end of training')
ap.add_argument('--w_final_start', type=float, default=0.3)
ap.add_argument('--w_final_end', type=float, default=1.0)
ap.add_argument('--warm_from', type=str, default=None)
args = ap.parse_args()

STEPS = args.steps
BATCH = args.batch
ACCUM = args.accum
SEQ_LEN = args.seq_len
LR_MAX = args.lr
LR_MIN = 1e-5
WARMUP = 500
H_INNER = args.h
TAG = args.tag
DEVICE = args.device
T_START, T_END = 2.0, 1.0
ENT_POW = args.entropy_power
MARGIN_TARGET = 1.0

# 4 intermediate checkpoints aligned with FRR's 4 scales.
# Inner iteration indices where we snapshot the student, and the
# teacher layer index we supervise against.
CHECKPOINT_STUDENT_SCALES = [0, 1, 2, 3]          # end of each scale
CHECKPOINT_TEACHER_LAYERS = [7, 14, 21, 28]       # evenly spaced through the teacher

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"HQ8 PIID training  teacher={args.teacher_cache}  h={H_INNER}  steps={STEPS}")
print(f"  tag={TAG}  device={DEVICE}")
print(f"  checkpoint map: student_scale_end -> teacher_layer")
for ss, tl in zip(CHECKPOINT_STUDENT_SCALES, CHECKPOINT_TEACHER_LAYERS):
    print(f"    scale {ss} end -> teacher layer {tl}")


# ---------- Teacher ----------
tb = load_qwen3_teacher(args.teacher_cache, device=DEVICE)
teacher = tb.teacher
H_OUTER = tb.h_outer
vocab_size = tb.vocab_size
N_TEACHER_LAYERS = tb.n_layers
assert N_TEACHER_LAYERS == 28, \
    f"HQ8 checkpoint mapping assumes 28-layer teacher, got {N_TEACHER_LAYERS}"

teacher_total_params = sum(p.numel() for p in [teacher.embed_weight,
    *([teacher.lm_head] if teacher.lm_head is not None else [])])
for layer in teacher.layers:
    for p in layer.parameters():
        teacher_total_params += p.numel()
print(f"  teacher total params: {teacher_total_params/1e9:.3f}B")


# ---------- Student ----------
candidate_heads = [16, 8, 12, 4]
n_heads_inner = next((h for h in candidate_heads if H_INNER % h == 0), 4)
print(f"  H_OUTER={H_OUTER}  H_INNER={H_INNER}  n_heads_inner={n_heads_inner}")


class TinyFRR_HQ8(nn.Module):
    """Standard TinyFRR with one extra method: forward_with_checkpoints.

    Returns (final_logits, final_latent, [latent_at_each_scale_end]).
    Intermediate latents are produced by applying the SAME proj_out + norm_outer
    used for the final output. This parameter sharing is the key novelty —
    a single projection must decode representations from 4 stages of refinement.
    """

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

    def forward_with_checkpoints(self, tokens):
        x_outer = self.embed(tokens).float()
        x = self.proj_in(x_outer)
        fr = self.inner
        checkpoint_latents = []
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
            # end of this scale -- snapshot
            intermediate_outer = self.proj_out(x)
            intermediate_latent = self.norm_outer(intermediate_outer)
            checkpoint_latents.append(intermediate_latent)
        final_latent = checkpoint_latents[-1]
        final_logits = F.linear(final_latent, self.lm_head_w)
        return final_logits, final_latent, checkpoint_latents

    def forward(self, tokens, return_latent=False):
        # Inference-time path; identical result to the generic trainer.
        final_logits, final_latent, _ = self.forward_with_checkpoints(tokens)
        if return_latent:
            return final_logits, final_latent
        return final_logits


student = TinyFRR_HQ8(H_OUTER, H_INNER, n_heads_inner, vocab_size,
                      tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)

if args.warm_from and os.path.exists(args.warm_from):
    warm = torch.load(args.warm_from, map_location=DEVICE, weights_only=False)
    sd = warm.get('state_dict', warm)
    missing, unexpected = student.load_state_dict(sd, strict=False)
    print(f"  warm-start {args.warm_from}  missing={len(missing)} unexpected={len(unexpected)}")
else:
    print("  WARNING: no warm_from checkpoint; HQ8 converges much faster when warm-starting from HQ5")

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
comp_ratio = teacher_total_params / trainable
print(f"  trainable = {trainable/1e6:.3f}M  compression = {comp_ratio:.1f}x")


# ---------- Data ----------
DATA_500M = 'fineweb_edu_500M_tokens.pt'
DATA_100M = 'fineweb_edu_100M_tokens.pt'
data_path = DATA_500M if os.path.exists(DATA_500M) else DATA_100M
all_tokens = torch.load(data_path, weights_only=True)
print(f"  Data: {data_path}  {all_tokens.numel()/1e6:.0f}M tokens")


# ---------- Schedules ----------
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.01)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return (LR_MIN / LR_MAX) + (1 - LR_MIN / LR_MAX) * 0.5 * (1 + math.cos(math.pi * prog))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler('cuda')


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end:
        return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


def ce_ramp_schedule(step):
    rs, re = int(STEPS * 0.2), int(STEPS * 0.6)
    if step < rs: return 0.5
    if step >= re: return 1.0
    return 0.5 + 0.5 * (step - rs) / (re - rs)


def piid_schedule(step):
    """Curriculum: intermediate loss heavy early, final loss heavy late."""
    rs, re = int(STEPS * 0.2), int(STEPS * 0.6)
    if step < rs:
        w_inter = args.w_inter_start
        w_final = args.w_final_start
    elif step >= re:
        w_inter = args.w_inter_end
        w_final = args.w_final_end
    else:
        prog = (step - rs) / (re - rs)
        w_inter = args.w_inter_start + (args.w_inter_end - args.w_inter_start) * prog
        w_final = args.w_final_start + (args.w_final_end - args.w_final_start) * prog
    return w_inter, w_final


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
        t_lp = F.log_softmax(tl[0].float(), -1)
        s_lp = F.log_softmax(sl[0].float(), -1)
        teacher_nll += -t_lp[torch.arange(SEQ_LEN), tgt].sum().item()
        student_nll += -s_lp[torch.arange(SEQ_LEN), tgt].sum().item()
        t1 = tl[0].argmax(-1)
        s1 = sl[0].argmax(-1)
        top1 += (t1 == s1).float().mean().item()
        for pos in range(SEQ_LEN):
            h10 = len(set(tl[0, pos].topk(10).indices.tolist())
                      & set(sl[0, pos].topk(10).indices.tolist())) / 10
            all_t10 += h10
            n_tok += 1
            if pos == SEQ_LEN - 1:
                last_t10 += h10
    student.train()
    return (all_t10 / n_tok, last_t10 / n, top1 / n,
            math.exp(student_nll / n_tok) / math.exp(teacher_nll / n_tok))


# ---------- Resume ----------
start_step = 0
best_quality = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_quality = ck.get('best_quality', 0.0)
    print(f"  resumed step={start_step}  best_quality={best_quality:.3f}")


# ---------- Train ----------
student.train()
t0 = time.time()
opt.zero_grad()

for step in range(start_step, STEPS):
    T = current_temp(step)
    ce_ramp = ce_ramp_schedule(step)
    w_inter, w_final = piid_schedule(step)

    accum_loss = accum_fkl = accum_rkl = accum_ce = accum_mrg = 0.0
    accum_inter = [0.0, 0.0, 0.0, 0.0]

    for micro in range(ACCUM):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
        toks = torch.stack([all_tokens[s:s + SEQ_LEN].long() for s in starts]).to(DEVICE)

        # Teacher hidden states at all layers
        with torch.no_grad():
            t_logits, t_hs = teacher.forward(
                toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
            # t_hs has N_TEACHER_LAYERS + 1 entries (incl. embedding layer).
            # layer_k_output = t_hs[k]. Use teacher.final_norm for the final latent.
            t_latent_final = teacher.final_norm(t_hs[-1]).float()
            # Intermediate teacher latents: apply final_norm (same decoder frame).
            t_latent_intermediates = [
                teacher.final_norm(t_hs[tl]).float()
                for tl in CHECKPOINT_TEACHER_LAYERS[:-1]
            ]
            t_logits_f = t_logits.float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits, s_latent_final, s_checkpoint_latents = \
                student.forward_with_checkpoints(toks)

            # --- final losses (same as HQ5) ---
            t_logp = F.log_softmax(t_logits_f / T, dim=-1)
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            t_prob = t_logp.exp()
            t_entropy = -(t_prob * t_logp).sum(-1)
            hw = (1.0 + t_entropy) ** ENT_POW
            hw = hw / hw.mean()

            fkl = ((t_prob * (t_logp - s_logp)).sum(-1) * hw).mean() * (T * T)
            s_prob = s_logp.exp()
            rkl = (s_prob * (s_logp - t_logp)).sum(-1).mean() * (T * T)

            sl_final_f = s_latent_final.float()
            final_latent_loss = (
                1.0 - F.cosine_similarity(sl_final_f, t_latent_final, dim=-1).mean()
            )

            t_argmax = t_logits_f.argmax(-1)
            ce = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                t_argmax.view(-1),
            )
            sf = s_logits.float()
            s_at_t1 = sf.gather(-1, t_argmax.unsqueeze(-1)).squeeze(-1)
            sm = sf.clone()
            sm.scatter_(-1, t_argmax.unsqueeze(-1), float('-inf'))
            runner_up = sm.max(-1).values
            margin = (F.relu(MARGIN_TARGET - (s_at_t1 - runner_up)) * hw).mean()

            # --- NOVEL: intermediate-iteration latent losses ---
            # student checkpoint k (scale k end, k=0..2) must match teacher layer 7/14/21
            # Use same final_norm frame so comparison is meaningful.
            inter_losses = []
            for k in range(len(CHECKPOINT_TEACHER_LAYERS) - 1):  # skip final
                s_lat_k = s_checkpoint_latents[k].float()
                t_lat_k = t_latent_intermediates[k]
                cos_k = 1.0 - F.cosine_similarity(s_lat_k, t_lat_k, dim=-1).mean()
                inter_losses.append(cos_k)
            inter_total = sum(inter_losses) / max(1, len(inter_losses))

            loss = (
                w_final * (fkl + args.rkl_w * rkl + 0.5 * final_latent_loss
                           + args.ce_w * ce_ramp * ce
                           + args.margin_w * ce_ramp * margin)
                + w_inter * inter_total
            ) / ACCUM

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_fkl += fkl.item() / ACCUM
        accum_rkl += rkl.item() / ACCUM
        accum_ce += ce.item() / ACCUM
        accum_mrg += margin.item() / ACCUM
        for k, il in enumerate(inter_losses):
            accum_inter[k] += il.item() / ACCUM

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad()

    if step % 100 == 0:
        lr = opt.param_groups[0]['lr']
        inter_str = '/'.join(f'{x:.3f}' for x in accum_inter[:3])
        print(f"  [{TAG}] step={step:6d}  loss={accum_loss:.4f}  fkl={accum_fkl:.4f}  "
              f"rkl={accum_rkl:.4f}  ce={accum_ce:.4f}  mrg={accum_mrg:.4f}  "
              f"inter={inter_str}  w_inter={w_inter:.2f}  w_final={w_final:.2f}  "
              f"T={T:.2f}  lr={lr:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    if step > 0 and step % 2000 == 0:
        all10, last10, t1, pr = quick_eval()
        quality = (all10 * 100 + t1 * 100 + 100.0 / pr) / 3.0
        best = quality > best_quality
        marker = ' *** NEW BEST ***' if best else ''
        print(f"  EVAL step={step}  all-T10={all10*100:.2f}%  last-T10={last10*100:.2f}%  "
              f"T1={t1*100:.2f}%  ppl-ratio={pr:.3f}  Q={quality:.2f}{marker}", flush=True)
        if best:
            best_quality = quality
            torch.save({
                'state_dict': student.state_dict(),
                'step': step, 'best_quality': best_quality,
                'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
                'trainable': trainable,
                'teacher_cache': args.teacher_cache,
                'teacher_total_params': teacher_total_params,
                'compression_ratio': comp_ratio,
                'h_outer': H_OUTER, 'vocab_size': vocab_size,
                'hq8': True,
            }, BEST)
        torch.save({
            'state_dict': student.state_dict(), 'opt': opt.state_dict(),
            'step': step, 'best_quality': best_quality,
            'h_inner': H_INNER, 'n_heads_inner': n_heads_inner,
            'trainable': trainable,
            'teacher_cache': args.teacher_cache,
        }, LATEST)

print(f"\nDone. Best quality = {best_quality:.2f}   checkpoint: {BEST}")
