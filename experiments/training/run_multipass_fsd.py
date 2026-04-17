"""
Multi-Pass Fused Spectral Distillation (MP-FSD)

NOVEL: Run the SAME FRR body TWICE with DIFFERENT modulation, concatenate
outputs, then fused decode from 2*h dims. This doubles effective capacity
WITHOUT doubling parameters (only adds ~8K params for extra modulation).

Why this is a breakthrough idea:
  With h=256, a single body pass can only produce vectors in a 256-dim subspace.
  But the fused decode needs 512 independent directions to match ASVD r=512.
  TWO passes with different modulation produce 2*256=512 effective dims,
  breaking the linear bottleneck of a single pass.

  Same body weights, same compute structure, but two views of the input.
  Like binocular vision: two eyes see the same scene from different angles
  and together perceive depth (3D) that neither eye alone can capture.

Architecture:
  embed(tokens) → x
  pass1 = body(x, mod_A) → z1 [h-dim]   # "what" channel
  pass2 = body(x, mod_B) → z2 [h-dim]   # "where" channel
  z = fused_decode([z1; z2]) → r-dim     # nonlinear combination
  logits = vocab_head(z)                  # r → V

  Extra params: just 2*h for mod_B gammas/betas = ~512 params
  Extra compute: 1 more body forward pass = ~1M FLOPs

Usage:
  python run_multipass_fsd.py --h 256 --r 512 --steps 80000 --device cuda:1
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
ap.add_argument('--h', type=int, default=256, help='FRR body inner dim')
ap.add_argument('--r', type=int, default=512, help='Vocab rank')
ap.add_argument('--n_passes', type=int, default=2, help='Number of body passes')
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--margin', type=float, default=1.0)
ap.add_argument('--margin_w', type=float, default=0.2)
ap.add_argument('--rkl_w', type=float, default=0.3)
args = ap.parse_args()

STEPS = args.steps
BATCH = 4
SEQ_LEN = args.seq_len
LR_MAX = 3e-4
LR_MIN = 1e-5
WARMUP = 2000
H_INNER = args.h
R = args.r
N_PASSES = args.n_passes
TAG = args.tag or f'mpfsd_h{H_INNER}_r{R}_p{N_PASSES}'
N_TEACHER_LAYERS = 28
DEVICE = args.device
T_START = 2.0
T_END = 1.0
K_TOPK = 128
MARGIN_TARGET = args.margin
MARGIN_W = args.margin_w
RKL_W = args.rkl_w

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"=== Multi-Pass Fused Spectral Distillation (MP-FSD) ===")
print(f"  h={H_INNER} r={R} passes={N_PASSES} steps={STEPS}")
print(f"  tag={TAG} device={DEVICE}")

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
embed_w = gd['token_embd.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE).float()
del gd

candidate_heads = [16, 8, 12, 4, 2]
n_heads_inner = next((h for h in candidate_heads if H_INNER % h == 0), 4)

# ==================== ASVD Init ====================
print("Computing ASVD initialization...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
Mcov = torch.zeros(H_OUTER, H_OUTER, device=DEVICE, dtype=torch.float32)
n_obs = 0
with torch.no_grad():
    calib_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (48,))
    for s in calib_starts:
        toks = all_tokens[int(s):int(s) + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        _, hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        latent = teacher.final_norm(hs[-1]).float()
        flat = latent.reshape(-1, H_OUTER)
        Mcov += flat.T @ flat
        n_obs += flat.shape[0]
Mcov /= n_obs
eps = 1e-3 * Mcov.diagonal().mean().item()
Mcov += eps * torch.eye(H_OUTER, device=DEVICE)
Lchol = torch.linalg.cholesky(Mcov)
Linv = torch.linalg.solve_triangular(Lchol, torch.eye(H_OUTER, device=DEVICE), upper=False)
WL = lm_head_w @ Lchol
U, S, Vt = torch.linalg.svd(WL, full_matrices=False)
U_r = U[:, :R].contiguous()
S_r = S[:R]
Vt_r = Vt[:R, :].contiguous()
asvd_proj_init = (Vt_r @ Linv).contiguous()
asvd_out_init = (U_r * S_r.unsqueeze(0)).contiguous()
del Mcov, Lchol, Linv, WL, U, S, Vt, U_r, S_r, Vt_r

with torch.no_grad():
    recon = asvd_out_init @ asvd_proj_init
    err = (recon - lm_head_w).norm() / lm_head_w.norm()
    print(f"  ASVD r={R}: recon error = {err:.4f}")
del recon, lm_head_w
torch.cuda.empty_cache()


# ==================== Multi-Pass Fused Decode ====================
class MultiPassFusedDecode(nn.Module):
    """Nonlinear fused decode from n_passes * h_inner dims to rank."""
    def __init__(self, h_inner, rank, n_passes):
        super().__init__()
        in_dim = h_inner * n_passes
        # Linear path
        self.linear = nn.Linear(in_dim, rank, bias=False)
        # Nonlinear residual (SwiGLU)
        mid = in_dim * 2
        self.mlp_gate = nn.Linear(in_dim, mid, bias=False)
        self.mlp_up = nn.Linear(in_dim, mid, bias=False)
        self.mlp_down = nn.Linear(mid, rank, bias=False)
        nn.init.zeros_(self.mlp_down.weight)  # residual starts at zero
        self.norm = nn.RMSNorm(in_dim)

    def forward(self, x):
        z_linear = self.linear(x)
        h = self.norm(x)
        z_residual = self.mlp_down(F.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return z_linear + z_residual


# ==================== Multi-Pass Model ====================
class MultiPassFSD(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, rank, n_passes,
                 embed_w, asvd_out):
        super().__init__()
        self.h_outer = h_outer
        self.h_inner = h_inner
        self.n_passes = n_passes

        # Frozen teacher embedding
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)

        # Body: shared proj_in + FRR
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7, vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters():
            p.requires_grad = False
        for p in self.inner.lm_head.parameters():
            p.requires_grad = False
        for p in self.inner.norm.parameters():
            p.requires_grad = False

        # Per-pass modulation (the NOVEL part): different gamma/beta per pass
        # Pass 0 uses the FRR's built-in scale_gamma/beta (already trained)
        # Passes 1..n_passes-1 get their own learned modulation
        if n_passes > 1:
            self.extra_gamma = nn.ParameterList([
                nn.Parameter(torch.ones(4, h_inner))  # 4 scales
                for _ in range(n_passes - 1)
            ])
            self.extra_beta = nn.ParameterList([
                nn.Parameter(torch.zeros(4, h_inner))
                for _ in range(n_passes - 1)
            ])

        # Fused decode from concatenated passes
        self.fused_decode = MultiPassFusedDecode(h_inner, rank, n_passes)

        # Vocab head
        self.vocab_head = nn.Linear(rank, vocab, bias=False)
        self.vocab_head.weight.data.copy_(asvd_out)

    def _run_body(self, x, gamma_set, beta_set):
        """Run FRR body with given modulation."""
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = gamma_set[scale]
            beta = beta_set[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        return x

    def forward(self, tokens):
        x_embed = self.embed(tokens).float()
        x_proj = self.proj_in(x_embed)

        outputs = []

        # Pass 0: use FRR's built-in modulation
        z0 = self._run_body(x_proj.clone(), self.inner.scale_gamma, self.inner.scale_beta)
        outputs.append(z0)

        # Additional passes: use learned modulation
        for p in range(1, self.n_passes):
            zp = self._run_body(x_proj.clone(), self.extra_gamma[p - 1], self.extra_beta[p - 1])
            outputs.append(zp)

        # Concatenate all passes
        z_cat = torch.cat(outputs, dim=-1)  # [B, S, n_passes * h_inner]

        # Fused decode
        z = self.fused_decode(z_cat)
        logits = self.vocab_head(z)
        return logits


# ==================== Build ====================
# Load fine-tuned ASVD vocab head
ASVD_FT = f'checkpoints_1.7b_asvd_r{R}_ft/best.pt'
if os.path.exists(ASVD_FT):
    ack = torch.load(ASVD_FT, map_location=DEVICE, weights_only=False)
    asvd_out_ft = ack.get('asvd_out', ack.get('state_dict', {}).get('asvd_out.weight'))
    if asvd_out_ft is not None:
        asvd_out_init = asvd_out_ft.contiguous()
        print(f"  Using fine-tuned ASVD vocab from {ASVD_FT}")
    del ack

student = MultiPassFSD(
    H_OUTER, H_INNER, n_heads_inner, vocab_size, R, N_PASSES,
    embed_w, asvd_out_init
).to(DEVICE)

# Warm-start body from best existing checkpoint
WARM_CANDIDATES = [
    f'checkpoints_1.7b_tinyfrr_v2_h{H_INNER}_r{R}/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_long/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}/best.pt',
]
WARM_PATH = next((p for p in WARM_CANDIDATES if os.path.exists(p)), None)
if WARM_PATH:
    warm = torch.load(WARM_PATH, map_location=DEVICE, weights_only=False)
    sd = warm.get('state_dict', warm)
    body_keys = {}
    for k, v in sd.items():
        if k.startswith('proj_in.') or k.startswith('inner.block.') or \
           k.startswith('inner.scale_') or k.startswith('inner.iter_'):
            if k in student.state_dict():
                body_keys[k] = v
    if body_keys:
        student.load_state_dict(body_keys, strict=False)
        print(f"  WARM START body from {WARM_PATH}: {len(body_keys)} keys loaded")
    del warm

# Also try warm-starting fused decode from FSD checkpoint
FSD_CKPT = f'checkpoints_1.7b_tinyfrr_fsd_h{H_INNER}_r{R}/best.pt'
if os.path.exists(FSD_CKPT):
    fsd = torch.load(FSD_CKPT, map_location=DEVICE, weights_only=False)
    fsd_sd = fsd.get('state_dict', fsd)
    # The FSD decode is h_inner -> r, but MP-FSD decode is n_passes*h_inner -> r
    # We can warm-start the linear layer's first h_inner columns from FSD's linear
    if 'fused_decode.linear.weight' in fsd_sd:
        fsd_linear = fsd_sd['fused_decode.linear.weight']  # [r, h_inner]
        # Initialize: first h cols from FSD, rest zero
        mp_linear = student.fused_decode.linear.weight.data
        mp_linear[:, :H_INNER] = fsd_linear
        mp_linear[:, H_INNER:] = 0  # extra passes start inactive
        print(f"  WARM START fused decode from {FSD_CKPT} (first pass)")
    del fsd

del embed_w, asvd_proj_init, asvd_out_init
torch.cuda.empty_cache()

# ==================== Param Accounting ====================
body_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and not n.startswith('vocab_head')
                  and not n.startswith('fused_decode') and not n.startswith('extra_'))
extra_mod_params = sum(p.numel() for n, p in student.named_parameters()
                       if n.startswith('extra_'))
decode_params = sum(p.numel() for n, p in student.named_parameters()
                    if p.requires_grad and n.startswith('fused_decode'))
head_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and n.startswith('vocab_head'))
total_train = body_params + extra_mod_params + decode_params + head_params
teacher_full = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + vocab_size * H_OUTER

print(f"  body={body_params / 1e6:.2f}M  extra_mod={extra_mod_params / 1e3:.1f}K  "
      f"decode={decode_params / 1e6:.2f}M  vocab={head_params / 1e6:.2f}M  "
      f"total={total_train / 1e6:.2f}M")
print(f"  effective_dims={H_INNER * N_PASSES}  (vs {H_INNER} single-pass)")
print(f"  teacher={teacher_full / 1e6:.1f}M  compression={teacher_full / total_train:.1f}x")

# ==================== Data ====================
if os.path.exists('fineweb_edu_500M_tokens.pt'):
    print("  Using 500M token dataset")
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
else:
    all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
print(f"  tokens: {all_tokens.numel() / 1e6:.0f}M")

# ==================== Optimizer ====================
opt = torch.optim.AdamW([
    {'params': [p for n, p in student.named_parameters()
                if p.requires_grad and not n.startswith('vocab_head')
                and not n.startswith('fused_decode')],
     'lr': LR_MAX},
    {'params': [p for n, p in student.named_parameters()
                if p.requires_grad and n.startswith('fused_decode')],
     'lr': LR_MAX * 0.5},
    {'params': [p for n, p in student.named_parameters()
                if p.requires_grad and n.startswith('vocab_head')],
     'lr': LR_MAX * 0.1},
], betas=(0.9, 0.95), weight_decay=0.01)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN / LR_MAX) + (1 - LR_MIN / LR_MAX) * cos


sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda, lr_lambda, lr_lambda])
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_top1 = 0.0
best_top10 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_top1 = ck.get('best_top1', 0.0)
    best_top10 = ck.get('best_top10', 0.0)
    print(f"  Resumed step={start_step}")


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end:
        return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


@torch.no_grad()
def quick_eval(n=100):
    student.eval()
    top1_agree = top10_overlap = n_tok = 0
    teacher_nll = student_nll = 0.0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (1,)).item()
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
        top1_agree += (t_top1 == s_top1).sum().item()
        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            top10_overlap += len(tt & st) / 10.0
        n_tok += SEQ_LEN
    student.train()
    t_ppl = math.exp(teacher_nll / n_tok)
    s_ppl = math.exp(student_nll / n_tok)
    return {
        'top1': 100.0 * top1_agree / n_tok,
        'top10': 100.0 * top10_overlap / n_tok,
        'ppl_ratio': s_ppl / t_ppl,
    }


# ==================== Training ====================
student.train()
t0 = time.time()

for step in range(start_step, STEPS):
    T = current_temp(step)
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
    toks = torch.stack([all_tokens[s:s + SEQ_LEN].long() for s in starts]).to(DEVICE)

    with torch.no_grad():
        t_logits = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        s_logits = student(toks)

        # Top-K Forward KL
        t_scaled = t_logits / T
        s_scaled = s_logits / T
        _, topk_idx = t_scaled.topk(K_TOPK, dim=-1)
        t_topk = t_scaled.gather(-1, topk_idx)
        s_topk = s_scaled.gather(-1, topk_idx)
        t_logp_k = F.log_softmax(t_topk, dim=-1)
        s_logp_k = F.log_softmax(s_topk, dim=-1)
        t_prob_k = t_logp_k.exp()
        s_prob_k = s_logp_k.exp()
        fkl = (t_prob_k * (t_logp_k - s_logp_k)).sum(-1).mean() * (T ** 2)

        # Reverse KL
        rkl = (s_prob_k * (s_logp_k - t_logp_k)).sum(-1).mean() * (T ** 2)

        # Top-1 Margin Loss
        with torch.no_grad():
            t_top1 = t_logits.argmax(-1)
        s_at_t1 = s_logits.gather(-1, t_top1.unsqueeze(-1)).squeeze(-1)
        s_masked = s_logits.clone()
        s_masked.scatter_(-1, t_top1.unsqueeze(-1), float('-inf'))
        s_runner_up = s_masked.max(-1).values
        margin = s_at_t1 - s_runner_up
        margin_loss = F.relu(MARGIN_TARGET - margin).mean()

        loss = fkl + RKL_W * rkl + MARGIN_W * margin_loss

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 100 == 0:
        pct_correct = (margin > 0).float().mean().item() * 100
        print(f"  [{TAG}] step={step:6d}  loss={loss.item():.4f}  fkl={fkl.item():.4f}  "
              f"rkl={rkl.item():.4f}  mrg={margin_loss.item():.4f}  t1_hit={pct_correct:.0f}%  "
              f"T={T:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    if step > 0 and step % 2000 == 0:
        r = quick_eval()
        combined = 0.5 * r['top1'] + 0.5 * r['top10']
        prev_combined = 0.5 * best_top1 + 0.5 * best_top10
        new_best = combined > prev_combined
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  T1={r['top1']:.1f}%  T10={r['top10']:.1f}%  "
              f"ppl={r['ppl_ratio']:.3f}  combined={combined:.1f}{marker}", flush=True)
        if new_best:
            best_top1 = r['top1']
            best_top10 = r['top10']
            torch.save({
                'state_dict': student.state_dict(), 'step': step,
                'best_top1': best_top1, 'best_top10': best_top10,
                'h_inner': H_INNER, 'rank': R, 'n_passes': N_PASSES,
                'n_heads_inner': n_heads_inner,
                'total_train': total_train,
            }, BEST)
        torch.save({
            'state_dict': student.state_dict(), 'opt': opt.state_dict(),
            'step': step, 'best_top1': best_top1, 'best_top10': best_top10,
            'h_inner': H_INNER, 'rank': R, 'n_passes': N_PASSES,
        }, LATEST)

# Final eval
r = quick_eval()
combined = 0.5 * r['top1'] + 0.5 * r['top10']
prev_combined = 0.5 * best_top1 + 0.5 * best_top10
if combined > prev_combined:
    best_top1 = r['top1']
    best_top10 = r['top10']
    torch.save({
        'state_dict': student.state_dict(), 'step': STEPS,
        'best_top1': best_top1, 'best_top10': best_top10,
        'h_inner': H_INNER, 'rank': R, 'n_passes': N_PASSES,
        'n_heads_inner': n_heads_inner,
        'total_train': total_train,
    }, BEST)

print(f"\nDONE {TAG}: T1={best_top1:.2f}%  T10={best_top10:.2f}%")
print(f"  body={body_params / 1e6:.2f}M  decode={decode_params / 1e6:.2f}M  "
      f"vocab={head_params / 1e6:.2f}M  total={total_train / 1e6:.2f}M  "
      f"compression={teacher_full / total_train:.1f}x")
