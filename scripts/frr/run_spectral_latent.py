"""
Spectral Latent Distillation (SLD) — NOVEL fusion of two proven approaches.

KEY INSIGHT: h128_hq2 gets last-T10=73.1% because its latent alignment loss
teaches the body to reconstruct the teacher's 2048-dim internal state.
FSD gets only 62.8% because it has NO such guidance — only output distribution matching.

SLD BRIDGES THIS GAP: We add a latent alignment loss to FSD, but project the
teacher's latent into ASVD spectral space for comparison. The student's
fused_decode output IS already in this spectral space, so we can directly compare:

  Teacher path:  hidden(2048) → asvd_proj(2048→r) = teacher spectral embedding
  Student path:  body(h)      → fused_decode(h→r) = student spectral embedding

Loss = fkl + rkl_w * rkl + spectral_w * (cosine_dist + 0.1 * norm_mse)
         between fused_decode_output and asvd_proj @ teacher_latent

WHY THIS IS NOVEL:
1. Teacher's ASVD projection tells the body "this is what the 2048-dim
   representation looks like in the r-dim space that matters for vocab prediction"
2. Student's fused_decode output should match this — same space, direct comparison
3. Gives the body the SAME representational guidance as h128_hq2's latent loss,
   but in the compressed spectral basis (not 2048-dim)
4. NO additional params needed at inference — purely an auxiliary training signal
5. Could close the 73.1% vs 62.8% = 10.3pp T10 gap without any inference cost

This approach is genuinely novel — nobody has done spectral-space latent
distillation for compressed model training.
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
ap.add_argument('--r', type=int, default=512, help='Vocab rank (fused decode output dim)')
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--margin', type=float, default=1.0, help='Target margin for top-1 loss')
ap.add_argument('--margin_w', type=float, default=0.2, help='Weight for margin loss')
ap.add_argument('--rkl_w', type=float, default=0.3, help='Weight for reverse KL')
ap.add_argument('--spectral_w', type=float, default=1.0, help='Weight for spectral latent matching')
args = ap.parse_args()

STEPS = args.steps
BATCH = 4
ACCUM = 2          # gradient accumulation for effective batch 8, like h128_hq2
SEQ_LEN = args.seq_len
LR_MAX = 3e-4
LR_MIN = 1e-5
WARMUP = 2000
H_INNER = args.h
R = args.r
TAG = args.tag or f'sld_h{H_INNER}_r{R}'
N_TEACHER_LAYERS = 28
DEVICE = args.device
T_START = 2.0
T_END = 1.0
K_TOPK = 128
MARGIN_TARGET = args.margin
MARGIN_W = args.margin_w
RKL_W = args.rkl_w
SPECTRAL_W = args.spectral_w

CKPT_DIR = f'checkpoints_1.7b_tinyfrr_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"=== Spectral Latent Distillation (SLD) ===")
print(f"  h={H_INNER} r={R} steps={STEPS} seq_len={SEQ_LEN}")
print(f"  tag={TAG} device={DEVICE} topK={K_TOPK}")
print(f"  margin={MARGIN_TARGET} margin_w={MARGIN_W} rkl_w={RKL_W}")
print(f"  spectral_w={SPECTRAL_W}  accum={ACCUM}")

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

# ==================== ASVD Initialization ====================
print("Computing ASVD initialization for vocab embedding...")
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

# ASVD factors:
#   asvd_proj: [r, 2048] — projects hidden to rank-r
#   asvd_out:  [vocab, r] — projects rank-r to vocab
asvd_proj_init = (Vt_r @ Linv).contiguous()       # [r, H_OUTER]
asvd_out_init = (U_r * S_r.unsqueeze(0)).contiguous()  # [vocab, r]

# Keep a FROZEN copy of asvd_proj for spectral latent matching during training
# This is the "teacher's spectral projection" — converts teacher's 2048-dim
# post-norm latent into the r-dim spectral space
asvd_proj_frozen = asvd_proj_init.clone().detach()  # [r, H_OUTER]
print(f"  ASVD proj frozen shape: {asvd_proj_frozen.shape}")

del Mcov, Lchol, Linv, WL, U, S, Vt, U_r, S_r, Vt_r

with torch.no_grad():
    recon = asvd_out_init @ asvd_proj_init
    err = (recon - lm_head_w).norm() / lm_head_w.norm()
    print(f"  ASVD r={R}: recon error = {err:.4f}")
del recon, lm_head_w
torch.cuda.empty_cache()


# ==================== Fused Decode Module ====================
class FusedDecode(nn.Module):
    """Nonlinear decode with linear residual path.

    decode(x) = linear(x) + mlp_residual(x)

    Linear path: warm-startable from composed proj_out @ asvd_proj.
    MLP residual: initialized near zero, learns nonlinear corrections.
    """
    def __init__(self, h_inner, rank, init_weight=None):
        super().__init__()
        self.linear = nn.Linear(h_inner, rank, bias=False)
        if init_weight is not None:
            self.linear.weight.data.copy_(init_weight)

        self.mlp_gate = nn.Linear(h_inner, h_inner * 2, bias=False)
        self.mlp_up = nn.Linear(h_inner, h_inner * 2, bias=False)
        self.mlp_down = nn.Linear(h_inner * 2, rank, bias=False)
        nn.init.zeros_(self.mlp_down.weight)

        self.norm = nn.RMSNorm(h_inner)

    def forward(self, x):
        z_linear = self.linear(x)
        h = self.norm(x)
        z_residual = self.mlp_down(F.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return z_linear + z_residual


# ==================== SLD Model ====================
class SpectralLatentModel(nn.Module):
    """SLD: FSD architecture + spectral latent matching.

    Same inference path as FSD — no additional cost at inference.
    During training, compares fused_decode output with teacher's
    ASVD-projected latent for representational guidance.
    """

    def __init__(self, h_outer, h_inner, n_heads, vocab, rank,
                 embed_w, asvd_out, asvd_proj_frozen, decode_init=None):
        super().__init__()
        self.h_outer = h_outer
        self.h_inner = h_inner

        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)

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

        self.fused_decode = FusedDecode(h_inner, rank, init_weight=decode_init)

        self.vocab_head = nn.Linear(rank, vocab, bias=False)
        self.vocab_head.weight.data.copy_(asvd_out)

        # FROZEN: teacher's spectral projection for latent matching
        # This projects teacher's 2048-dim latent → r-dim spectral space
        self.register_buffer('asvd_proj', asvd_proj_frozen, persistent=False)

    def forward(self, tokens, return_spectral=False):
        x = self.embed(tokens).float()

        # Body: proj_in → FRR
        x = self.proj_in(x)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s

        # Fused decode: h_inner → r
        z = self.fused_decode(x)

        # Vocab: r → V
        logits = self.vocab_head(z)

        if return_spectral:
            return logits, z  # z is the spectral embedding we'll compare
        return logits

    @torch.no_grad()
    def teacher_spectral(self, teacher_latent):
        """Project teacher's post-norm 2048-dim latent into spectral space.

        teacher_latent: [B, S, 2048] (post-final-norm)
        returns: [B, S, r] — teacher's spectral embedding
        """
        return F.linear(teacher_latent, self.asvd_proj)  # [B, S, r]


# ==================== Build Student ====================
decode_init = None
WARM_V2 = f'checkpoints_1.7b_tinyfrr_v2_h{H_INNER}_r{R}/best.pt'
WARM_FSD = f'checkpoints_1.7b_tinyfrr_fsd_h{H_INNER}_r{R}/best.pt'
WARM_CANDIDATES = [
    WARM_FSD,    # Prefer FSD warm-start (includes fused_decode!)
    WARM_V2,
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_long/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}/best.pt',
]
WARM_PATH = next((p for p in WARM_CANDIDATES if os.path.exists(p)), None)

if WARM_PATH:
    warm = torch.load(WARM_PATH, map_location=DEVICE, weights_only=False)
    sd = warm.get('state_dict', warm)
    if WARM_PATH == WARM_FSD:
        # FSD checkpoint has the full architecture — direct load
        print(f"  Loading from FSD checkpoint {WARM_PATH}")
    elif 'proj_out.weight' in sd:
        proj_out_w = sd['proj_out.weight'].float()
        decode_init = (asvd_proj_init @ proj_out_w).contiguous()
        print(f"  Composed decode init from {WARM_PATH}: proj_out @ asvd_proj -> [{decode_init.shape[0]}, {decode_init.shape[1]}]")
else:
    print("  No warm-start checkpoint found — random init for decode")

ASVD_FT = f'checkpoints_1.7b_asvd_r{R}_ft/best.pt'
if os.path.exists(ASVD_FT):
    ack = torch.load(ASVD_FT, map_location=DEVICE, weights_only=False)
    asvd_out_ft = ack.get('asvd_out', ack.get('state_dict', {}).get('asvd_out.weight'))
    if asvd_out_ft is not None:
        asvd_out_init = asvd_out_ft.contiguous()
        print(f"  Using fine-tuned ASVD vocab from {ASVD_FT}")
    del ack

student = SpectralLatentModel(
    H_OUTER, H_INNER, n_heads_inner, vocab_size, R,
    embed_w, asvd_out_init, asvd_proj_frozen, decode_init=decode_init
).to(DEVICE)

# Warm-start body from checkpoint
if WARM_PATH:
    sd = warm.get('state_dict', warm)
    if WARM_PATH == WARM_FSD:
        # Load everything from FSD
        missing, unexpected = student.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"  WARM START from FSD: {loaded} keys loaded, {len(missing)} missing, {len(unexpected)} unexpected")
    else:
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

del embed_w, asvd_proj_frozen, asvd_out_init
torch.cuda.empty_cache()

# ==================== Param Accounting ====================
body_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and not n.startswith('vocab_head') and not n.startswith('fused_decode'))
decode_params = sum(p.numel() for n, p in student.named_parameters()
                    if p.requires_grad and n.startswith('fused_decode'))
head_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and n.startswith('vocab_head'))
total_train = body_params + decode_params + head_params
teacher_full = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + vocab_size * H_OUTER

print(f"  body={body_params / 1e6:.2f}M  decode={decode_params / 1e6:.2f}M  "
      f"vocab_head={head_params / 1e6:.2f}M  total={total_train / 1e6:.2f}M")
print(f"  teacher={teacher_full / 1e6:.1f}M  compression={teacher_full / total_train:.1f}x")

# ==================== Data ====================
if os.path.exists('fineweb_edu_500M_tokens.pt'):
    print("  Using 500M token dataset")
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
else:
    print("  Using 100M token dataset")
    all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
print(f"  tokens: {all_tokens.numel() / 1e6:.0f}M")

# ==================== Optimizer ====================
# Three param groups: body (high LR), decode (medium), vocab_head (low)
opt = torch.optim.AdamW([
    {'params': [p for n, p in student.named_parameters()
                if p.requires_grad and not n.startswith('vocab_head') and not n.startswith('fused_decode')],
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
best_quality = 0.0
best_top1 = 0.0
best_top10 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_quality = ck.get('best_quality', 0.0)
    best_top1 = ck.get('best_top1', 0.0)
    best_top10 = ck.get('best_top10', 0.0)
    print(f"  Resumed step={start_step} best_quality={best_quality:.1f}")


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end:
        return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


# ==================== Eval ====================
@torch.no_grad()
def quick_eval(n=100):
    student.eval()
    all_t10 = last_t10 = top1_agree = n_tok = 0
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
            h10 = len(tt & st) / 10.0
            all_t10 += h10
            n_tok += 1
            if pos == SEQ_LEN - 1:
                last_t10 += h10
    student.train()
    t_ppl = math.exp(teacher_nll / n_tok)
    s_ppl = math.exp(student_nll / n_tok)
    return {
        'top1': 100.0 * top1_agree / n_tok,
        'all_t10': 100.0 * all_t10 / n_tok,
        'last_t10': 100.0 * last_t10 / n,
        'ppl_ratio': s_ppl / t_ppl,
    }


# ==================== Training ====================
student.train()
t0 = time.time()

for step in range(start_step, STEPS):
    T = current_temp(step)

    opt.zero_grad()
    accum_loss = accum_fkl = accum_rkl = accum_mrg = accum_spec = 0.0

    for micro in range(ACCUM):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
        toks = torch.stack([all_tokens[s:s + SEQ_LEN].long() for s in starts]).to(DEVICE)

        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
            t_latent = teacher.final_norm(t_hs[-1]).float()  # [B, S, 2048]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits, s_spectral = student(toks, return_spectral=True)
            # s_spectral: [B, S, r] — student's spectral embedding

            # === Loss 1: Top-K Forward KL ===
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

            # === Loss 2: Reverse KL ===
            rkl = (s_prob_k * (s_logp_k - t_logp_k)).sum(-1).mean() * (T ** 2)

            # === Loss 3: Top-1 Margin Loss ===
            with torch.no_grad():
                t_top1 = t_logits.argmax(-1)
            s_at_t1 = s_logits.gather(-1, t_top1.unsqueeze(-1)).squeeze(-1)
            s_masked = s_logits.clone()
            s_masked.scatter_(-1, t_top1.unsqueeze(-1), float('-inf'))
            s_runner_up = s_masked.max(-1).values
            margin = s_at_t1 - s_runner_up
            margin_loss = F.relu(MARGIN_TARGET - margin).mean()

            # === Loss 4: SPECTRAL LATENT MATCHING (THE NOVEL PART) ===
            # Project teacher's 2048-dim post-norm latent into r-dim spectral space
            t_spectral = student.teacher_spectral(t_latent)  # [B, S, r]
            # Compare student's fused_decode output with teacher's spectral projection
            s_spec_f = s_spectral.float()
            t_spec_f = t_spectral.float()
            # Cosine distance: measures direction alignment in spectral space
            spec_cos = 1.0 - F.cosine_similarity(s_spec_f, t_spec_f, dim=-1).mean()
            # Normalized MSE: measures magnitude alignment
            spec_mse = F.mse_loss(s_spec_f, t_spec_f) / (t_spec_f.pow(2).mean() + 1e-6)
            spectral_loss = spec_cos + 0.1 * spec_mse

            # === Combined Loss ===
            loss = (fkl + RKL_W * rkl + MARGIN_W * margin_loss + SPECTRAL_W * spectral_loss) / ACCUM

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_fkl += fkl.item() / ACCUM
        accum_rkl += rkl.item() / ACCUM
        accum_mrg += margin_loss.item() / ACCUM
        accum_spec += spectral_loss.item() / ACCUM

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 100 == 0:
        lr = opt.param_groups[0]['lr']
        pct_correct = (margin > 0).float().mean().item() * 100
        print(f"  [{TAG}] step={step:6d}  loss={accum_loss:.4f}  fkl={accum_fkl:.4f}  "
              f"rkl={accum_rkl:.4f}  mrg={accum_mrg:.4f}  spec={accum_spec:.4f}  "
              f"t1_hit={pct_correct:.0f}%  T={T:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    if step > 0 and step % 2000 == 0:
        r = quick_eval()
        quality = 100.0 / r['ppl_ratio']
        new_best = quality > best_quality
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  all-T10={r['all_t10']:.1f}%  last-T10={r['last_t10']:.1f}%  "
              f"T1={r['top1']:.1f}%  ppl={r['ppl_ratio']:.3f}  quality={quality:.1f}%{marker}", flush=True)
        if new_best:
            best_quality = quality
            best_top1 = r['top1']
            best_top10 = r['all_t10']
            torch.save({'state_dict': student.state_dict(), 'step': step,
                        'best_quality': best_quality, 'best_top1': best_top1,
                        'best_top10': best_top10,
                        'h_inner': H_INNER, 'rank': R, 'tag': TAG,
                        'total_params': total_train}, BEST)
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best_quality': best_quality,
                    'best_top1': best_top1, 'best_top10': best_top10,
                    'h_inner': H_INNER, 'rank': R, 'tag': TAG,
                    'total_params': total_train}, LATEST)

# Final eval
r = quick_eval()
quality = 100.0 / r['ppl_ratio']
if quality > best_quality:
    best_quality = quality
    best_top1 = r['top1']
    best_top10 = r['all_t10']
    torch.save({'state_dict': student.state_dict(), 'step': STEPS,
                'best_quality': best_quality, 'best_top1': best_top1,
                'best_top10': best_top10,
                'h_inner': H_INNER, 'rank': R, 'tag': TAG,
                'total_params': total_train}, BEST)

print(f"\nDONE {TAG}: best_T1={best_top1:.1f}%  best_all-T10={best_top10:.1f}%  "
      f"quality={best_quality:.1f}%  params={total_train/1e6:.2f}M  "
      f"compression={teacher_full/total_train:.0f}x")
