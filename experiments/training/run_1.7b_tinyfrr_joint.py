"""
TinyFRR + ASVD Head — Joint Training.

THE big experiment: combine body compression (TinyFRR h_inner) with head
compression (ASVD rank-r factored lm_head) and fine-tune BOTH together.

Currently:
  - TinyFRR body h=128 alone → 43% top-1 (using full 311M lm_head)
  - ASVD head r=1024 alone  → 71% top-1 (using teacher latent)
  - Stacked without joint tuning → both hurt each other

Joint training loss:
  1. KL on factored logits (fwd + rev KL, T-schedule)
  2. Latent cosine+MSE vs teacher post-final-norm hidden state
  3. Body projections + FRR block + ASVD U,V matrices are ALL trainable

The student is: embed → proj_in → FRR → proj_out → norm → ASVD(V_r, U_r)
  - proj_in:  [2048, h_inner]  trainable
  - FRR block: shared block  trainable
  - proj_out: [h_inner, 2048]  trainable
  - norm:     [2048] frozen (teacher's)
  - V_proj:   [r, 2048]   trainable (ASVD input-side, initialized from ASVD)
  - U_proj:   [V, r]      trainable (ASVD output-side, initialized from ASVD)

Total trainable = TinyFRR body + r*(V+2048) for ASVD head.
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
ap.add_argument('--h', type=int, default=128, help='TinyFRR inner dim')
ap.add_argument('--r', type=int, default=512, help='ASVD rank for lm_head')
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--latent_w', type=float, default=0.5)
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
TAG = args.tag or f'h{H_INNER}_r{R}_joint'
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

print(f"TinyFRR+ASVD Joint: h={H_INNER} r={R}  steps={STEPS}  seq_len={SEQ_LEN}")
print(f"  tag={TAG}  device={DEVICE}  T:{T_START}->{T_END}  rkl_w={RKL_W}  lat_w={LATENT_W}")

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
lm_head_w = gd['output.weight'].to(DEVICE).float()
del gd

candidate_heads = [16, 8, 12, 4]
n_heads_inner = next((h for h in candidate_heads if H_INNER % h == 0), 4)

# ==================== ASVD Initialization ====================
print("computing ASVD initialization for lm_head factorization...")
# Collect latent covariance for activation-aware SVD
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
Mcov = torch.zeros(H_OUTER, H_OUTER, device=DEVICE, dtype=torch.float32)
n_obs = 0
with torch.no_grad():
    calib_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (48,))
    for s in calib_starts:
        toks = all_tokens[int(s):int(s)+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
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
# rank-r factors:  logits = Out @ Proj @ latent
# Proj = V_r^T @ L^{-1}  shape [r, H_OUTER]
# Out = U_r * S_r         shape [V, r]
U_r = U[:, :R].contiguous()
S_r = S[:R]
Vt_r = Vt[:R, :].contiguous()
asvd_proj_init = (Vt_r @ Linv).contiguous()       # [r, H_OUTER]
asvd_out_init = (U_r * S_r.unsqueeze(0)).contiguous()  # [V, r]
del Mcov, Lchol, Linv, WL, U, S, Vt, U_r, S_r, Vt_r

# Quick sanity: recon error of ASVD init
with torch.no_grad():
    recon = asvd_out_init @ asvd_proj_init
    err = (recon - lm_head_w).norm() / lm_head_w.norm()
    print(f"  ASVD r={R}: recon error = {err:.4f}")

asvd_head_params = R * (vocab_size + H_OUTER)
print(f"  ASVD head: r={R}  params={asvd_head_params/1e6:.2f}M  vs full head {lm_head_w.numel()/1e6:.1f}M  ({lm_head_w.numel()/asvd_head_params:.1f}x shrink)")


# ==================== Combined Model ====================
class TinyFRR_ASVD(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, rank,
                 embed_w, norm_w, asvd_proj, asvd_out):
        super().__init__()
        # Body: TinyFRR
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

        # Frozen teacher embed + norm
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

        # Head: ASVD factored lm_head (TRAINABLE)
        # asvd_proj: [r, h_outer] — projects latent down to rank-r
        # asvd_out:  [V, r]      — projects up to vocab
        self.asvd_proj = nn.Linear(h_outer, rank, bias=False)
        self.asvd_out = nn.Linear(rank, vocab, bias=False)
        self.asvd_proj.weight.data.copy_(asvd_proj)      # [r, h_outer]
        self.asvd_out.weight.data.copy_(asvd_out)         # [V, r]

    def forward(self, tokens, return_latent=False):
        x = self.embed(tokens).float()           # [B,S,H_outer]
        x = self.proj_in(x)                       # [B,S,H_inner]
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x = self.proj_out(x)                       # [B,S,H_outer]
        latent = self.norm_outer(x)                # post-norm
        z = self.asvd_proj(latent)                 # [B,S,r]
        logits = self.asvd_out(z)                  # [B,S,V]
        if return_latent:
            return logits, latent
        return logits


# ==================== Build student ====================
student = TinyFRR_ASVD(H_OUTER, H_INNER, n_heads_inner, vocab_size, R,
                        embed_w_outer, norm_w_outer,
                        asvd_proj_init, asvd_out_init).to(DEVICE)

# Warm-start body from best existing checkpoint
WARM_CANDIDATES = [
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_hq2/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_hq/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}_long/best.pt',
    f'checkpoints_1.7b_tinyfrr_h{H_INNER}/best.pt',
]
WARM_PATH = next((p for p in WARM_CANDIDATES if os.path.exists(p)), None)
if WARM_PATH:
    warm = torch.load(WARM_PATH, map_location=DEVICE, weights_only=False)
    # Only load body keys (proj_in, proj_out, inner.*) not head keys
    body_keys = {k: v for k, v in warm['state_dict'].items()
                 if not k.startswith('asvd_') and not k.startswith('lm_head')}
    missing, unexpected = student.load_state_dict(body_keys, strict=False)
    print(f"  WARM START body from {WARM_PATH}")
    print(f"    loaded {len(body_keys)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
else:
    print("  no body warm-start found; training from scratch")

# Param budget
body_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and not n.startswith('asvd_'))
head_params = sum(p.numel() for n, p in student.named_parameters()
                  if p.requires_grad and n.startswith('asvd_'))
total_train = body_params + head_params
teacher_full = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + lm_head_w.numel()
print(f"  body={body_params/1e6:.2f}M  head={head_params/1e6:.2f}M  total={total_train/1e6:.2f}M")
print(f"  teacher(core+head)={teacher_full/1e6:.1f}M  total_compression={teacher_full/total_train:.1f}x")

# ==================== Data ====================
# Auto-detect larger dataset
if os.path.exists('fineweb_edu_500M_tokens.pt'):
    print("  Using 500M token dataset", flush=True)
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
else:
    print("  Using 100M token dataset", flush=True)
    all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
print(f"  tokens: {all_tokens.numel()/1e6:.0f}M")

# ==================== Optimizer ====================
# Two param groups: body (higher LR) and head (lower LR, already well-initialized)
opt = torch.optim.AdamW([
    {'params': [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith('asvd_')],
     'lr': LR_MAX},
    {'params': [p for n, p in student.named_parameters() if p.requires_grad and n.startswith('asvd_')],
     'lr': LR_MAX * 0.1},  # 10x lower for head (already ASVD-initialized)
], betas=(0.9, 0.95), weight_decay=0.01)

def lr_lambda(step):
    if step < WARMUP: return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN / LR_MAX) + (1 - LR_MIN / LR_MAX) * cos

sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda, lr_lambda])
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_ppl_ratio = 999.0; best_top1 = 0.0; best_top10 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck['state_dict'], strict=False)
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_ppl_ratio = ck.get('best_ppl_ratio', 999.0)
    best_top1 = ck.get('best_top1', 0.0)
    best_top10 = ck.get('best_top10', 0.0)
    print(f"  resumed step={start_step}  best_ppl={best_ppl_ratio:.3f}  top1={best_top1:.3f}  top10={best_top10:.3f}")

def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end: return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


@torch.no_grad()
def quick_eval(n=80):
    student.eval()
    all_t10 = last_t10 = top1 = 0.0
    teacher_nll = student_nll = 0.0
    n_tok = 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (1,)).item()
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


# Keep full lm_head for teacher logit reference
teacher_lm_head = lm_head_w  # [V, H_OUTER]

# ==================== Train ====================
student.train()
t0 = time.time()
for step in range(start_step, STEPS):
    T = current_temp(step)
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
    toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts]).to(DEVICE)

    with torch.no_grad():
        t_logits, t_hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        t_latent = teacher.final_norm(t_hs[-1]).float()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        s_logits, s_latent = student(toks, return_latent=True)

        # Top-K filtered KL: only compute KL on teacher's top-K tokens per position
        # This focuses capacity on matching the tokens that matter and is 10x faster
        K = 128  # top-128 covers >99% of teacher probability mass
        t_scaled = t_logits / T
        s_scaled = s_logits / T

        # Get teacher top-K indices
        _, topk_idx = t_scaled.topk(K, dim=-1)  # [B, S, K]

        # Gather logits for top-K tokens
        t_topk = t_scaled.gather(-1, topk_idx)  # [B, S, K]
        s_topk = s_scaled.gather(-1, topk_idx)  # [B, S, K]

        # Softmax over K tokens only
        t_logp_k = F.log_softmax(t_topk, dim=-1)
        s_logp_k = F.log_softmax(s_topk, dim=-1)
        t_prob_k = t_logp_k.exp()
        s_prob_k = s_logp_k.exp()

        fkl = (t_prob_k * (t_logp_k - s_logp_k)).sum(-1).mean() * (T ** 2)
        rkl = (s_prob_k * (s_logp_k - t_logp_k)).sum(-1).mean() * (T ** 2)

        # Latent matching (cosine + normalized MSE)
        tl = t_latent.float(); sl = s_latent.float()
        cos = 1.0 - F.cosine_similarity(sl, tl, dim=-1).mean()
        mse = F.mse_loss(sl, tl) / (tl.pow(2).mean() + 1e-6)
        lat = cos + 0.1 * mse

        loss = fkl + RKL_W * rkl + LATENT_W * lat

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    scaler.step(opt); scaler.update(); sched.step()

    if step % 100 == 0:
        lr = opt.param_groups[0]['lr']
        print(f"  [{TAG}] step={step:6d}  loss={loss.item():.4f}  fkl={fkl.item():.4f}  rkl={rkl.item():.4f}  lat={lat.item():.4f}  T={T:.2f}  lr={lr:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if step > 0 and step % 2000 == 0:
        all10, last10, t1, ppl_ratio = quick_eval()
        quality = 100.0 / ppl_ratio
        new_best = ppl_ratio < best_ppl_ratio
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  all-T10={all10*100:.1f}%  last-T10={last10*100:.1f}%  top1={t1*100:.1f}%  ppl={ppl_ratio:.3f}  quality={quality:.1f}%{marker}", flush=True)
        if new_best:
            best_ppl_ratio = ppl_ratio; best_top1 = t1; best_top10 = last10
            torch.save({'state_dict': student.state_dict(), 'step': step,
                        'best_ppl_ratio': best_ppl_ratio, 'best_top1': best_top1,
                        'best_top10': best_top10, 'h_inner': H_INNER, 'rank': R,
                        'n_heads_inner': n_heads_inner, 'body_params': body_params,
                        'head_params': head_params, 'total_train': total_train}, BEST)
        torch.save({'state_dict': student.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best_ppl_ratio': best_ppl_ratio,
                    'best_top1': best_top1, 'best_top10': best_top10,
                    'h_inner': H_INNER, 'rank': R}, LATEST)

# Final eval
all10, last10, t1, ppl_ratio = quick_eval()
if ppl_ratio < best_ppl_ratio:
    best_ppl_ratio = ppl_ratio; best_top1 = t1; best_top10 = last10
    torch.save({'state_dict': student.state_dict(), 'step': STEPS,
                'best_ppl_ratio': best_ppl_ratio, 'best_top1': best_top1,
                'best_top10': best_top10, 'h_inner': H_INNER, 'rank': R,
                'n_heads_inner': n_heads_inner, 'body_params': body_params,
                'head_params': head_params, 'total_train': total_train}, BEST)

print(f"\nDONE {TAG}: top1={best_top1*100:.2f}%  top10={best_top10*100:.2f}%  ppl={best_ppl_ratio:.3f}  quality={100/best_ppl_ratio:.1f}%")
print(f"  body={body_params/1e6:.2f}M  head={head_params/1e6:.2f}M  total={total_train/1e6:.2f}M  compression={teacher_full/total_train:.1f}x")
