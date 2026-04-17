"""
Fine-tune ASVD factored lm_head with KL distillation.

No TinyFRR body — this uses the TEACHER's latent directly through the
factored head. Tests how far we can push head-only quality.

ASVD r=1024 cold start: 71.0% top-1, 81.2% top-10.
Fine-tuning with KL should push significantly higher since the ASVD
matrices can adapt to actually minimize KL, not just minimize
reconstruction error.

Architecture: teacher embed → teacher layers → teacher norm → ASVD(V,U) → logits
Only V_proj [r, 2048] and U_out [V, r] are trainable.
"""
import lib.unbuffered
import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer

ap = argparse.ArgumentParser()
ap.add_argument('--r', type=int, default=1024, help='ASVD rank')
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--device', type=str, default='cuda:0')
args = ap.parse_args()

R = args.r
STEPS = args.steps
BATCH = 4
SEQ_LEN = args.seq_len
LR = 1e-4
WARMUP = 500
TAG = args.tag or f'asvd_r{R}_ft'
N_TEACHER_LAYERS = 28
DEVICE = args.device
T_START = 2.0
T_END = 1.0

CKPT_DIR = f'checkpoints_1.7b_{TAG}'
BEST = os.path.join(CKPT_DIR, 'best.pt')
LATEST = os.path.join(CKPT_DIR, 'latest.pt')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"ASVD Fine-tune: r={R}  steps={STEPS}  seq_len={SEQ_LEN}  tag={TAG}")

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
lm_head_w = gd['output.weight'].to(DEVICE).float()
del gd

# ==================== ASVD Initialization ====================
print("Computing ASVD initialization...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
Mcov = torch.zeros(H_OUTER, H_OUTER, device=DEVICE, dtype=torch.float32)
n_obs = 0
with torch.no_grad():
    calib_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (64,))
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
U_r = U[:, :R].contiguous()
S_r = S[:R]
Vt_r = Vt[:R, :].contiguous()
asvd_proj_init = (Vt_r @ Linv).contiguous()
asvd_out_init = (U_r * S_r.unsqueeze(0)).contiguous()
del Mcov, Lchol, Linv, WL, U, S, Vt, U_r, S_r, Vt_r

# Sanity check
with torch.no_grad():
    recon = asvd_out_init @ asvd_proj_init
    err = (recon - lm_head_w).norm() / lm_head_w.norm()
    print(f"  ASVD r={R}: recon error = {err:.4f}")

head_params = R * (vocab_size + H_OUTER)
print(f"  Trainable: {head_params/1e6:.2f}M  (full head: {lm_head_w.numel()/1e6:.1f}M, {lm_head_w.numel()/head_params:.1f}x shrink)")

# ==================== Trainable ASVD Head ====================
asvd_proj = nn.Linear(H_OUTER, R, bias=False).to(DEVICE)
asvd_out = nn.Linear(R, vocab_size, bias=False).to(DEVICE)
asvd_proj.weight.data.copy_(asvd_proj_init)
asvd_out.weight.data.copy_(asvd_out_init)
del asvd_proj_init, asvd_out_init

# Use 500M data if available
if os.path.exists('fineweb_edu_500M_tokens.pt'):
    print("  Using 500M token dataset", flush=True)
    all_tokens = torch.load('fineweb_edu_500M_tokens.pt', weights_only=True)
print(f"  tokens: {all_tokens.numel()/1e6:.0f}M")

opt = torch.optim.AdamW(
    list(asvd_proj.parameters()) + list(asvd_out.parameters()),
    lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

def lr_lambda(step):
    if step < WARMUP: return step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_step = 0
best_top1 = 0.0
best_top10 = 0.0
if os.path.exists(LATEST):
    ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
    asvd_proj.load_state_dict({'weight': ck['asvd_proj']})
    asvd_out.load_state_dict({'weight': ck['asvd_out']})
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    best_top1 = ck.get('best_top1', 0.0)
    best_top10 = ck.get('best_top10', 0.0)
    print(f"  resumed step={start_step}  best_top1={best_top1:.3f}  best_top10={best_top10:.3f}")


def current_temp(step):
    anneal_end = int(STEPS * 0.8)
    if step >= anneal_end: return T_END
    return T_START + (T_END - T_START) * (step / anneal_end)


@torch.no_grad()
def quick_eval(n=100):
    asvd_proj.eval(); asvd_out.eval()
    top1 = top10 = 0
    teacher_nll = student_nll = 0.0
    n_tok = 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (1,)).item()
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]; tgt = toks[0, 1:SEQ_LEN+1]
        t_logits, t_hs = teacher.forward(inp, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        latent = teacher.final_norm(t_hs[-1]).float()
        z = asvd_proj(latent)
        s_logits = asvd_out(z)

        t_logp = F.log_softmax(t_logits[0].float(), dim=-1)
        s_logp = F.log_softmax(s_logits[0].float(), dim=-1)
        teacher_nll += -t_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        student_nll += -s_logp[torch.arange(SEQ_LEN), tgt].sum().item()

        t_top1 = t_logits[0].argmax(-1)
        s_top1 = s_logits[0].argmax(-1)
        top1 += (t_top1 == s_top1).sum().item()

        for pos in range(SEQ_LEN):
            tt = set(t_logits[0, pos].topk(10).indices.tolist())
            st = set(s_logits[0, pos].topk(10).indices.tolist())
            top10 += len(tt & st) / 10.0
        n_tok += SEQ_LEN

    asvd_proj.train(); asvd_out.train()
    t_ppl = math.exp(teacher_nll / n_tok)
    s_ppl = math.exp(student_nll / n_tok)
    return top1/n_tok, top10/n_tok, s_ppl/t_ppl


# ==================== Train ====================
t0 = time.time()
asvd_proj.train(); asvd_out.train()
for step in range(start_step, STEPS):
    T = current_temp(step)
    starts_batch = torch.randint(0, all_tokens.numel() - SEQ_LEN, (BATCH,))
    toks = torch.stack([all_tokens[s:s+SEQ_LEN].long() for s in starts_batch]).to(DEVICE)

    with torch.no_grad():
        t_logits, t_hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        latent = teacher.final_norm(t_hs[-1]).float()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        z = asvd_proj(latent)
        s_logits = asvd_out(z)

        # Top-K KL
        K = 128
        t_scaled = t_logits / T; s_scaled = s_logits / T
        _, topk_idx = t_scaled.topk(K, dim=-1)
        t_topk = t_scaled.gather(-1, topk_idx)
        s_topk = s_scaled.gather(-1, topk_idx)
        t_logp = F.log_softmax(t_topk, dim=-1)
        s_logp = F.log_softmax(s_topk, dim=-1)
        t_prob = t_logp.exp()
        fkl = (t_prob * (t_logp - s_logp)).sum(-1).mean() * (T ** 2)

    opt.zero_grad()
    scaler.scale(fkl).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(
        list(asvd_proj.parameters()) + list(asvd_out.parameters()), 1.0)
    scaler.step(opt); scaler.update(); sched.step()

    if step % 100 == 0:
        print(f"  [{TAG}] step={step:6d}  fkl={fkl.item():.4f}  T={T:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    if step > 0 and step % 2000 == 0:
        t1, t10, ppl_ratio = quick_eval()
        new_best = t1 > best_top1
        marker = ' *** NEW BEST ***' if new_best else ''
        print(f"  EVAL step={step:6d}  top1={t1*100:.1f}%  top10={t10*100:.1f}%  ppl={ppl_ratio:.3f}{marker}", flush=True)
        if new_best:
            best_top1 = t1; best_top10 = t10
            torch.save({'asvd_proj': asvd_proj.weight.data,
                        'asvd_out': asvd_out.weight.data,
                        'step': step, 'best_top1': best_top1,
                        'best_top10': best_top10, 'rank': R}, BEST)
        torch.save({'asvd_proj': asvd_proj.weight.data,
                    'asvd_out': asvd_out.weight.data,
                    'opt': opt.state_dict(), 'step': step,
                    'best_top1': best_top1, 'best_top10': best_top10, 'rank': R}, LATEST)

# Final eval
t1, t10, ppl_ratio = quick_eval()
if t1 > best_top1:
    best_top1 = t1; best_top10 = t10
    torch.save({'asvd_proj': asvd_proj.weight.data,
                'asvd_out': asvd_out.weight.data,
                'step': STEPS, 'best_top1': best_top1,
                'best_top10': best_top10, 'rank': R}, BEST)

print(f"\nDONE {TAG}: top1={best_top1*100:.2f}%  top10={best_top10*100:.2f}%  ppl={ppl_ratio:.3f}")
print(f"  params={head_params/1e6:.2f}M  vs full {lm_head_w.numel()/1e6:.1f}M ({lm_head_w.numel()/head_params:.1f}x shrink)")
