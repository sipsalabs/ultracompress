"""
Standard Hinton-2015-style KD baseline at matched parameter count.

Purpose: Answer the skeptical reviewer's first question --
"Does your nested-fractal + entropy-weighted loss actually matter, or does
ANY small student hit the same numbers at 1.5M params?"

Architecture: proj_in(2048->H) -> N standard transformer blocks (GQA+SwiGLU)
-> proj_out(H->2048) -> frozen teacher norm + lm_head. Blocks are STANDARD,
no fractal structure, no nested iterations, no per-iter learned alpha/gamma.

Loss: classic Hinton 2015 KD
    L = alpha * KL(student_logits/T || teacher_logits/T) * T^2
        + (1 - alpha) * CE(student_logits, targets)
No entropy weighting, no margin, no latent matching.

Tags saved: checkpoints_1.7b_baseline_kd_{tag}/best.pt with schema compatible
with hires_eval / wikitext_eval (h_inner=H, state_dict keyed for TinyFRR's
proj_in/proj_out/norm_outer but with a STANDARD inner module).

Do NOT launch while both GPUs are occupied by HQ6 / HQ7 / combined-stack.
Queue after HQ7 chain completes.

Usage:
    python run_baseline_distill.py --h 256 --n_layers 2 --steps 80000 \
        --device cuda:0 --tag baseline_h256_L2
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

from ultracompress.inference import ModelConfig, MiniTransformer

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, default=256, help='inner hidden dim')
ap.add_argument('--n_layers', type=int, default=2, help='standard transformer blocks')
ap.add_argument('--n_heads', type=int, default=16)
ap.add_argument('--steps', type=int, default=80000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--temperature', type=float, default=2.0)
ap.add_argument('--alpha', type=float, default=0.9, help='KD weight vs CE')
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--tag', type=str, required=True)
ap.add_argument('--eval_every', type=int, default=2000)
ap.add_argument('--save_every', type=int, default=2000)
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
BATCH = args.batch
N_TEACHER_LAYERS = 28

CKPT_DIR = f'checkpoints_1.7b_baseline_kd_{args.tag}'
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------- teacher ----------
print("Loading teacher (Qwen3-1.7B)...")
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
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd


# ---------- student: standard stack of transformer blocks ----------
class StdBlock(nn.Module):
    """Standard pre-norm transformer block: RMSNorm->MHA->residual->RMSNorm->SwiGLU->residual."""
    def __init__(self, h, n_heads, ff_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = h // n_heads
        assert h % n_heads == 0
        self.norm1 = nn.RMSNorm(h)
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.norm2 = nn.RMSNorm(h)
        ff = h * ff_mult
        self.w_gate = nn.Linear(h, ff, bias=False)
        self.w_up = nn.Linear(h, ff, bias=False)
        self.w_down = nn.Linear(ff, h, bias=False)

    def forward(self, x):
        B, T, H = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, H)
        x = x + self.o(y)
        h2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
        return x


class BaselineStudent(nn.Module):
    def __init__(self, h_outer, h, n_layers, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h, bias=False)
        self.blocks = nn.ModuleList([StdBlock(h, n_heads) for _ in range(n_layers)])
        self.proj_out = nn.Linear(h, h_outer, bias=False)
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x_outer = self.embed(tokens).float()
        x = self.proj_in(x_outer)
        for blk in self.blocks:
            x = blk(x)
        x_outer = self.proj_out(x)
        x_outer = self.norm_outer(x_outer)
        return F.linear(x_outer, self.lm_head_w)


student = BaselineStudent(H_OUTER, args.h, args.n_layers, args.n_heads,
                          vocab_size, embed_w, lm_head_w, norm_w).to(DEVICE)
n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"Baseline student: h={args.h}  n_layers={args.n_layers}  params={n_params/1e6:.3f}M")


# ---------- data ----------
DATA_500M = 'fineweb_edu_500M_tokens.pt'
DATA_100M = 'fineweb_edu_100M_tokens.pt'
data_path = DATA_500M if os.path.exists(DATA_500M) else DATA_100M
all_tokens = torch.load(data_path, weights_only=True)
print(f"  data: {data_path}  {all_tokens.numel()/1e6:.0f}M tokens")


# ---------- eval ----------
@torch.no_grad()
def quick_eval(n=100):
    student.eval()
    eval_starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (n,))
    t1, t10, tnll, snll, nt = 0, 0.0, 0.0, 0.0, 0
    for s in eval_starts:
        s = int(s.item())
        toks = all_tokens[s:s + SEQ_LEN + 1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]
        tgt = toks[0, 1:SEQ_LEN + 1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)[0]
        sl = student(inp)[0]
        for pos in range(SEQ_LEN):
            tt = tl[pos].topk(10).indices
            st = sl[pos].topk(10).indices
            t1 += int(st[0] == tt[0])
            t10 += len(set(tt.tolist()) & set(st.tolist())) / 10.0
        tnll += -F.log_softmax(tl, -1).gather(-1, tgt.unsqueeze(-1)).sum().item()
        snll += -F.log_softmax(sl, -1).gather(-1, tgt.unsqueeze(-1)).sum().item()
        nt += SEQ_LEN
    student.train()
    t_ppl = math.exp(tnll / nt)
    s_ppl = math.exp(snll / nt)
    return {'T1': t1 / (n * SEQ_LEN), 'T10': t10 / (n * SEQ_LEN),
            't_ppl': t_ppl, 's_ppl': s_ppl, 'ratio': s_ppl / t_ppl}


# ---------- train ----------
opt = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=args.lr, total_steps=args.steps,
    pct_start=0.05, div_factor=25, final_div_factor=10,
)

T = args.temperature
alpha = args.alpha
best_quality = -1.0
t0 = time.time()

print(f"Training {args.steps} steps | T={T} alpha={alpha} lr={args.lr}")
for step in range(1, args.steps + 1):
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (BATCH,))
    toks = torch.stack([all_tokens[s:s + SEQ_LEN + 1].long() for s in starts]).to(DEVICE)
    inp = toks[:, :SEQ_LEN]
    tgt = toks[:, 1:SEQ_LEN + 1]

    with torch.no_grad():
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)  # (B, T, V)
    sl = student(inp)

    t_soft = F.log_softmax(tl / T, dim=-1)
    s_soft = F.log_softmax(sl / T, dim=-1)
    kd = F.kl_div(s_soft, t_soft, reduction='batchmean', log_target=True) * (T * T)
    ce = F.cross_entropy(sl.reshape(-1, vocab_size), tgt.reshape(-1))
    loss = alpha * kd + (1 - alpha) * ce

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()
    sched.step()

    if step % 100 == 0:
        el = time.time() - t0
        print(f"  [baseline_kd_{args.tag}] step={step:6d}  loss={loss.item():.4f}  "
              f"kd={kd.item():.4f}  ce={ce.item():.4f}  lr={opt.param_groups[0]['lr']:.5f}  ({el:.0f}s)")

    if step % args.eval_every == 0 or step == args.steps:
        ev = quick_eval(n=100)
        quality = 0.5 * ev['T10'] + 0.5 * (1.0 / ev['ratio'])
        print(f"  [eval step={step}] T1={ev['T1']*100:.2f}%  T10={ev['T10']*100:.2f}%  "
              f"ppl_ratio={ev['ratio']:.3f}  Q={quality*100:.2f}")
        if quality > best_quality:
            best_quality = quality
            torch.save({
                'state_dict': student.state_dict(),
                'h_inner': args.h,
                'n_layers': args.n_layers,
                'n_heads_inner': args.n_heads,
                'step': step,
                'quality': quality,
                'T1': ev['T1'], 'T10': ev['T10'],
                'ppl_ratio': ev['ratio'],
                'config': vars(args),
            }, f'{CKPT_DIR}/best.pt')
            print(f"    -> saved best.pt (Q={quality*100:.2f})")

print(f"\nDone. Best Q = {best_quality*100:.2f}   checkpoint: {CKPT_DIR}/best.pt")
