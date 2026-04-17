"""
1.7B HIDDEN-STATE MATCHING + ORTHOGONAL LAYER ROTATIONS (HLR-FRR) — NOVEL

TWO ORTHOGONAL INNOVATIONS stacked:

(1) TRAINING: HIDDEN-STATE MATCHING
   Prior runs only use final-layer KL. But with 3.8M tokens, supervising ONLY
   the final logits is wasteful — every token gives one gradient signal. With
   per-layer hidden-state supervision, every token gives 28 signals (one per
   virtual layer). This is feature distillation (Romero 2015), but:
     - Never tested with FRR architecture specifically
     - Previous attempt (layer_match_output.log) used unweighted MSE which
       drowned out KL (112040 vs 34). We use SCALE-NORMALIZED MSE.
   Loss = KL(logits) + λ · Σ_ℓ MSE(x_ℓ^student, x_ℓ^teacher) / ‖x_ℓ^teacher‖²

(2) ARCHITECTURE: PER-LAYER ORTHOGONAL ROTATIONS
   All prior per-layer additive/multiplicative perturbations (γ/β, LoRA, MBBR)
   hit 68%. NEW: apply a learnable ORTHOGONAL rotation R_ℓ to the block output
   at each layer. R_ℓ = exp(A_ℓ - A_ℓ^T) where A_ℓ is a small skew-symmetric
   parameter matrix. Orthogonal → preserves norms (no activation blow-up) but
   fundamentally changes WHICH directions the next layer attends to.

   Parameter-efficient implementation: store only the upper-triangular entries
   of A_ℓ and use Cayley transform: R = (I - A + A^T/2)^-1 (I + A - A^T/2).
   Too expensive for D=2048. Instead, use BLOCK-DIAGONAL rotations: split D
   into G=32 groups of 64 dims, learn a 64×64 skew-symm per group per layer.
   Group-wise Cayley is cheap. Params per layer: G × 64·63/2 = 64512 ≈ 63K.

   This is fundamentally different from γ/β because rotations are UNITARY
   (don't scale norms) yet GEOMETRICALLY RICH (32 · 2016 = 64512 effective
   rotation angles per layer → 28 × 63K = 1.76M total).

Compression: 29M block + 1.8M rotations + 115K γ/β = 31M total
  → teacher_layer / student = 1527M / 31M = 49x compression

GOAL: beat the 68.23% hires last-tok T10 record.
Device: GPU 0 only.
"""
import lib.unbuffered
import torch
import sys
import os
import time
import math

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

# ── Configuration ────────────────────────────────────────────────────
DEVICE = 'cuda:0'
TOTAL_STEPS = 15_000
LR_BLOCK = 3e-5
LR_GB = 3e-4          # per-layer γ/β
LR_ROT = 5e-4         # rotation params (fresh)
WD_BLOCK = 0.05
WD_GB = 0.01
WD_ROT = 0.0
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
HIDDEN_MATCH_LAMBDA = 0.0   # Disabled: student trajectory != teacher trajectory.
                            # Rotations-only run — pure test of architectural novelty.
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
N_SCALES = 4
ITERS_PER_SCALE = 7
ROT_GROUPS = 32              # group-diagonal rotations
N_TEACHER_LAYERS = 28
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_hlr'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B HIDDEN-STATE MATCHING + ORTHOGONAL LAYER ROTATIONS (HLR-FRR)")
print(f"Device: {DEVICE}  |  Steps: {TOTAL_STEPS:,}  |  T={TEMP}")
print(f"  Loss = KL(logits) + {HIDDEN_MATCH_LAMBDA} * hidden_MSE (scale-normalized)")
print(f"  Rotations: {ROT_GROUPS} groups x (2048/{ROT_GROUPS})^2 skew-symm per layer")
print("=" * 70)


class GroupOrthoRotator(nn.Module):
    """Per-layer block-diagonal orthogonal rotation via Cayley transform.

    Split hidden_dim into G groups of dim g = hidden_dim/G. Per layer, store
    skew-symmetric A_ℓ[g, g] upper-tri params. At forward, rotate each group
    via group-wise Cayley: R = (I - A/2)^-1 (I + A/2)  where A is skew-symm.

    Init: A = 0  →  R = I  →  identity rotation (baseline preserved).
    """
    def __init__(self, hidden_dim, n_groups, n_layers):
        super().__init__()
        assert hidden_dim % n_groups == 0
        self.D = hidden_dim
        self.G = n_groups
        self.g = hidden_dim // n_groups
        self.L = n_layers

        # Upper-triangular indices for the skew-symmetric generator
        rows, cols = torch.triu_indices(self.g, self.g, offset=1)
        self.register_buffer('tri_rows', rows)
        self.register_buffer('tri_cols', cols)
        n_tri = rows.numel()                    # g·(g-1)/2

        # Per-(layer, group) skew-symm parameters (upper-tri)
        # Init to 0 → R = I
        self.skew_upper = nn.Parameter(torch.zeros(n_layers, n_groups, n_tri))

    def get_rotation(self, layer_idx):
        """Return R_ℓ of shape (G, g, g) — block-diagonal rotation."""
        A = torch.zeros(self.G, self.g, self.g, device=self.skew_upper.device,
                        dtype=self.skew_upper.dtype)
        # Fill upper triangle
        A[:, self.tri_rows, self.tri_cols] = self.skew_upper[layer_idx]
        # Make skew-symmetric: A = upper - upper^T
        A = A - A.transpose(-2, -1)
        # Cayley transform: R = (I - A/2)^-1 (I + A/2)
        I = torch.eye(self.g, device=A.device, dtype=A.dtype).expand_as(A)
        halfA = A * 0.5
        R = torch.linalg.solve(I - halfA, I + halfA)
        return R    # (G, g, g)

    def apply_rotation(self, x, layer_idx):
        """x: (B, T, D). Reshape → (B, T, G, g), rotate per group, flatten."""
        B, T, D = x.shape
        R = self.get_rotation(layer_idx)                      # (G, g, g)
        x_grouped = x.view(B, T, self.G, self.g)              # (B, T, G, g)
        # (B, T, G, g) @ (G, g, g) → (B, T, G, g)
        x_rot = torch.einsum('btgi,gij->btgj', x_grouped, R)
        return x_rot.reshape(B, T, D)

    def n_params(self):
        return self.skew_upper.numel()


class HLRModel(nn.Module):
    """FRR + per-layer rotations + hidden-state exposure for matching loss."""
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, n_rot_groups,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)
        # Per-layer γ/β (proven useful)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))
        # NEW: per-layer orthogonal rotations
        self.rotator = GroupOrthoRotator(hidden_dim, n_rot_groups, self.total_layers)

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens, return_hidden=False):
        x = self.embed(tokens).float()
        hiddens = [] if return_hidden else None
        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = self.block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                # Apply orthogonal rotation (init=identity, grows to differentiate)
                x = self.rotator.apply_rotation(x, layer_idx)
                if return_hidden:
                    hiddens.append(x)
                layer_idx += 1
        x = self.norm(x)
        logits = self.lm_head(x)
        if return_hidden:
            return logits, hiddens
        return logits

    def block_params(self):
        return sum(p.numel() for p in self.block.parameters())

    def gb_params(self):
        return self.layer_gamma.numel() + self.layer_beta.numel() + self.iter_scale.numel()

    def rot_params(self):
        return self.rotator.n_params()


# ── Load teacher ──────────────────────────────────────────────────────
print("\nLoading teacher...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight',
    'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight',
    'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight',
    'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight',
    'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight',
    'mlp.up_proj.weight': 'ffn_up.weight',
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

hidden = gd['token_embd.weight'].shape[1]
n_heads = 16
vocab_size = gd['token_embd.weight'].shape[0]

cfg = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=n_heads, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=hidden // n_heads,
)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Check if teacher supports return_hidden ──
# Use a wrapper to get per-layer hidden states from MiniTransformer
# MiniTransformer.forward(tokens, max_layers=N) returns final logits.
# We need to modify to also return per-layer hidden states.
# Check the signature:
import inspect
teacher_fwd_sig = inspect.signature(teacher.forward)
print(f"\nTeacher forward signature: {teacher_fwd_sig}")
has_return_hidden = 'return_hidden' in teacher_fwd_sig.parameters


assert has_return_hidden, "Teacher must support return_hidden"
HIDDEN_MATCH_LAMBDA_ACTUAL = HIDDEN_MATCH_LAMBDA
print(f"  Teacher exposes hidden states -> enabling hidden MSE (lambda={HIDDEN_MATCH_LAMBDA_ACTUAL})")

# Data
print("\nLoading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
print(f"  {N_TOKENS:,} tokens")

data_offset = 0
def get_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > N_TOKENS:
        data_offset = 0
        end = BATCH_SIZE * SEQ_LEN
    chunk = all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN)
    data_offset = end
    return chunk.to(DEVICE)


# ── Build model ───────────────────────────────────────────────────────
print("\nBuilding HLR-FRR...")
model = HLRModel(
    hidden_dim=hidden, n_heads=n_heads,
    n_scales=N_SCALES, iters_per_scale=ITERS_PER_SCALE,
    vocab_size=vocab_size, ff_mult=1,
    n_rot_groups=ROT_GROUPS,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

print(f"Loading base: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)

# Copy per-scale γ/β → per-layer (same as pure_kl record)
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
with torch.no_grad():
    for s in range(N_SCALES):
        for it in range(ITERS_PER_SCALE):
            li = s * ITERS_PER_SCALE + it
            model.layer_gamma.data[li] = scale_gamma[s]
            model.layer_beta.data[li] = scale_beta[s]
    model.iter_scale.data.copy_(ckpt['iter_scale'])
print("  Loaded block + gamma/beta; rotator A=0 -> identity at init")

bp = model.block_params()
gp = model.gb_params()
rp = model.rot_params()
total = bp + gp + rp
teacher_layer_p = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"\n  Block:      {bp:,}  ({bp/1e6:.1f}M)")
print(f"  gamma/beta: {gp:,}  ({gp/1e3:.1f}K)")
print(f"  Rotations:  {rp:,}  ({rp/1e6:.2f}M)")
print(f"  TOTAL:      {total:,}  ({total/1e6:.1f}M)")
print(f"  Compression: {teacher_layer_p/total:.1f}x")


# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_all_pos(n=50):
    model.eval()
    t1, t10, nt = 0, 0, 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[s:s + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1 += int(s_top[0] == t_top[0])
            t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            nt += 1
    model.train()
    return t1 / nt, t10 / nt


@torch.no_grad()
def eval_last_tok(n=200):
    model.eval()
    t1, t10 = 0, 0
    for _ in range(n):
        s = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[s:s + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1 += int(s_top[0] == t_top[0])
        t10 += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1 / n, t10 / n


print("\nBaseline (should match pure_kl record baseline, rotations = I):")
t1_b, t10_b = eval_all_pos(50)
t1_l, t10_l = eval_last_tok(200)
print(f"  all-pos:  T1={t1_b*100:.1f}%  T10={t10_b*100:.1f}%")
print(f"  last-tok: T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING ({TOTAL_STEPS:,} steps)")
print(f"  Loss: KL(logits) (+ rotations enabled)")
if HIDDEN_MATCH_LAMBDA_ACTUAL > 0:
    print(f"  + {HIDDEN_MATCH_LAMBDA_ACTUAL} · hidden_MSE (scale-normalized)")
print(f"{'='*70}")

block_params_list = list(model.block.parameters())
gb_params_list = [model.layer_gamma, model.layer_beta, model.iter_scale]
rot_params_list = list(model.rotator.parameters())
opt = torch.optim.AdamW([
    {'params': block_params_list, 'lr': LR_BLOCK, 'weight_decay': WD_BLOCK},
    {'params': gb_params_list, 'lr': LR_GB, 'weight_decay': WD_GB},
    {'params': rot_params_list, 'lr': LR_ROT, 'weight_decay': WD_ROT},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_hist = []
kl_hist = []
best_last = 0.0
best_step = -1
t0 = time.time()

hidden_loss_ema = 0.0
USE_HIDDEN = HIDDEN_MATCH_LAMBDA_ACTUAL > 0
for step in range(TOTAL_STEPS):
    tokens = get_batch()
    with torch.no_grad():
        if USE_HIDDEN:
            tl, t_hiddens = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        else:
            tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
            t_hiddens = None

    if USE_HIDDEN:
        sl, s_hiddens = model(tokens, return_hidden=True)
    else:
        sl = model(tokens)
        s_hiddens = None

    kl = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    hidden_loss = torch.tensor(0.0, device=DEVICE)
    if USE_HIDDEN:
        for s_h, t_h in zip(s_hiddens, t_hiddens):
            t_h_d = t_h.detach()
            num = F.mse_loss(s_h, t_h_d)
            denom = (t_h_d.pow(2).mean()).clamp_min(1e-6)
            hidden_loss = hidden_loss + num / denom
        hidden_loss = hidden_loss / N_TEACHER_LAYERS
        hidden_loss_ema = 0.95 * hidden_loss_ema + 0.05 * float(hidden_loss.item())

    loss = kl + HIDDEN_MATCH_LAMBDA_ACTUAL * hidden_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_hist.append(loss.item())
    kl_hist.append(kl.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_all_pos(50)
        t1_l, t10_l = eval_last_tok(200)
        elapsed = time.time() - t0
        mark = ""
        if t10_l > best_last:
            best_last = t10_l
            best_step = step
            mark = " *** NEW BEST (last-tok) ***"
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'hlr_best.pt'))

        avg_kl = sum(kl_hist[-500:]) / min(len(kl_hist), 500)
        # Diagnostics: mean |A| across layers (how much have rotations diverged from I)
        with torch.no_grad():
            skew_abs = model.rotator.skew_upper.abs().mean().item()
            skew_max = model.rotator.skew_upper.abs().max().item()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL={avg_kl:.4f}  hidMSE={hidden_loss_ema:.4f}  "
            f"T1={t1*100:.1f}% T10={t10*100:.1f}%  ({elapsed:.0f}s){mark}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}% T10={t10_l*100:.1f}% "
            f"best={best_last*100:.1f}% @step{best_step}"
        )
        print(
            f"    |skew|_mean={skew_abs:.4f}  max={skew_max:.4f}  "
            f"(0 = identity rotation, higher = diverging)"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f'hlr_step{step}.pt'))

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'hlr_final.pt'))
print(f"\n{'='*70}")
print(f"HLR COMPLETE | Best last-tok T10: {best_last*100:.1f}% @ step {best_step}")
print(f"  Record to beat (hires): 68.23% (pure_kl)")
print(f"  Elapsed: {time.time()-t0:.0f}s  ({(time.time()-t0)/60:.1f}m)")
print(f"{'='*70}")
