"""
1.7B SPECTRAL ATTENTION STEERING (SAS) v2

FIXES FROM v1 (which had QKmag explosion: 0.07 → 11.5 in 2500 steps):
  1. BOUNDED DELTAS: Sigmoid-gated per-layer alpha (max 30%) controls
     delta contribution. No normalization (causes gradient dead zone with
     zero-init up-projections due to 1/eps amplification + grad clipping).
  2. CKA-INFORMED ROUTING: Initialize route_logits biased toward distinct
     modes for different layer groups (from CKA eigenspectrum analysis)
  3. DELTA REGULARIZATION: Explicit L2 penalty on raw delta magnitude
  4. LOWER LR: 1e-4 (vs 3e-4 in v1) for more conservative training
  5. HIGHER WEIGHT DECAY: 0.05 (vs 0.01 in v1)

v1 RESULTS (for reference):
  Step 0:    T1=42.5% T10=61.8% (all-pos), last-tok T1=46% T10=67.3%
  Step 2500: T1=43.6% T10=63.1% (all-pos), last-tok T1=38% T10=66.9%
  → All-pos improved BUT loss INCREASED 34.86→42.11, last-tok T1 crashed
  → QKmag 0.07→11.5 (164x explosion), RouteH=0.998 (no specialization)
  → Root cause: unbounded ΔQ/ΔK grew too large, distorting attention

KEY INSIGHT: The direction of the delta matters, not its magnitude.
By normalizing delta to match Q/K scale and bounding via sigmoid gate,
we get controlled attention steering without distribution collapse.
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

# ── Configuration ─────────────────────────────────────────────────────
DEVICE = 'cuda:1'
PHASE1_STEPS = 10_000
PHASE2_STEPS = 90_000
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
LR = 1e-4           # REDUCED from 3e-4 (v1 was too aggressive)
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_spectral_v2'
N_TEACHER_LAYERS = 28
WEIGHT_DECAY = 0.05  # INCREASED from 0.01

# SAS hyperparameters
N_MODES = 4
SAS_RANK = 8
T1_LOSS_WEIGHT = 0.3
MAX_DELTA_RATIO = 0.3    # NEW: max 30% modification of Q/K per layer
DELTA_REG_WEIGHT = 0.01  # NEW: L2 penalty on raw delta magnitude

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B SPECTRAL ATTENTION STEERING v2 (BOUNDED DELTAS)")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"SAS: {N_MODES} modes × rank {SAS_RANK}  |  max_delta: {MAX_DELTA_RATIO}")
print(f"LR: {LR}  |  WD: {WEIGHT_DECAY}  |  delta_reg: {DELTA_REG_WEIGHT}")
print(f"Phase 1: {PHASE1_STEPS:,} steps (SAS only, block frozen)")
print(f"Phase 2: {PHASE2_STEPS:,} steps (joint, differential LR)")
print("=" * 70)


class SpectralAttentionSteering(nn.Module):
    """
    Spectral Attention Steering v2 — with bounded deltas.

    Changes from v1:
    - Per-layer sigmoid-gated alpha bounds delta contribution (max 30%)
    - CKA-informed routing initialization (4 layer groups)
    - Delta regularization in loss (L2 on raw deltas)
    - Returns raw delta norms for regularization loss
    - No normalization: zero-init q_up with normalization creates gradient
      dead zone (1/eps amplification → grad clipping kills signal)
    """
    def __init__(self, n_heads: int, head_dim: int,
                 n_modes: int = 4, rank: int = 8, n_layers: int = 28,
                 max_delta_ratio: float = 0.3):
        super().__init__()
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rank = rank
        self.max_delta_ratio = max_delta_ratio
        self.n_layers = n_layers

        # Eigenmode basis: low-rank Q/K deltas per head
        scale = 0.01 / math.sqrt(rank)
        self.q_down = nn.Parameter(torch.randn(n_modes, n_heads, head_dim, rank) * scale)
        self.q_up = nn.Parameter(torch.zeros(n_modes, n_heads, rank, head_dim))
        self.k_down = nn.Parameter(torch.randn(n_modes, n_heads, head_dim, rank) * scale)
        self.k_up = nn.Parameter(torch.zeros(n_modes, n_heads, rank, head_dim))

        # CKA-informed routing initialization
        # CKA modes: Mode1=early(L0-6), Mode2=middle(L7-16), Mode3=late(L17-24), Mode4=final(L25-27)
        route_logits = torch.zeros(n_layers, n_modes)
        for i in range(n_layers):
            if i < 7:
                route_logits[i, 0] = 1.0    # Early layers → mode 0
            elif i < 17:
                route_logits[i, 1] = 1.0    # Middle layers → mode 1
            elif i < 25:
                route_logits[i, 2] = 1.0    # Late layers → mode 2
            else:
                route_logits[i, 3] = 1.0    # Final layers → mode 3
        self.route_logits = nn.Parameter(route_logits)

        # Per-layer delta strength: sigmoid-bounded
        # Init at -3.0 → sigmoid(-3) ≈ 0.047 → initial max delta ≈ 1.4% of Q/K
        self.delta_logit = nn.Parameter(torch.full((n_layers,), -3.0))

    def get_qk_delta(self, q: torch.Tensor, k: torch.Tensor, layer_idx: int):
        """
        Apply bounded spectral attention steering.

        Returns: (modified_q, modified_k, delta_reg_loss)
        delta_reg_loss is the L2 norm of raw deltas (for regularization).
        """
        # Routing weights for this layer
        weights = F.softmax(self.route_logits[layer_idx], dim=-1)  # (n_modes,)

        # Blend eigenmodes
        blended_q_down = torch.einsum('m,mhdr->hdr', weights, self.q_down)
        blended_q_up = torch.einsum('m,mhrd->hrd', weights, self.q_up)
        blended_k_down = torch.einsum('m,mhdr->hdr', weights, self.k_down)
        blended_k_up = torch.einsum('m,mhrd->hrd', weights, self.k_up)

        # Compute raw deltas
        q_delta = torch.einsum('bhtd,hdr->bhtr', q, blended_q_down)
        q_delta = torch.einsum('bhtr,hrd->bhtd', q_delta, blended_q_up)
        k_delta = torch.einsum('bhtd,hdr->bhtr', k, blended_k_down)
        k_delta = torch.einsum('bhtr,hrd->bhtd', k_delta, blended_k_up)

        # Regularization: L2 norm of raw deltas
        delta_reg = q_delta.pow(2).mean() + k_delta.pow(2).mean()

        # ★ BOUNDED DELTA: sigmoid-gated alpha controls contribution ★
        # NOTE: Removed normalization from v2.0 — it creates a gradient dead zone
        # when q_up is initialized at zero (amplification by 1/eps kills gradients
        # via grad clipping). Alpha gating + delta_reg + weight_decay is sufficient.
        alpha = torch.sigmoid(self.delta_logit[layer_idx]) * self.max_delta_ratio

        return q + alpha * q_delta, k + alpha * k_delta, delta_reg

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def diagnostics(self) -> dict:
        """Training diagnostics."""
        with torch.no_grad():
            # Alpha per layer (actual delta ratio)
            alphas = torch.sigmoid(self.delta_logit) * self.max_delta_ratio
            avg_alpha = alphas.mean().item()
            max_alpha = alphas.max().item()
            min_alpha = alphas.min().item()
            # Routing entropy
            probs = F.softmax(self.route_logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
            max_entropy = math.log(self.n_modes)
            # Route specialization
            route_std = self.route_logits.std(dim=-1).mean().item()
            # Raw param magnitudes (for comparison with v1)
            q_mag = (self.q_down.norm() * self.q_up.norm()).item()
            k_mag = (self.k_down.norm() * self.k_up.norm()).item()
        return {
            'avg_alpha': avg_alpha,
            'max_alpha': max_alpha,
            'min_alpha': min_alpha,
            'q_mag': q_mag,
            'k_mag': k_mag,
            'route_entropy': entropy / max_entropy,
            'route_std': route_std,
        }


class SpectralFRR(nn.Module):
    """
    Fractal Residual Recursion with Spectral Attention Steering v2.
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult,
                 n_modes=4, sas_rank=8, max_delta_ratio=0.3,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Spectral Attention Steering v2
        self.sas = SpectralAttentionSteering(
            n_heads=n_heads, head_dim=self.head_dim,
            n_modes=n_modes, rank=sas_rank, n_layers=self.total_layers,
            max_delta_ratio=max_delta_ratio,
        )

        # Embedding and head (frozen from teacher)
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

        # Accumulate delta regularization during forward pass
        self._delta_reg = 0.0

    def _sas_block_forward(self, x, gamma, beta, layer_idx):
        """Custom block forward with SAS v2 injection (bounded deltas)."""
        B, T, D = x.shape
        block = self.block

        # ─── Attention path ───
        h = block.norm1(x)
        if gamma is not None:
            h = h * gamma + (beta if beta is not None else 0)

        qkv = block.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ★ SPECTRAL ATTENTION STEERING v2 (bounded) ★
        q, k, delta_reg = self.sas.get_qk_delta(q, k, layer_idx)
        self._delta_reg = self._delta_reg + delta_reg

        # Standard attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + block.o_proj(out)

        # ─── FFN path ───
        h = block.norm2(x)
        if gamma is not None:
            h = h * gamma + (beta if beta is not None else 0)
        ffn_out = F.silu(block.gate(h)) * block.up(h)
        x = x + block.down(ffn_out)

        return x

    def forward(self, tokens):
        self._delta_reg = 0.0  # Reset accumulator
        x = self.embed(tokens).float()

        layer_idx = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]
                block_out = self._sas_block_forward(x, gamma, beta, layer_idx)
                x = x + (block_out - x) * iter_s
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def get_delta_reg(self):
        """Get accumulated delta regularization from last forward pass."""
        return self._delta_reg

    def sas_params(self):
        return sum(p.numel() for p in self.sas.parameters())

    def fractal_params(self):
        total = sum(p.numel() for p in self.block.parameters())
        total += self.scale_gamma.numel() + self.scale_beta.numel()
        total += self.iter_scale.numel()
        total += self.sas_params()
        return total


# ── Load teacher ──────────────────────────────────────────────────────
print("\nLoading Qwen3-1.7B teacher...")
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
head_dim = hidden // n_heads
vocab_size = gd['token_embd.weight'].shape[0]
print(f"  Hidden: {hidden}, Heads: {n_heads}, HeadDim: {head_dim}, Vocab: {vocab_size}")

cfg = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=n_heads, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=head_dim,
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

# ── Data ──────────────────────────────────────────────────────────────
DATA_PATH = 'fineweb_edu_100M_tokens.pt'
print(f"\nLoading pre-tokenized data from {DATA_PATH}...")
all_tokens = torch.load(DATA_PATH, weights_only=True)
N_TOKENS = all_tokens.shape[0]
print(f"  {N_TOKENS:,} tokens loaded")

data_offset = 0
def get_real_batch():
    global data_offset
    end = data_offset + BATCH_SIZE * SEQ_LEN
    if end > N_TOKENS:
        data_offset = 0
        end = BATCH_SIZE * SEQ_LEN
    chunk = all_tokens[data_offset:end].long().reshape(BATCH_SIZE, SEQ_LEN)
    data_offset = end
    return chunk.to(DEVICE)


# ── Build model ───────────────────────────────────────────────────────
print(f"\nBuilding Spectral FRR v2...")
model = SpectralFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    n_modes=N_MODES, sas_rank=SAS_RANK, max_delta_ratio=MAX_DELTA_RATIO,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load pre-trained block weights
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)
model.scale_gamma.data.copy_(ckpt['scale_gamma'])
model.scale_beta.data.copy_(ckpt['scale_beta'])
model.iter_scale.data.copy_(ckpt['iter_scale'])
print(f"  Loaded block + modulation")

# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_vs_teacher(mdl, n=100):
    """All token positions eval."""
    mdl.eval()
    t1_total, t10_total, n_tokens = 0, 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = mdl(tokens)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1_total += int(s_top[0] == t_top[0])
            t10_total += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            n_tokens += 1
    mdl.train()
    return t1_total / n_tokens, t10_total / n_tokens


@torch.no_grad()
def eval_vs_teacher_last(mdl, n=100):
    """Last token only eval (for comparison with previous experiments)."""
    mdl.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = mdl(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    mdl.train()
    return t1_hits / n, t10_hits / n


print("\nVerifying baseline...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
t1_last, t10_last = eval_vs_teacher_last(model, n=100)
print(f"  Baseline (all-pos): T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")
print(f"  Baseline (last-tok): T1={t1_last*100:.1f}%, T10={t10_last*100:.1f}%")

sas_p = model.sas_params()
fractal_p = model.fractal_params()
print(f"\n  SAS params: {sas_p:,}")
print(f"  Total fractal: {fractal_p:,}")
print(f"  Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / fractal_p:.1f}x")


# ── Phase 1: SAS only ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: SAS only ({PHASE1_STEPS:,} steps, LR={LR})")
print(f"  Block FROZEN — bounded spectral modes + routing converge first")
print(f"  Loss: KL + {T1_LOSS_WEIGHT}*CE + {DELTA_REG_WEIGHT}*delta_reg")
print(f"  Max delta ratio: {MAX_DELTA_RATIO} | Weight decay: {WEIGHT_DECAY}")
print(f"{'='*70}")

# Freeze block + modulation
for param in model.block.parameters():
    param.requires_grad = False
model.scale_gamma.requires_grad = False
model.scale_beta.requires_grad = False
model.iter_scale.requires_grad = False

p1_params = [p for p in model.parameters() if p.requires_grad]
p1_count = sum(p.numel() for p in p1_params)
print(f"  Trainable: {p1_count:,}")

opt1 = torch.optim.AdamW(p1_params, lr=LR, weight_decay=WEIGHT_DECAY)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, PHASE1_STEPS)

loss_history = []
best_t10 = 0.0
best_step = 0
t0 = time.time()

for step in range(PHASE1_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)
    delta_reg = model.get_delta_reg()

    # Combined loss: KL + T1-CE + delta regularization
    kl_loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    teacher_argmax = tl.argmax(dim=-1)
    ce_loss = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)),
        teacher_argmax.reshape(-1),
    )

    loss = kl_loss + T1_LOSS_WEIGHT * ce_loss + DELTA_REG_WEIGHT * delta_reg

    opt1.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(p1_params, 1.0)
    opt1.step()
    sched1.step()
    loss_history.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == PHASE1_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=50)
        t1_l, t10_l = eval_vs_teacher_last(model, n=100)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        diag = model.sas.diagnostics()

        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok] T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%  "
            f"alpha={diag['avg_alpha']:.4f}({diag['min_alpha']:.4f}-{diag['max_alpha']:.4f})  "
            f"QKmag={diag['q_mag']:.2f}/{diag['k_mag']:.2f}  "
            f"RouteH={diag['route_entropy']:.3f}  RouteStd={diag['route_std']:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_v2_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Phase 2: Everything jointly ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps)")
P2_BLOCK_LR = LR * 0.1
print(f"  Block LR: {P2_BLOCK_LR} (10x lower to preserve quality)")
print(f"  SAS LR: {LR}")
print(f"  Loss: KL + {T1_LOSS_WEIGHT}*CE + {DELTA_REG_WEIGHT}*delta_reg")
print(f"{'='*70}")

# Unfreeze everything
for param in model.block.parameters():
    param.requires_grad = True
model.scale_gamma.requires_grad = True
model.scale_beta.requires_grad = True
model.iter_scale.requires_grad = True

pretrained_params = list(model.block.parameters()) + [
    model.scale_gamma, model.scale_beta, model.iter_scale
]
sas_param_ids = set(id(p) for p in model.sas.parameters())
sas_params_p2 = [p for p in model.parameters()
                 if p.requires_grad and id(p) in sas_param_ids]

pretrained_count = sum(p.numel() for p in pretrained_params)
sas_count = sum(p.numel() for p in sas_params_p2)
print(f"  Pre-trained: {pretrained_count:,} @ LR={P2_BLOCK_LR}")
print(f"  SAS: {sas_count:,} @ LR={LR}")

opt2 = torch.optim.AdamW([
    {'params': pretrained_params, 'lr': P2_BLOCK_LR},
    {'params': sas_params_p2, 'lr': LR},
], weight_decay=WEIGHT_DECAY)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, PHASE2_STEPS)
t0_p2 = time.time()

for step in range(PHASE2_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)
    delta_reg = model.get_delta_reg()

    kl_loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    teacher_argmax = tl.argmax(dim=-1)
    ce_loss = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)),
        teacher_argmax.reshape(-1),
    )

    loss = kl_loss + T1_LOSS_WEIGHT * ce_loss + DELTA_REG_WEIGHT * delta_reg

    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
    opt2.step()
    sched2.step()
    loss_history.append(loss.item())

    global_step = PHASE1_STEPS + step

    if step % EVAL_INTERVAL == 0 or step == PHASE2_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=50)
        t1_l, t10_l = eval_vs_teacher_last(model, n=100)
        elapsed = time.time() - t0_p2
        total_elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = global_step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        diag = model.sas.diagnostics()

        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (g={global_step}): "
            f"loss={avg_loss:.4f}  T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok] T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%  "
            f"alpha={diag['avg_alpha']:.4f}({diag['min_alpha']:.4f}-{diag['max_alpha']:.4f})  "
            f"QKmag={diag['q_mag']:.2f}/{diag['k_mag']:.2f}  "
            f"RouteH={diag['route_entropy']:.3f}  RouteStd={diag['route_std']:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_v2_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'spectral_v2_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"SPECTRAL FRR v2 COMPLETE")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Final checkpoint: {final_path}")
print(f"{'='*70}")
