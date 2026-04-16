"""
1.7B SPECTRAL ATTENTION STEERING (SAS) v1

NOVEL: Per-layer attention pattern modification via spectral decomposition.

INSIGHT: CKA eigenanalysis revealed 4 functional modes capture 99.7% of
inter-layer variation in the teacher. The single shared QKV is the
fundamental bottleneck — post-block mechanisms (frame rotation, epigenetic
markers) can't differentiate attention patterns across virtual layers.

SAS ARCHITECTURE:
  - 4 learned "attention eigenmodes" — low-rank Q/K modifications
  - Per-layer routing blends eigenmodes to produce layer-specific attention
  - Applied INSIDE the attention computation (between QKV and softmax)
  - Initialized near-zero (LoRA-style) to preserve pre-trained quality

WHY SAS MIGHT WORK WHERE OTHERS FAILED:
  - DFC: Gated conditioning distorted block behavior (T10 DECLINED)
  - Multi-block: Multiple blocks catastrophically interfered in Phase 2
  - Resonance v1/v2: Post-block modifications spread distribution (T1 drops)
  - SAS: Directly modifies WHAT the attention heads attend to, per-layer
    This is the minimal intervention needed: one QKV → 28 attention patterns

IMPROVEMENTS OVER RESONANCE:
  1. Targets root cause (attention, not post-hoc)
  2. Spectral basis from CKA (4 modes, not 28 independent params)
  3. T1-aware loss (CE on argmax + KL on distribution)
  4. Better eval (all token positions, not just last)
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
LR = 3e-4
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_spectral_v1'
N_TEACHER_LAYERS = 28

# SAS hyperparameters
N_MODES = 4       # Number of attention eigenmodes (from CKA: 4 modes = 99.7%)
SAS_RANK = 8      # Rank per eigenmode
T1_LOSS_WEIGHT = 0.3  # Weight of T1-targeted CE loss

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B SPECTRAL ATTENTION STEERING v1")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"SAS: {N_MODES} modes × rank {SAS_RANK}  |  T1 loss weight: {T1_LOSS_WEIGHT}")
print(f"Phase 1: {PHASE1_STEPS:,} steps (SAS only, block frozen)")
print(f"Phase 2: {PHASE2_STEPS:,} steps (joint, differential LR)")
print("=" * 70)


class SpectralAttentionSteering(nn.Module):
    """
    Novel: Spectral decomposition of attention variation across virtual layers.

    CKA analysis revealed 4 functional eigenmodes capture 99.7% of inter-layer
    variation. Instead of 28 separate Q/K modifications (expensive, overfitting),
    we learn 4 attention eigenmodes and blend per-layer via learned routing.

    Each eigenmode: low-rank Q/K modification applied per-head.
    Per-layer routing: softmax-normalized 4D vector selects mode blend.
    Initialization: up-projections zeroed (LoRA-style) → identity at init.

    Param budget: ~262K (< 1% overhead on 29M fractal params)
    """
    def __init__(self, n_heads: int, head_dim: int,
                 n_modes: int = 4, rank: int = 8, n_layers: int = 28):
        super().__init__()
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rank = rank

        # Eigenmode basis: low-rank Q/K deltas per head
        # Down-proj: (n_modes, n_heads, head_dim, rank) — random init
        # Up-proj: (n_modes, n_heads, rank, head_dim) — zero init (LoRA)
        scale = 0.01 / math.sqrt(rank)
        self.q_down = nn.Parameter(torch.randn(n_modes, n_heads, head_dim, rank) * scale)
        self.q_up = nn.Parameter(torch.zeros(n_modes, n_heads, rank, head_dim))
        self.k_down = nn.Parameter(torch.randn(n_modes, n_heads, head_dim, rank) * scale)
        self.k_up = nn.Parameter(torch.zeros(n_modes, n_heads, rank, head_dim))

        # Per-layer routing: weights over eigenmodes
        # Init uniform → equal blend of all modes
        self.route_logits = nn.Parameter(torch.zeros(n_layers, n_modes))

    def get_qk_delta(self, q: torch.Tensor, k: torch.Tensor, layer_idx: int):
        """
        Apply spectral attention steering to Q and K tensors.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_heads, T, head_dim)
            layer_idx: which virtual layer (0..27)

        Returns:
            Modified q, k with per-layer attention pattern adjustment
        """
        # Routing weights for this layer
        weights = F.softmax(self.route_logits[layer_idx], dim=-1)  # (n_modes,)

        # Blend eigenmodes: weighted sum over modes
        # q_down: (n_modes, n_heads, head_dim, rank)
        blended_q_down = torch.einsum('m,mhdr->hdr', weights, self.q_down)
        blended_q_up = torch.einsum('m,mhrd->hrd', weights, self.q_up)
        blended_k_down = torch.einsum('m,mhdr->hdr', weights, self.k_down)
        blended_k_up = torch.einsum('m,mhrd->hrd', weights, self.k_up)

        # Apply low-rank delta: q' = q + q @ down @ up
        q_delta = torch.einsum('bhtd,hdr->bhtr', q, blended_q_down)
        q_delta = torch.einsum('bhtr,hrd->bhtd', q_delta, blended_q_up)

        k_delta = torch.einsum('bhtd,hdr->bhtr', k, blended_k_down)
        k_delta = torch.einsum('bhtr,hrd->bhtd', k_delta, blended_k_up)

        return q + q_delta, k + k_delta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def diagnostics(self) -> dict:
        """Training diagnostics."""
        with torch.no_grad():
            # Average magnitude of Q/K deltas (should grow from ~0)
            q_mag = (self.q_down.norm() * self.q_up.norm()).item()
            k_mag = (self.k_down.norm() * self.k_up.norm()).item()
            # Routing entropy (high = uniform, low = specialized)
            probs = F.softmax(self.route_logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
            max_entropy = math.log(self.n_modes)
            # Route specialization per layer
            route_std = self.route_logits.std(dim=-1).mean().item()
        return {
            'q_mag': q_mag,
            'k_mag': k_mag,
            'route_entropy': entropy / max_entropy,  # 1.0 = uniform, 0.0 = single mode
            'route_std': route_std,
        }


class SpectralFRR(nn.Module):
    """
    Fractal Residual Recursion with Spectral Attention Steering.

    Same as standard FRR but with per-layer attention pattern modification.
    Custom forward re-implements attention to inject SAS between QKV and softmax.
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult,
                 n_modes=4, sas_rank=8,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Shared block (core — reused at all depths)
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # NOVEL: Spectral Attention Steering
        self.sas = SpectralAttentionSteering(
            n_heads=n_heads, head_dim=self.head_dim,
            n_modes=n_modes, rank=sas_rank, n_layers=self.total_layers,
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

    def _sas_block_forward(self, x, gamma, beta, layer_idx):
        """
        Custom block forward with SAS injection.

        Reimplements FractalBlock.forward but adds SAS between QKV and attention.
        Uses the block's existing sub-modules (qkv, o_proj, norms, FFN).
        """
        B, T, D = x.shape
        block = self.block

        # ─── Attention path ───
        h = block.norm1(x)
        if gamma is not None:
            h = h * gamma + (beta if beta is not None else 0)

        # QKV projection (shared)
        qkv = block.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ★ SPECTRAL ATTENTION STEERING ★
        # Per-layer Q/K modification via eigenmode blending
        q, k = self.sas.get_qk_delta(q, k, layer_idx)

        # Attention computation (standard)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + block.o_proj(out)

        # ─── FFN path (standard) ───
        h = block.norm2(x)
        if gamma is not None:
            h = h * gamma + (beta if beta is not None else 0)
        ffn_out = F.silu(block.gate(h)) * block.up(h)
        x = x + block.down(ffn_out)

        return x

    def forward(self, tokens):
        x = self.embed(tokens).float()

        layer_idx = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]
                # Custom forward with SAS injection
                block_out = self._sas_block_forward(x, gamma, beta, layer_idx)
                x = x + (block_out - x) * iter_s
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def sas_params(self):
        """Just the SAS parameters."""
        return sum(p.numel() for p in self.sas.parameters())

    def fractal_params(self):
        """All fractal-specific params (block + modulation + SAS)."""
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
print(f"\nBuilding Spectral FRR v1...")
model = SpectralFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    n_modes=N_MODES, sas_rank=SAS_RANK,
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

# ── Eval (IMPROVED: all token positions) ──────────────────────────────
@torch.no_grad()
def eval_vs_teacher(mdl, n=100):
    """
    Improved eval: compares ALL token positions, not just last.
    With 100 samples × 64 tokens = 6400 comparisons → ~0.6% standard error.
    """
    mdl.eval()
    t1_total, t10_total, n_tokens = 0, 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)

        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = mdl(tokens)

        # Compare all positions (not just last)
        # tl, sl shape: (1, SEQ_LEN, vocab_size)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1_total += int(s_top[0] == t_top[0])
            t10_total += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            n_tokens += 1

    mdl.train()
    return t1_total / n_tokens, t10_total / n_tokens


# Also keep the original eval for comparison
@torch.no_grad()
def eval_vs_teacher_last(mdl, n=100):
    """Original eval: last token only (for comparison with previous experiments)."""
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
print(f"  Block FROZEN — spectral modes + routing converge first")
print(f"  Loss: KL + {T1_LOSS_WEIGHT} × CE(teacher argmax)")
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

opt1 = torch.optim.AdamW(p1_params, lr=LR, weight_decay=0.01)
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

    # Combined loss: KL divergence + T1-targeted cross-entropy
    kl_loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    # T1 loss: cross-entropy on teacher's argmax token
    teacher_argmax = tl.argmax(dim=-1)  # (B, T)
    ce_loss = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)),
        teacher_argmax.reshape(-1),
    )

    loss = kl_loss + T1_LOSS_WEIGHT * ce_loss

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
            f"QKmag={diag['q_mag']:.4f}/{diag['k_mag']:.4f}  "
            f"RouteH={diag['route_entropy']:.3f}  RouteStd={diag['route_std']:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_v1_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Phase 2: Everything jointly ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps)")
P2_BLOCK_LR = LR * 0.1
print(f"  Block LR: {P2_BLOCK_LR} (10x lower to preserve quality)")
print(f"  SAS LR: {LR}")
print(f"  Loss: KL + {T1_LOSS_WEIGHT} × CE(teacher argmax)")
print(f"{'='*70}")

# Unfreeze everything
for param in model.block.parameters():
    param.requires_grad = True
model.scale_gamma.requires_grad = True
model.scale_beta.requires_grad = True
model.iter_scale.requires_grad = True

# Separate param groups
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
], weight_decay=0.01)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, PHASE2_STEPS)
t0_p2 = time.time()

for step in range(PHASE2_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)

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

    loss = kl_loss + T1_LOSS_WEIGHT * ce_loss

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
            f"QKmag={diag['q_mag']:.4f}/{diag['k_mag']:.4f}  "
            f"RouteH={diag['route_entropy']:.3f}  RouteStd={diag['route_std']:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_v1_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'spectral_v1_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"SPECTRAL FRR v1 COMPLETE")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Final checkpoint: {final_path}")
print(f"{'='*70}")
