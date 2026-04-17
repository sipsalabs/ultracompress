"""
1.7B SPECTRAL CORRECTION FRR — One base block + K tiny correction networks.

INSIGHT FROM CKA ANALYSIS:
  Function space has only 3-4 modes (vs 27D weight space).
  EV1=62% (one shared block captures this).
  EV2=85%, EV3=94%, EV4=99.7%.
  We need K=4 functional corrections to capture the remaining 38%.

ARCHITECTURE:
  Base block = pre-trained 100K block (captures dominant mode)
  K=4 correction MLPs (tiny):
    Each: Linear(hidden, rank) -> SiLU -> Linear(rank, hidden)
    Learns a residual correction for each functional mode.
  Per-layer soft mixing: 28 × K weights choose how much of each correction.

  Forward: h_base = block(x, gamma, beta)
           correction = sum_k(alpha[l,k] * corr_k(x))
           x = x + (h_base + correction - x) * iter_scale

  Key advantage over Multi-Block:
  - Only ONE heavy forward pass (vs K for multi-block)
  - Corrections are cheap (rank-32 MLPs vs full transformer blocks)
  - 30.4M params, 50x compression (vs 58.8M/26x for 2-block)

PARAMETERS:
  Base block: 29.4M (pre-trained, fine-tuned)
  4 correction MLPs: 4 × (2048×32 + 32×2048) = 524K
  Per-layer mixing: 28 × 4 = 112
  Per-scale modulation: 16K (from base model)
  Total: ~30.0M → 51x compression

TRAINING:
  Phase 1 (10K): Corrections + mix only, base block frozen
  Phase 2 (90K): Everything jointly
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
PHASE1_STEPS = 10_000   # Corrections + mix only (base frozen)
PHASE2_STEPS = 90_000   # Everything jointly
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 3e-4
TEMP = 2.0
N_SCALES = 4
ITERS_PER_SCALE = 7
TOTAL_LAYERS = N_SCALES * ITERS_PER_SCALE  # 28
K_MODES = 4          # Number of functional correction modes
CORR_RANK = 32       # Rank of each correction MLP
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_spectral'
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B SPECTRAL CORRECTION FRR")
print(f"  Base block + {K_MODES} correction MLPs (rank-{CORR_RANK})")
print(f"  Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"  Phase 1: {PHASE1_STEPS:,} steps (corrections + mix only)")
print(f"  Phase 2: {PHASE2_STEPS:,} steps (everything jointly)")
print("=" * 70)

# ── Load 1.7B Teacher ────────────────────────────────────────────────
print("Loading Qwen3-1.7B teacher...")
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
N_TEACHER_LAYERS = 28
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
print(f"  Hidden: {hidden}, Heads: {n_heads}, Vocab: {vocab_size}")

config = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=n_heads, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=head_dim,
)
teacher = MiniTransformer(config, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)

embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Data ──────────────────────────────────────────────────────────────
LOCAL_TOKENS_FILE = 'fineweb_edu_100M_tokens.pt'
print(f"Loading pre-tokenized data from {LOCAL_TOKENS_FILE}...")
ALL_TOKENS = torch.load(LOCAL_TOKENS_FILE, weights_only=True).to(torch.long)
print(f"  {ALL_TOKENS.numel():,} tokens loaded")


def get_real_batch(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    starts = torch.randint(0, ALL_TOKENS.numel() - seq_len, (batch_size,))
    batch = torch.stack([ALL_TOKENS[s:s + seq_len] for s in starts])
    return batch.to(DEVICE)


def eval_vs_teacher(model, n: int = 100):
    model.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        starts = torch.randint(0, ALL_TOKENS.numel() - 32, (1,))
        tokens = ALL_TOKENS[starts[0]:starts[0] + 32].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
            sl = model(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1_hits / n, t10_hits / n


# ── Correction MLP ────────────────────────────────────────────────────
class CorrectionMLP(nn.Module):
    """Tiny rank-bottleneck MLP that learns a functional correction.

    Each one captures a 'mode' of functional variation between layers.
    init zeros so corrections start at zero (preserves baseline quality).
    """
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)  # Start with zero correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


# ── Spectral Correction FRR Model ─────────────────────────────────────
class SpectralCorrectionFRR(nn.Module):
    """One shared block + K tiny correction MLPs + per-layer mixing.

    Novel architecture based on CKA analysis showing only 3-4 functional
    modes exist. Each correction MLP captures one mode of inter-layer
    variation. Per-layer soft mixing selects the appropriate correction.

    Forward per layer:
      h = block(x, gamma, beta)           # base computation
      corr = sum_k(alpha[l,k] * corr_k(x)) # spectral correction
      x = x + (h + corr - x) * iter_scale  # residual blend
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, embed_weight, lm_head_weight, norm_weight,
                 k_modes=4, corr_rank=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.k_modes = k_modes

        # THE shared block (pre-trained from 100K checkpoint)
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # K correction MLPs (tiny, zero-initialized)
        self.corrections = nn.ModuleList([
            CorrectionMLP(hidden_dim, corr_rank) for _ in range(k_modes)
        ])

        # Per-layer mixing: 28 × K logits → softmax → mixing weights
        # Init to equal mixing across modes (zeros → softmax = 1/K each)
        self.mix_logits = nn.Parameter(torch.zeros(self.total_layers, k_modes))

        # Per-scale modulation (from base model)
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # Frozen teacher embeddings
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        layer_count = 0

        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]

                # Base block computation (heavy)
                h = self.block(x, gamma, beta)

                # Spectral corrections (lightweight)
                # mix weights via softmax → proper probability over modes
                alpha = F.softmax(self.mix_logits[layer_count], dim=0)
                corr = torch.zeros_like(x)
                for k in range(self.k_modes):
                    if alpha[k] > 0.01:  # Skip negligible corrections
                        corr = corr + alpha[k] * self.corrections[k](x)

                # Apply with iter_scale
                x = x + (h + corr - x) * iter_s
                layer_count += 1

        x = self.norm(x)
        return self.lm_head(x)

    def fractal_params(self):
        total = sum(p.numel() for p in self.block.parameters())
        total += sum(p.numel() for p in self.corrections.parameters())
        total += self.scale_gamma.numel() + self.scale_beta.numel()
        total += self.iter_scale.numel() + self.mix_logits.numel()
        return total


# ── Build Model ───────────────────────────────────────────────────────
print(f"\nBuilding Spectral Correction FRR ({K_MODES} modes, rank-{CORR_RANK})...")
model = SpectralCorrectionFRR(
    hidden, n_heads, N_SCALES, ITERS_PER_SCALE, vocab_size, 1,
    embed_w, lm_head_w, norm_w,
    k_modes=K_MODES, corr_rank=CORR_RANK,
).to(DEVICE)

# Load pre-trained block weights
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)

block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)

for k in ['scale_gamma', 'scale_beta', 'iter_scale']:
    if k in ckpt:
        getattr(model, k).data.copy_(ckpt[k])
del ckpt

# Verify baseline (corrections are zero → should match single-block FRR)
print("Verifying baseline (all corrections zero)...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

block_params = sum(p.numel() for p in model.block.parameters())
corr_params = sum(p.numel() for p in model.corrections.parameters())
total_fractal = model.fractal_params()
teacher_params = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)

print(f"\n  Base block: {block_params:,} params")
print(f"  Corrections ({K_MODES} x rank-{CORR_RANK}): {corr_params:,} params")
print(f"  Mix weights: {model.mix_logits.numel()}")
print(f"  Total fractal: {total_fractal:,}")
print(f"  Compression: {teacher_params/total_fractal:.1f}x")
print(f"  Overhead vs single-block: +{(total_fractal/block_params - 1)*100:.1f}%")

# ── Phase 1: Corrections + mix only (base frozen) ────────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: Corrections + mix ({PHASE1_STEPS:,} steps, LR={LR})")
print(f"  Base block FROZEN — stability")
print(f"  Learning: {K_MODES} correction MLPs + {TOTAL_LAYERS}x{K_MODES} mixing weights")
print(f"{'='*70}")

# Freeze base block and embeddings
for param in model.block.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if 'embed' in name or 'lm_head' in name or name == 'norm.weight':
        param.requires_grad = False

p1_params = [p for p in model.parameters() if p.requires_grad]
p1_count = sum(p.numel() for p in p1_params)
print(f"  Trainable: {p1_count:,}")

opt1 = torch.optim.AdamW(p1_params, lr=LR, weight_decay=0.01)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, PHASE1_STEPS)
t0 = time.time()
best_t10 = t10_base
best_step = -1
loss_history = []

for step in range(PHASE1_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt1.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(p1_params, 1.0)
    opt1.step()
    sched1.step()
    loss_history.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == PHASE1_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=100)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        # Show mixing profile
        with torch.no_grad():
            alpha = F.softmax(model.mix_logits, dim=1)  # (28, K)
            # Show which mode dominates at each layer
            dominant = alpha.argmax(dim=1)
            mode_str_early = ' '.join(f'{dominant[i]}' for i in range(7))
            mode_str_late = ' '.join(f'{dominant[i]}' for i in range(21, 28))
            # Show max alpha per layer for early and late
            max_alpha_early = ' '.join(f'{alpha[i].max():.2f}' for i in range(7))
            max_alpha_late = ' '.join(f'{alpha[i].max():.2f}' for i in range(21, 28))

        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(f"    Dominant mode (early): {mode_str_early}  |  (late): {mode_str_late}")
        print(f"    Max alpha   (early): {max_alpha_early}  |  (late): {max_alpha_late}")

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Phase 2: Everything jointly ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps, LR={LR})")
print(f"  Base block + corrections + mix + modulation")
print(f"{'='*70}")

for param in model.block.parameters():
    param.requires_grad = True

p2_params = [p for p in model.parameters() if p.requires_grad]
p2_count = sum(p.numel() for p in p2_params)
print(f"  Trainable: {p2_count:,}")

opt2 = torch.optim.AdamW(p2_params, lr=LR, weight_decay=0.01)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, PHASE2_STEPS)
t0_p2 = time.time()

for step in range(PHASE2_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(p2_params, 1.0)
    opt2.step()
    sched2.step()
    loss_history.append(loss.item())

    global_step = PHASE1_STEPS + step

    if step % EVAL_INTERVAL == 0 or step == PHASE2_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=100)
        elapsed = time.time() - t0_p2
        total_elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = global_step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        with torch.no_grad():
            alpha = F.softmax(model.mix_logits, dim=1)
            dominant = alpha.argmax(dim=1)
            mode_str_early = ' '.join(f'{dominant[i]}' for i in range(7))
            mode_str_late = ' '.join(f'{dominant[i]}' for i in range(21, 28))
            max_alpha_early = ' '.join(f'{alpha[i].max():.2f}' for i in range(7))
            max_alpha_late = ' '.join(f'{alpha[i].max():.2f}' for i in range(21, 28))

        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (g={global_step}): loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )
        print(f"    Dominant mode (early): {mode_str_early}  |  (late): {mode_str_late}")
        print(f"    Max alpha   (early): {max_alpha_early}  |  (late): {max_alpha_late}")

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'spectral_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'spectral_1.7b_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"  Final: T1={t1*100:.1f}%, T10={t10*100:.1f}%")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {time.time()-t0:.0f}s")
print(f"{'='*70}")

# Mode specialization analysis
print(f"\nMODE SPECIALIZATION ANALYSIS:")
with torch.no_grad():
    alpha = F.softmax(model.mix_logits, dim=1)
    for scale in range(N_SCALES):
        start = scale * ITERS_PER_SCALE
        end = start + ITERS_PER_SCALE
        avg_alpha = alpha[start:end].mean(0)
        print(f"  Scale {scale} (layers {start}-{end-1}):")
        for k in range(K_MODES):
            print(f"    Mode {k}: avg={avg_alpha[k]:.3f}")

    # Correction norms
    print(f"\n  Correction MLP output norms (on random input):")
    test_x = torch.randn(1, 32, hidden, device=DEVICE)
    for k, corr in enumerate(model.corrections):
        out_norm = corr(test_x).norm().item()
        print(f"    Mode {k}: {out_norm:.4f}")
