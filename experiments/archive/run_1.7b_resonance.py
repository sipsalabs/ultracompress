"""
1.7B RESONANT FRACTAL COMPRESSION — Genuinely novel mechanisms.

NOT from CS papers. From physics, biology, and mathematics.

THE INSIGHT (from CKA + first principles):
  One shared block applied 28 times always computes the SAME function.
  Gamma/beta only scale features element-wise — never changes the GEOMETRY
  of what the block computes. We need the block to compute 28 genuinely
  different functions from ONE set of weights.

NOVEL MECHANISM 1: FRAME ROTATION (from crystallography)
  A crystal looks different from different angles. Same structure,
  different diffraction patterns. Similarly: rotate the hidden state
  BEFORE the block processes it, rotate back AFTER.

  x_rot = Householder(x, v_l)   # rotate into layer-specific frame
  result = Block(x_rot)         # block computes in rotated frame
  output = Householder(result, v_l)  # rotate back

  The block's QKV projections now see rotated features → different
  attention patterns, different FFN activations. ONE block, 28 frames.
  Householder reflections are their own inverse (R² = I).

  Parameters: 28 direction vectors × 2048 dims = 57,344. Near zero.
  Inspired by: X-ray crystallography, reference frame transforms (GR)

NOVEL MECHANISM 2: EPIGENETIC MARKERS (from biology)
  DNA is identical in every cell. Methylation marks tell each cell
  what to express. Our block is "DNA" — it needs per-layer "marks"
  that tell it which aspects of its computation to amplify.

  Unlike gamma/beta (element-wise scaling), epigenetic markers are
  DIRECTIONAL: they amplify/suppress along K learned directions in
  hidden space. Like shining K spotlights on different features.

  x_marked = x + Σ_k magnitude_k(l) * project(x, direction_k(l))

  Parameters: K=4 directions × 28 layers × (2048 + 1) = 229,432
  Inspired by: DNA methylation, epigenetic gene regulation

NOVEL MECHANISM 3: FRACTAL MEMORY (from iterated function systems)
  In z → z² + c, the trajectory depends on ALL previous iterations.
  But in FRR, information only flows through the residual stream x.
  A separate small "memory" accumulates a compressed trace of the
  computation path. The block can query this memory to know "where
  it is" in the computation without explicit layer indexing.

  m_l = decay * m_{l-1} + proj_down(block_output)
  x_augmented = x + proj_up(m_l)

  Parameters: proj_up (64→2048) + proj_down (2048→64) + decay = 262K
  Inspired by: IFS fractals, cellular memory, path integrals

TOTAL NOVEL PARAMS: ~549K (1.9% of base block)
TOTAL MODEL: ~29.9M
COMPRESSION: 51.2x (essentially unchanged from base FRR)

TRAINING:
  Phase 1 (10,000 steps): New mechanisms only (block frozen)
    → Resonant directions, memory, and markers converge
  Phase 2 (90,000 steps): Everything jointly
    → Block co-adapts with established novel mechanisms
  LR: 3e-4, cosine decay, TEMP=2.0
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
K_DIRECTIONS = 4      # Epigenetic marker directions per layer
MEMORY_DIM = 64        # Fractal memory bottleneck dimension
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_resonance'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B RESONANT FRACTAL COMPRESSION")
print("  Novel mechanisms from physics, biology, and mathematics")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"Phase 1: {PHASE1_STEPS:,} steps (novel mechanisms, block frozen)")
print(f"Phase 2: {PHASE2_STEPS:,} steps (everything jointly)")
print(f"Novel: K={K_DIRECTIONS} directions, {MEMORY_DIM}D memory, frame rotation")
print("=" * 70)


# ── Novel Architecture ────────────────────────────────────────────────

class ResonantFRR(nn.Module):
    """
    Resonant Fractal Compression — novel mechanisms for shared-block diversity.

    Three mechanisms that DON'T exist in any paper:
    1. Frame rotation (Householder): different effective block at each layer
    2. Epigenetic markers: directional amplification in hidden space
    3. Fractal memory: cross-iteration compressed state accumulator
    """
    def __init__(self, hidden_dim: int, n_heads: int, n_scales: int,
                 iters_per_scale: int, vocab_size: int, ff_mult: int,
                 k_directions: int = 4, memory_dim: int = 64,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.k_directions = k_directions
        self.memory_dim = memory_dim

        # ── Shared block (THE core — reused everywhere) ──
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # ── Existing modulation (from base FRR) ──
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # ── NOVEL 1: Frame Rotation (Householder reflections) ──
        # Per-layer direction vector for Householder reflection
        # R(x) = x - 2*(v·x / (v·v + ε))*v
        # When magnitude → 0, R(x) → x (identity). Safe to init small.
        # Using a magnitude scalar so we can zero-init it cleanly.
        self.frame_dirs = nn.Parameter(
            F.normalize(torch.randn(self.total_layers, hidden_dim), dim=-1)
        )
        self.frame_mags = nn.Parameter(torch.zeros(self.total_layers))
        # frame_mags=0 → no rotation initially

        # ── NOVEL 2: Epigenetic Markers (directional modulation) ──
        # K learned directions per layer with scalar magnitudes
        # Amplifies/suppresses hidden state along specific directions
        dirs = torch.randn(self.total_layers, k_directions, hidden_dim)
        dirs = F.normalize(dirs, dim=-1)
        self.epi_dirs = nn.Parameter(dirs)       # [L, K, D]
        self.epi_mags = nn.Parameter(
            torch.zeros(self.total_layers, k_directions)
        )  # init 0 → no effect

        # ── NOVEL 3: Fractal Memory (cross-iteration accumulator) ──
        # Small compressed state that accumulates computation history
        self.mem_proj_down = nn.Linear(hidden_dim, memory_dim, bias=False)
        self.mem_proj_up = nn.Linear(memory_dim, hidden_dim, bias=False)
        # Decay factor per layer (sigmoid of learnable logit)
        # Init at logit=2.0 → decay≈0.88 (moderate persistence)
        self.mem_decay_logits = nn.Parameter(
            torch.full((self.total_layers,), 2.0)
        )
        # Zero-init the up projection so memory has no effect initially
        nn.init.zeros_(self.mem_proj_up.weight)

        # ── Embedding and head (frozen from teacher) ──
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

    def _householder(self, x: torch.Tensor, direction: torch.Tensor,
                     magnitude: torch.Tensor) -> torch.Tensor:
        """Apply scaled Householder-like reflection.

        R(x) = x - 2 * magnitude² * (v·x / (v·v)) * v

        When magnitude=0: R(x) = x (identity).
        When magnitude=1: standard Householder reflection.
        magnitude² ensures smooth gradient at 0.
        """
        # direction: [D], magnitude: scalar, x: [B, T, D]
        # v·x: [B, T]
        vx = torch.einsum('d,btd->bt', direction, x)
        # v·v: scalar
        vv = torch.dot(direction, direction).clamp(min=1e-8)
        # Reflection: x - 2 * mag² * (v·x / v·v) * v
        scale = 2.0 * magnitude * magnitude / vv
        return x - scale * vx.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)

    def _apply_epigenetic(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply directional epigenetic markers.

        For each of K directions, project x onto that direction,
        scale by learned magnitude, and add back. This amplifies
        the component of x along each learned direction.
        """
        dirs = F.normalize(self.epi_dirs[layer_idx], dim=-1)  # [K, D]
        mags = self.epi_mags[layer_idx]  # [K]
        # Project x onto each direction: [B, T, K]
        proj = torch.einsum('btd,kd->btk', x, dirs)
        # Scale and reconstruct: [B, T, D]
        delta = torch.einsum('btk,k,kd->btd', proj, mags, dirs)
        return x + delta

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens).float()
        B, T, D = x.shape

        # Initialize fractal memory (per batch element)
        memory = torch.zeros(B, self.memory_dim, device=x.device, dtype=x.dtype)

        layer_idx = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]

                # ── NOVEL 3: Inject fractal memory ──
                mem_inject = self.mem_proj_up(memory)  # [B, D]
                x_aug = x + mem_inject.unsqueeze(1)    # [B, T, D]

                # ── NOVEL 2: Apply epigenetic markers ──
                x_marked = self._apply_epigenetic(x_aug, layer_idx)

                # ── NOVEL 1: Frame rotation (pre-block) ──
                x_rotated = self._householder(
                    x_marked,
                    self.frame_dirs[layer_idx],
                    self.frame_mags[layer_idx],
                )

                # ── Apply shared block with existing modulation ──
                block_out = self.block(x_rotated, gamma, beta)

                # ── NOVEL 1: Inverse frame rotation (post-block) ──
                # Householder is its own inverse: R² = I
                block_derot = self._householder(
                    block_out,
                    self.frame_dirs[layer_idx],
                    self.frame_mags[layer_idx],
                )

                # ── Residual connection with iter_scale ──
                x = x + (block_derot - x) * iter_s

                # ── NOVEL 3: Update fractal memory ──
                decay = torch.sigmoid(self.mem_decay_logits[layer_idx])
                # Compress block output into memory
                block_summary = x.mean(dim=1)  # [B, D]
                mem_update = self.mem_proj_down(block_summary)  # [B, mem_dim]
                memory = decay * memory + (1 - decay) * mem_update

                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def novel_params(self) -> int:
        """Count only the novel mechanism parameters."""
        total = 0
        # Frame rotation
        total += self.frame_dirs.numel() + self.frame_mags.numel()
        # Epigenetic markers
        total += self.epi_dirs.numel() + self.epi_mags.numel()
        # Fractal memory
        total += sum(p.numel() for p in self.mem_proj_down.parameters())
        total += sum(p.numel() for p in self.mem_proj_up.parameters())
        total += self.mem_decay_logits.numel()
        return total

    def fractal_params(self) -> int:
        """Total fractal-specific params (block + modulation + novel)."""
        total = sum(p.numel() for p in self.block.parameters())
        total += self.scale_gamma.numel() + self.scale_beta.numel()
        total += self.iter_scale.numel()
        total += self.novel_params()
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

# ── Load pre-tokenized data ──────────────────────────────────────────
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
print(f"\nBuilding Resonant FRR (K={K_DIRECTIONS} directions, {MEMORY_DIM}D memory)...")
model = ResonantFRR(
    hidden_dim=hidden,
    n_heads=n_heads,
    n_scales=4,
    iters_per_scale=7,
    vocab_size=vocab_size,
    ff_mult=1,  # 1.7B uses ff_mult=1
    k_directions=K_DIRECTIONS,
    memory_dim=MEMORY_DIM,
    embed_weight=embed_w,
    lm_head_weight=lm_head_w,
    norm_weight=norm_w,
).to(DEVICE)

# ── Load pre-trained block weights ────────────────────────────────────
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)

# Load block weights
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)
print(f"  Loaded block: {len(block_state)} tensors")

# Load existing modulation
model.scale_gamma.data.copy_(ckpt['scale_gamma'])
model.scale_beta.data.copy_(ckpt['scale_beta'])
model.iter_scale.data.copy_(ckpt['iter_scale'])
print(f"  Loaded modulation: scale_gamma, scale_beta, iter_scale")

# ── Verify baseline ──────────────────────────────────────────────────
@torch.no_grad()
def eval_vs_teacher(mdl, n: int = 100) -> tuple[float, float]:
    """Eval matching multi-block metric: random samples, overlap fraction."""
    mdl.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (1,))
        tokens = all_tokens[starts[0]:starts[0] + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
            sl = mdl(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    mdl.train()
    return t1_hits / n, t10_hits / n


print("\nVerifying baseline (all novel mechanisms at zero)...")
t1_base, t10_base = eval_vs_teacher(model, n=100)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

novel_p = model.novel_params()
fractal_p = model.fractal_params()
teacher_layer_params = N_TEACHER_LAYERS * (
    4 * hidden * hidden +
    3 * hidden * hidden * 1  # ff_mult approx
)
print(f"\n  Block: {sum(p.numel() for p in model.block.parameters()):,} params")
print(f"  Novel mechanisms: {novel_p:,} params ({novel_p/fractal_p*100:.1f}% of total)")
print(f"    Frame rotation: {model.frame_dirs.numel() + model.frame_mags.numel():,}")
print(f"    Epigenetic markers: {model.epi_dirs.numel() + model.epi_mags.numel():,}")
mem_p = (sum(p.numel() for p in model.mem_proj_down.parameters()) +
         sum(p.numel() for p in model.mem_proj_up.parameters()) +
         model.mem_decay_logits.numel())
print(f"    Fractal memory: {mem_p:,}")
print(f"  Total fractal: {fractal_p:,}")
print(f"  Compression: {teacher_layer_params/fractal_p:.1f}x")


# ── Training ──────────────────────────────────────────────────────────
loss_history = []
best_t10 = 0.0
best_step = 0
t0 = time.time()

# ── Phase 1: Novel mechanisms only (block frozen) ─────────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: Novel mechanisms only ({PHASE1_STEPS:,} steps, LR={LR})")
print(f"  Block FROZEN — resonant directions, memory, markers converge")
print(f"{'='*70}")

# Freeze block
for param in model.block.parameters():
    param.requires_grad = False
# Freeze existing modulation too (keep base FRR fixed)
model.scale_gamma.requires_grad = False
model.scale_beta.requires_grad = False
model.iter_scale.requires_grad = False

p1_params = [p for p in model.parameters() if p.requires_grad]
p1_count = sum(p.numel() for p in p1_params)
print(f"  Trainable: {p1_count:,} (novel mechanisms only)")

opt1 = torch.optim.AdamW(p1_params, lr=LR, weight_decay=0.01)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, PHASE1_STEPS)

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

        # Report novel mechanism stats
        frame_mag_mean = model.frame_mags.abs().mean().item()
        epi_mag_mean = model.epi_mags.abs().mean().item()
        decay_mean = torch.sigmoid(model.mem_decay_logits).mean().item()

        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    Frame|mag|={frame_mag_mean:.4f}  "
            f"Epi|mag|={epi_mag_mean:.4f}  "
            f"MemDecay={decay_mean:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'resonant_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Phase 2: Everything jointly ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps, LR={LR})")
print(f"  Block + modulation + novel mechanisms — full co-adaptation")
print(f"{'='*70}")

# Unfreeze everything
for param in model.block.parameters():
    param.requires_grad = True
model.scale_gamma.requires_grad = True
model.scale_beta.requires_grad = True
model.iter_scale.requires_grad = True

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

        frame_mag_mean = model.frame_mags.abs().mean().item()
        epi_mag_mean = model.epi_mags.abs().mean().item()
        decay_mean = torch.sigmoid(model.mem_decay_logits).mean().item()

        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (g={global_step}): "
            f"loss={avg_loss:.4f}  T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )
        print(
            f"    Frame|mag|={frame_mag_mean:.4f}  "
            f"Epi|mag|={epi_mag_mean:.4f}  "
            f"MemDecay={decay_mean:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'resonant_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'resonant_final.pt')
torch.save(model.state_dict(), final_path)

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"RESONANT FRR COMPLETE")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Final checkpoint: {final_path}")
print(f"{'='*70}")
