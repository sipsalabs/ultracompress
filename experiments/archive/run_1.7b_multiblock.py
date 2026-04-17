"""
1.7B MULTI-BLOCK FRR — Two shared blocks for richer functional basis.

THE INSIGHT (from weight manifold analysis):
  The 28 teacher layers are nearly ORTHOGONAL in weight space (cosine ~0.001).
  Their variation spans 27 dimensions with FLAT eigenvalue spectrum.
  One shared block captures only ~64% T10 because it provides only ONE
  functional basis for 28 layers that need 27 dimensions of variation.

THE SOLUTION:
  Two shared blocks — "early" and "late" — with per-layer SOFT MIXING.
  Each layer's output = mix[l] * block_A(x) + (1-mix[l]) * block_B(x)
  This provides 2 functional bases, doubling the expressiveness.

  Block A starts as the pre-trained 100K block (preserves quality).
  Block B starts as a copy (starts identical, then differentiates).
  mix[l] ∈ (0,1) is learned per layer (sigmoid, init=0 → mix=0.5).

PARAMETERS:
  Block A: 29.4M (pre-trained from 100K checkpoint)
  Block B: 29.4M (copy of A, differentiates during training)
  Mix weights: 28 (per-layer logits)
  Per-scale gamma/beta: 16K (shared across blocks)
  Total: ~58.8M → 26x compression (still excellent for 100T→20GB)

TRAINING:
  Phase 1 (20K): Train block B only (A frozen) + mix weights
  Phase 2 (80K): Train both blocks + mix weights jointly
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
PHASE1_STEPS = 20_000   # Block B only (A frozen) + mix
PHASE2_STEPS = 80_000   # Both blocks + mix
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 3e-4
TEMP = 2.0
N_SCALES = 4
ITERS_PER_SCALE = 7
TOTAL_LAYERS = N_SCALES * ITERS_PER_SCALE  # 28
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_multiblock'
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B MULTI-BLOCK FRR — Two shared blocks, per-layer soft mixing")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"Phase 1: {PHASE1_STEPS:,} steps (Block B + mix, A frozen)")
print(f"Phase 2: {PHASE2_STEPS:,} steps (Both blocks + mix)")
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
print(f"  Hidden: {hidden}, Heads: {n_heads}, HeadDim: {head_dim}, Vocab: {vocab_size}")

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


# ── Multi-Block FRR Model ─────────────────────────────────────────────
class MultiBlockFRR(nn.Module):
    """Two shared blocks with per-layer soft mixing.

    Novel: instead of 1 block for all layers (standard FRR), we use 2 blocks
    and let each layer CHOOSE its mix. This provides 2 functional bases,
    addressing the fundamental limit of single-block FRR (1 basis for 27D variation).
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, embed_weight, lm_head_weight, norm_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # TWO shared blocks (the core of multi-block FRR)
        self.block_a = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.block_b = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-layer mixing logit: sigmoid → mix ∈ (0,1)
        # mix=0.5 initially (equal contribution from both blocks)
        self.mix_logits = nn.Parameter(torch.zeros(self.total_layers))

        # Per-scale modulation (shared across both blocks)
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
                mix = torch.sigmoid(self.mix_logits[layer_count])

                # Both blocks process the same input with same modulation
                out_a = self.block_a(x, gamma, beta)
                out_b = self.block_b(x, gamma, beta)

                # Soft mixing: weighted combination of both blocks
                blended = mix * out_a + (1 - mix) * out_b

                # Apply with iter_scale (same as standard FRR)
                x = x + (blended - x) * iter_s
                layer_count += 1

        x = self.norm(x)
        return self.lm_head(x)

    def fractal_params(self):
        total = sum(p.numel() for p in self.block_a.parameters())
        total += sum(p.numel() for p in self.block_b.parameters())
        total += self.scale_gamma.numel() + self.scale_beta.numel()
        total += self.iter_scale.numel() + self.mix_logits.numel()
        return total


# ── Build Model ───────────────────────────────────────────────────────
print("\nBuilding Multi-Block FRR (2 shared blocks + soft mixing)...")
model = MultiBlockFRR(
    hidden, n_heads, N_SCALES, ITERS_PER_SCALE, vocab_size, 1,
    embed_w, lm_head_w, norm_w,
).to(DEVICE)

# Load pre-trained block weights into BOTH blocks
print(f"Loading base checkpoint into both blocks: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)

# Map single-block checkpoint to dual-block model
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block_a.load_state_dict(block_state)
model.block_b.load_state_dict(block_state)  # Copy — will differentiate during training

# Load modulation weights
for k in ['scale_gamma', 'scale_beta', 'iter_scale']:
    if k in ckpt:
        getattr(model, k).data.copy_(ckpt[k])
del ckpt

# Verify baseline
print("Verifying baseline (both blocks identical, mix=0.5)...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

block_a_params = sum(p.numel() for p in model.block_a.parameters())
block_b_params = sum(p.numel() for p in model.block_b.parameters())
total_fractal = model.fractal_params()
teacher_params = N_TEACHER_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)

print(f"\n  Block A: {block_a_params:,} params (pre-trained)")
print(f"  Block B: {block_b_params:,} params (copy, will differentiate)")
print(f"  Mix weights: {model.mix_logits.numel()} (per-layer)")
print(f"  Total fractal: {total_fractal:,}")
print(f"  Compression: {teacher_params/total_fractal:.1f}x")

# ── Phase 1: Train Block B + mix only (A frozen) ─────────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: Block B + mix ({PHASE1_STEPS:,} steps, LR={LR})")
print(f"  Block A FROZEN — preserves learned computation")
print(f"  Block B differentiates — learns complementary computation")
print(f"{'='*70}")

# Freeze block A and embeddings
for param in model.block_a.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if 'embed' in name or 'lm_head' in name or name == 'norm.weight':
        param.requires_grad = False

p1_params = [p for p in model.parameters() if p.requires_grad]
p1_count = sum(p.numel() for p in p1_params)
print(f"  Trainable: {p1_count:,} (Block B + modulation + mix)")

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
            mix = torch.sigmoid(model.mix_logits)
            mix_str = ' '.join(f'{m:.2f}' for m in mix[:7])
            mix_str2 = ' '.join(f'{m:.2f}' for m in mix[21:])

        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(f"    Mix (early): {mix_str}  |  (late): {mix_str2}")

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_mb_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Phase 2: Train everything jointly ─────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps, LR={LR})")
print(f"  Both blocks + modulation + mix")
print(f"{'='*70}")

# Unfreeze block A
for param in model.block_a.parameters():
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
            mix = torch.sigmoid(model.mix_logits)
            mix_str = ' '.join(f'{m:.2f}' for m in mix[:7])
            mix_str2 = ' '.join(f'{m:.2f}' for m in mix[21:])

        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (g={global_step}): loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )
        print(f"    Mix (early): {mix_str}  |  (late): {mix_str2}")

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_mb_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_multiblock_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"  Final: T1={t1*100:.1f}%, T10={t10*100:.1f}%")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {time.time()-t0:.0f}s")
print(f"{'='*70}")

# Block specialization analysis
print(f"\nBLOCK SPECIALIZATION ANALYSIS:")
print(f"  Mix values: 1.0 = all Block A, 0.0 = all Block B")
with torch.no_grad():
    mix = torch.sigmoid(model.mix_logits)
    for scale in range(N_SCALES):
        start = scale * ITERS_PER_SCALE
        end = start + ITERS_PER_SCALE
        mix_slice = mix[start:end]
        mix_str = ' '.join(f'{m:.3f}' for m in mix_slice)
        avg_mix = mix_slice.mean().item()
        print(f"  Scale {scale} (layers {start}-{end-1}): {mix_str}  avg={avg_mix:.3f}")

    # Weight divergence
    cos_sims = []
    for (na, pa), (nb, pb) in zip(model.block_a.named_parameters(), model.block_b.named_parameters()):
        cos = F.cosine_similarity(pa.flatten().unsqueeze(0), pb.flatten().unsqueeze(0)).item()
        cos_sims.append((na, cos))
    print(f"\n  Block A vs B weight cosine similarity:")
    for name, cos in cos_sims:
        print(f"    {name}: {cos:.6f}")
