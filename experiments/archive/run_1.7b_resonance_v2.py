"""
1.7B RESONANT FRR v2 — Fixed gradient + output-side modulation.

FIXES from v1:
1. Frame rotation: tanh(mag) instead of mag² → gradient alive at init
2. Epigenetic markers: applied POST-block (preserve block's T1 precision)
3. Better ordering: block sees clean x, novel mechanisms adjust output

THE ARCHITECTURE:
  For each virtual layer l:
    block_out = Block(x, gamma, beta)              # block sees clean input
    rotated = Householder(block_out, v_l, tanh(m_l))  # frame rotate output
    marked = Epigenetic(rotated, dirs_l, mags_l)   # directional adjust
    augmented = marked + MemoryInject(memory_l)     # add memory trace
    x = x + (augmented - x) * iter_scale           # residual

  This preserves the block's attention patterns (T1) while allowing
  novel mechanisms to reshape the output (T10).
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
K_DIRECTIONS = 4
MEMORY_DIM = 64
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_resonance_v2'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B RESONANT FRR v2 — Output-side modulation, fixed gradients")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"Phase 1: {PHASE1_STEPS:,} steps (novel mechanisms, block frozen)")
print(f"Phase 2: {PHASE2_STEPS:,} steps (everything jointly)")
print(f"Fixes: tanh(mag) for frame, post-block epigenetic, output-side memory")
print("=" * 70)


class ResonantFRRv2(nn.Module):
    """
    Resonant FRR v2 — novel mechanisms applied POST-block.

    Key change: Block sees CLEAN input → preserves T1 precision.
    Novel mechanisms adjust block OUTPUT → adds T10 diversity.
    """
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, k_directions=4, memory_dim=64,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.k_directions = k_directions
        self.memory_dim = memory_dim

        # Shared block
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Existing modulation
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

        # NOVEL 1: Frame Rotation (Householder reflections)
        # FIXED: tanh(mag) for alive gradient at init
        self.frame_dirs = nn.Parameter(
            F.normalize(torch.randn(self.total_layers, hidden_dim), dim=-1)
        )
        self.frame_mag_logits = nn.Parameter(torch.zeros(self.total_layers))
        # mag = tanh(logit). logit=0 → mag=0 → no rotation. d/d(logit) tanh(0) = 1.

        # NOVEL 2: Epigenetic Markers (directional output modulation)
        dirs = torch.randn(self.total_layers, k_directions, hidden_dim)
        dirs = F.normalize(dirs, dim=-1)
        self.epi_dirs = nn.Parameter(dirs)
        self.epi_mags = nn.Parameter(torch.zeros(self.total_layers, k_directions))

        # NOVEL 3: Fractal Memory
        self.mem_proj_down = nn.Linear(hidden_dim, memory_dim, bias=False)
        self.mem_proj_up = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.mem_decay_logits = nn.Parameter(
            torch.full((self.total_layers,), 2.0)
        )
        nn.init.zeros_(self.mem_proj_up.weight)

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

    def _householder(self, x, direction, mag_logit):
        """Householder reflection with tanh-scaled magnitude.

        R(x) = x - 2 * tanh(logit) * (v·x / v·v) * v

        tanh(0) = 0 → identity. d/d(logit) tanh(0) = 1 → full gradient.
        |tanh| ≤ 1 → bounded rotation magnitude (stable).
        """
        mag = torch.tanh(mag_logit)
        vx = torch.einsum('d,btd->bt', direction, x)
        vv = torch.dot(direction, direction).clamp(min=1e-8)
        scale = 2.0 * mag / vv
        return x - scale * vx.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)

    def _apply_epigenetic(self, x, layer_idx):
        """Directional amplification — applied to block output."""
        dirs = F.normalize(self.epi_dirs[layer_idx], dim=-1)
        mags = self.epi_mags[layer_idx]
        proj = torch.einsum('btd,kd->btk', x, dirs)
        delta = torch.einsum('btk,k,kd->btd', proj, mags, dirs)
        return x + delta

    def forward(self, tokens):
        x = self.embed(tokens).float()
        B, T, D = x.shape
        memory = torch.zeros(B, self.memory_dim, device=x.device, dtype=x.dtype)

        layer_idx = 0
        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]
            for it in range(self.iters_per_scale):
                iter_s = self.iter_scale[scale, it]

                # Block sees CLEAN input (preserves T1)
                block_out = self.block(x, gamma, beta)

                # NOVEL 1: Frame rotation on OUTPUT
                rotated = self._householder(
                    block_out, self.frame_dirs[layer_idx],
                    self.frame_mag_logits[layer_idx],
                )

                # NOVEL 2: Epigenetic markers on OUTPUT
                marked = self._apply_epigenetic(rotated, layer_idx)

                # NOVEL 3: Memory injection on OUTPUT
                mem_inject = self.mem_proj_up(memory)
                augmented = marked + mem_inject.unsqueeze(1)

                # Residual with iter_scale
                x = x + (augmented - x) * iter_s

                # NOVEL 3: Update memory from processed output
                decay = torch.sigmoid(self.mem_decay_logits[layer_idx])
                block_summary = x.mean(dim=1)
                mem_update = self.mem_proj_down(block_summary)
                memory = decay * memory + (1 - decay) * mem_update

                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def novel_params(self):
        total = 0
        total += self.frame_dirs.numel() + self.frame_mag_logits.numel()
        total += self.epi_dirs.numel() + self.epi_mags.numel()
        total += sum(p.numel() for p in self.mem_proj_down.parameters())
        total += sum(p.numel() for p in self.mem_proj_up.parameters())
        total += self.mem_decay_logits.numel()
        return total

    def fractal_params(self):
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
print(f"\nBuilding Resonant FRR v2...")
model = ResonantFRRv2(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    k_directions=K_DIRECTIONS, memory_dim=MEMORY_DIM,
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
t1_base, t10_base = eval_vs_teacher(model, n=100)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

novel_p = model.novel_params()
fractal_p = model.fractal_params()
print(f"\n  Novel mechanisms: {novel_p:,} params")
print(f"  Total fractal: {fractal_p:,}")
print(f"  Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / fractal_p:.1f}x")


# ── Phase 1: Novel mechanisms only ────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: Novel mechanisms only ({PHASE1_STEPS:,} steps, LR={LR})")
print(f"  Block FROZEN — frame rotation, epigenetic markers, memory converge")
print(f"{'='*70}")

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

        frame_mag = torch.tanh(model.frame_mag_logits).abs().mean().item()
        epi_mag = model.epi_mags.abs().mean().item()
        decay = torch.sigmoid(model.mem_decay_logits).mean().item()

        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    Frame|mag|={frame_mag:.4f}  "
            f"Epi|mag|={epi_mag:.4f}  "
            f"MemDecay={decay:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'resonant_v2_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")


# ── Phase 2: Everything jointly ───────────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps, LR={LR})")
# CRITICAL FIX: Use lower LR for pre-trained block to avoid destroying it
P2_BLOCK_LR = LR * 0.1  # 10x lower for pre-trained weights
print(f"  Block LR: {P2_BLOCK_LR} (10x lower to preserve quality)")
print(f"  Novel LR: {LR}")
print(f"{'='*70}")

for param in model.block.parameters():
    param.requires_grad = True
model.scale_gamma.requires_grad = True
model.scale_beta.requires_grad = True
model.iter_scale.requires_grad = True

# Separate param groups: lower LR for pre-trained components
pretrained_params = list(model.block.parameters()) + [
    model.scale_gamma, model.scale_beta, model.iter_scale
]
novel_param_ids = set(id(p) for p in p1_params)
novel_params_p2 = [p for p in model.parameters()
                   if p.requires_grad and id(p) in novel_param_ids]
pretrained_count = sum(p.numel() for p in pretrained_params)
novel_count = sum(p.numel() for p in novel_params_p2)
print(f"  Pre-trained: {pretrained_count:,} @ LR={P2_BLOCK_LR}")
print(f"  Novel: {novel_count:,} @ LR={LR}")

opt2 = torch.optim.AdamW([
    {'params': pretrained_params, 'lr': P2_BLOCK_LR},
    {'params': novel_params_p2, 'lr': LR},
], weight_decay=0.01)
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
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()), 1.0
    )
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

        frame_mag = torch.tanh(model.frame_mag_logits).abs().mean().item()
        epi_mag = model.epi_mags.abs().mean().item()
        decay = torch.sigmoid(model.mem_decay_logits).mean().item()

        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (g={global_step}): "
            f"loss={avg_loss:.4f}  T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )
        print(
            f"    Frame|mag|={frame_mag:.4f}  "
            f"Epi|mag|={epi_mag:.4f}  "
            f"MemDecay={decay:.3f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'resonant_v2_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'resonant_v2_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"RESONANT FRR v2 COMPLETE")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Final checkpoint: {final_path}")
print(f"{'='*70}")
