"""
1.7B DUAL-BLOCK FINE-TUNE (pure KL)

HYPOTHESIS: 1 shared block caps T10 at ~68.2% (hires) because 28 teacher
layers are nearly orthogonal (cosine ~0.001). Need more capacity.

APPROACH: Duplicate the pre-trained block into 2 blocks:
  - block_early: used for scales 0-1 (layers 0-13)
  - block_late:  used for scales 2-3 (layers 14-27)

Both start from the same pre-trained weights, will diverge through fine-tuning.
Pure KL loss. Per-layer gamma/beta (28 sets) also trained.

Compression: 28×2048² × 3 / (2×29M + 114K) ≈ 13.5x (down from 28x)
Params: 2×29.4M block + 114K mod = 58.8M trainable

Why split at scale 1/2 boundary?
  - Scale 0 (layers 0-6): iter_scale 0.24-0.46 (light touch, early features)
  - Scale 1 (layers 7-13): iter_scale 1.06-2.08 (moderate, increasing)
  - Scale 2 (layers 14-20): iter_scale 2.01-3.19 (DOMINANT — most important)
  - Scale 3 (layers 21-27): iter_scale ~1.97 (late integration)
  This puts the dominant scale 2 in its own block.
"""
import lib.unbuffered
import torch
import sys
import os
import time

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

# ── Configuration ─────────────────────────────────────────────────────
DEVICE = 'cuda:1'
TOTAL_STEPS = 20_000
LR_BLOCK = 3e-5         # gentle LR so blocks diverge slowly
LR_MOD = 3e-4
WD_BLOCK = 0.05
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_dual_block'
N_TEACHER_LAYERS = 28
SPLIT_SCALE = 2  # scales 0,1 use block_early; scales 2,3 use block_late

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B DUAL-BLOCK FINE-TUNE (Pure KL)")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"LR_block: {LR_BLOCK}  |  LR_mod: {LR_MOD}")
print(f"Split at scale {SPLIT_SCALE}: early={0}-{SPLIT_SCALE-1}, late={SPLIT_SCALE}-3")
print("=" * 70)


class DualBlockFRR(nn.Module):
    """FRR with 2 shared blocks split by scale."""
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, split_scale=2,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.split_scale = split_scale
        self.total_layers = n_scales * iters_per_scale

        # TWO blocks instead of one
        self.block_early = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.block_late = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-layer modulation (28 sets)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

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

    def forward(self, tokens):
        x = self.embed(tokens).float()

        layer_idx = 0
        for scale in range(self.n_scales):
            # Select block based on scale
            block = self.block_early if scale < self.split_scale else self.block_late
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)


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

# ── Data ──────────────────────────────────────────────────────────────
print("\nLoading pre-tokenized data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
print(f"  {N_TOKENS:,} tokens")

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
print(f"\nBuilding Dual-Block FRR...")
model = DualBlockFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1, split_scale=SPLIT_SCALE,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load pretrained block into BOTH blocks
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block_early.load_state_dict(block_state)
model.block_late.load_state_dict(block_state)  # identical init
print(f"  Both blocks initialized from pretrained (identical).")

# Convert per-scale → per-layer gamma/beta
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
for scale in range(4):
    for it in range(7):
        layer_idx = scale * 7 + it
        model.layer_gamma.data[layer_idx] = scale_gamma[scale]
        model.layer_beta.data[layer_idx] = scale_beta[scale]
model.iter_scale.data.copy_(ckpt['iter_scale'])

n_block = sum(p.numel() for p in model.block_early.parameters()) * 2
n_mod = model.layer_gamma.numel() + model.layer_beta.numel() + model.iter_scale.numel()
print(f"  Block params (2×): {n_block:,}")
print(f"  Mod params: {n_mod:,}")
print(f"  Total trainable: {n_block + n_mod:,}")
print(f"  Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / (n_block + n_mod):.1f}x")


# ── Eval ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_vs_teacher(mdl, n=100):
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


print("\nBaseline (before training):")
t1_base, t10_base = eval_vs_teacher(model, n=50)
t1_last, t10_last = eval_vs_teacher_last(model, n=200)  # more samples for stability
print(f"  all-pos:  T1={t1_base*100:.1f}%  T10={t10_base*100:.1f}%")
print(f"  last-tok: T1={t1_last*100:.1f}%  T10={t10_last*100:.1f}% (n=200)")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: DUAL-BLOCK PURE KL ({TOTAL_STEPS:,} steps)")
print(f"{'='*70}")

block_params = (list(model.block_early.parameters())
                + list(model.block_late.parameters()))
mod_params = [model.layer_gamma, model.layer_beta, model.iter_scale]
opt = torch.optim.AdamW([
    {'params': block_params, 'lr': LR_BLOCK, 'weight_decay': WD_BLOCK},
    {'params': mod_params, 'lr': LR_MOD, 'weight_decay': WD_MOD},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_history = []
best_t10 = 0.0
best_t10_last = 0.0
best_step = 0
t0 = time.time()

# Track block divergence
with torch.no_grad():
    init_early = torch.cat([p.flatten() for p in model.block_early.parameters()]).clone()
    init_late = torch.cat([p.flatten() for p in model.block_late.parameters()]).clone()

for step in range(TOTAL_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    loss_history.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=50)
        t1_l, t10_l = eval_vs_teacher_last(model, n=200)
        elapsed = time.time() - t0

        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST (all-pos) ***"
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'dual_block_best.pt'))
        if t10_l > best_t10_last:
            best_t10_last = t10_l

        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        # Compute block divergence
        with torch.no_grad():
            now_early = torch.cat([p.flatten() for p in model.block_early.parameters()])
            now_late = torch.cat([p.flatten() for p in model.block_late.parameters()])
            drift_early = (now_early - init_early).norm().item() / init_early.norm().item()
            drift_late = (now_late - init_late).norm().item() / init_late.norm().item()
            divergence = (now_early - now_late).norm().item() / now_early.norm().item()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL_loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok n=200] T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%  "
            f"best_last={best_t10_last*100:.1f}%"
        )
        print(
            f"    drift: early={drift_early*100:.2f}%  late={drift_late*100:.2f}%  "
            f"divergence={divergence*100:.2f}%"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'dual_block_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

final_path = os.path.join(CHECKPOINT_DIR, 'dual_block_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"DUAL-BLOCK TRAINING COMPLETE")
print(f"  Best T10 (all-pos): {best_t10*100:.1f}% at step {best_step}")
print(f"  Best T10 (last-tok, n=200): {best_t10_last*100:.1f}%")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"{'='*70}")
