"""
1.7B PURE KL DISTILLATION (No CE Term)

CRITICAL INSIGHT: Every experiment since baseline (resonance v1/v2, SAS v1/v2,
per-layer mod) uses loss = KL + 0.3 * CE(teacher_argmax). The CE term forces
distribution sharpening on teacher's top-1, which FIGHTS the KL objective that
tries to match the full distribution.

EVIDENCE:
  - Loss ALWAYS increases (34.86→41→42) because KL and CE have conflicting gradients
  - All-pos T10 improves (63.3% at step 5000 in per-layer mod) but last-tok CRASHES
  - Peak at step 2500, drop by 5000 pattern in EVERY experiment
  - SAS v1 and v2 had IDENTICAL loss=42.11 at step 2500 despite very different QKmag

HYPOTHESIS: Removing the CE term will:
  1. Stop the loss from increasing (pure KL should decrease monotonically)
  2. Allow the model to match the broader distribution (help T10)
  3. Prevent the peak-then-drop pattern (no conflicting gradients)
  4. Both all-pos AND last-tok should improve together

APPROACH: Per-layer gamma/beta (proven to help all-pos T10) + pure KL loss.
Block unfrozen at 10x lower LR to allow co-adaptation.

Two param groups:
  - Block: LR=3e-5, WD=0.05 (careful adaptation)
  - Per-layer gamma/beta: LR=3e-4, WD=0.01 (faster differentiation)
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
LR_BLOCK = 3e-5        # block = 10x lower
LR_MOD = 3e-4           # gamma/beta = standard
WD_BLOCK = 0.05
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_pure_kl'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B PURE KL DISTILLATION (No CE Term)")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"LR_block: {LR_BLOCK}  |  LR_mod: {LR_MOD}")
print(f"Loss: PURE KL (no CE term)")
print("=" * 70)


class PerLayerFRR(nn.Module):
    """FRR with per-iteration (per-layer) gamma/beta instead of per-scale."""
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult,
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

        # PER-LAYER modulation
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))

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

    def forward(self, tokens):
        x = self.embed(tokens).float()

        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = self.block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def block_params(self):
        return sum(p.numel() for p in self.block.parameters())

    def mod_params(self):
        return self.layer_gamma.numel() + self.layer_beta.numel() + self.iter_scale.numel()

    def fractal_params(self):
        return self.block_params() + self.mod_params()


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
print(f"\nBuilding Per-Layer FRR...")
model = PerLayerFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load pre-trained block weights
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)

# Convert per-scale gamma/beta → per-layer
scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
for scale in range(4):
    for it in range(7):
        layer_idx = scale * 7 + it
        model.layer_gamma.data[layer_idx] = scale_gamma[scale]
        model.layer_beta.data[layer_idx] = scale_beta[scale]
model.iter_scale.data.copy_(ckpt['iter_scale'])
print(f"  Loaded block + converted scale->layer modulation")

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
    """Last token only eval."""
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


print("\nVerifying baseline (should match pre-trained)...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
t1_last, t10_last = eval_vs_teacher_last(model, n=100)
print(f"  Baseline (all-pos): T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")
print(f"  Baseline (last-tok): T1={t1_last*100:.1f}%, T10={t10_last*100:.1f}%")

block_p = model.block_params()
mod_p = model.mod_params()
total_p = block_p + mod_p
print(f"\n  Block params: {block_p:,}")
print(f"  Per-layer mod params: {mod_p:,}")
print(f"  Total trainable: {total_p:,}")
print(f"  Compression: {N_TEACHER_LAYERS * 7 * hidden * hidden / model.fractal_params():.1f}x")


# ── Training: Pure KL with all params ────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: PURE KL — block + per-layer gamma/beta ({TOTAL_STEPS:,} steps)")
print(f"  Block LR={LR_BLOCK} WD={WD_BLOCK}  |  Mod LR={LR_MOD} WD={WD_MOD}")
print(f"  Loss: PURE KL divergence (NO CE term)")
print(f"{'='*70}")

# Two param groups with different LRs
block_params_list = list(model.block.parameters())
mod_params_list = [model.layer_gamma, model.layer_beta, model.iter_scale]
opt = torch.optim.AdamW([
    {'params': block_params_list, 'lr': LR_BLOCK, 'weight_decay': WD_BLOCK},
    {'params': mod_params_list, 'lr': LR_MOD, 'weight_decay': WD_MOD},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

total_trainable = sum(p.numel() for group in opt.param_groups for p in group['params'])
print(f"  Trainable: {total_trainable:,}")

loss_history = []
kl_history = []
best_t10 = 0.0
best_step = 0
t0 = time.time()

for step in range(TOTAL_STEPS):
    tokens = get_real_batch()
    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)

    sl = model(tokens)

    # ★ PURE KL — no CE term ★
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
        t1_l, t10_l = eval_vs_teacher_last(model, n=100)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
            # Save best
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'pure_kl_best.pt'))

        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        # Diagnostics
        with torch.no_grad():
            gamma_std = model.layer_gamma.std(dim=0).mean().item()
            beta_std = model.layer_beta.std(dim=0).mean().item()
            # Block weight change magnitude
            block_grad_norm = sum(
                p.grad.norm().item() for p in model.block.parameters() if p.grad is not None
            )

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL_loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR_blk={opt.param_groups[0]['lr']:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok] T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%  "
            f"gamma_std={gamma_std:.4f}  beta_std={beta_std:.4f}  "
            f"blk_grad={block_grad_norm:.4f}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'pure_kl_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Final ─────────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'pure_kl_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"PURE KL DISTILLATION COMPLETE")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Final checkpoint: {final_path}")
print(f"{'='*70}")
