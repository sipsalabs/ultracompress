"""
1.7B FROZEN BLOCK + PURE KL + PER-LAYER GAMMA/BETA

DIAGNOSTIC EXPERIMENT: Under pure KL (run 1.7b_pure_kl.py), KL loss rose
33.9 → 43.3 over 7500 steps EVEN THOUGH CE term was removed. This means
the block fine-tuning at LR=3e-5 is DEGRADING the pre-trained block weights
against the pure KL objective.

HYPOTHESIS: Freezing the block entirely and only training the 28 per-layer
gamma/beta vectors should:
  1. Keep KL loss STABLE or DECREASING (no block degradation)
  2. Still capture per-layer variation via mod-only
  3. Isolate the modulation contribution from block changes

COMPARISON:
  - Pure KL (unfrozen block, LR=3e-5): last-tok T10=69.4% @ step 5000, KL rising
  - Per-layer mod (frozen, KL+CE): last-tok CRASHED, KL+CE term conflict
  - NEW (frozen, pure KL): hopefully stable monotonic improvement

Higher mod LR (1e-3 vs 3e-4) since only 114K params need to move.
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
LR_MOD = 1e-3           # higher LR since only mod is training
WD_MOD = 0.01
BATCH_SIZE = 4
SEQ_LEN = 64
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'
CHECKPOINT_DIR = 'checkpoints_1.7b_frozen_block'
N_TEACHER_LAYERS = 28

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B FROZEN BLOCK + PURE KL + PER-LAYER MOD")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"LR_mod: {LR_MOD}  (block FROZEN)")
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

        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

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
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = self.block(x, gamma, beta)
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
head_dim = hidden // n_heads
vocab_size = gd['token_embd.weight'].shape[0]

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
print(f"\nLoading data from {DATA_PATH}...")
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
print(f"\nBuilding Per-Layer FRR (block will be FROZEN)...")
model = PerLayerFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
block_state = {k.replace('block.', ''): v for k, v in ckpt.items() if k.startswith('block.')}
model.block.load_state_dict(block_state)

scale_gamma = ckpt['scale_gamma']
scale_beta = ckpt['scale_beta']
for scale in range(4):
    for it in range(7):
        layer_idx = scale * 7 + it
        model.layer_gamma.data[layer_idx] = scale_gamma[scale]
        model.layer_beta.data[layer_idx] = scale_beta[scale]
model.iter_scale.data.copy_(ckpt['iter_scale'])

# FREEZE block
for p in model.block.parameters():
    p.requires_grad = False

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_trainable:,}  (block frozen)")


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


print("\nBaseline (before any training):")
t1_base, t10_base = eval_vs_teacher(model, n=50)
t1_last, t10_last = eval_vs_teacher_last(model, n=100)
print(f"  all-pos: T1={t1_base*100:.1f}%  T10={t10_base*100:.1f}%")
print(f"  last-tok: T1={t1_last*100:.1f}%  T10={t10_last*100:.1f}%")


# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TRAINING: FROZEN BLOCK + PURE KL ({TOTAL_STEPS:,} steps)")
print(f"  Only training per-layer gamma/beta + iter_scale (mod only)")
print(f"{'='*70}")

mod_params_list = [model.layer_gamma, model.layer_beta, model.iter_scale]
opt = torch.optim.AdamW(mod_params_list, lr=LR_MOD, weight_decay=WD_MOD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

loss_history = []
best_t10 = 0.0
best_t10_last = 0.0
best_step = 0
t0 = time.time()

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
    torch.nn.utils.clip_grad_norm_(mod_params_list, 1.0)
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
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'frozen_block_best.pt'))
        if t10_l > best_t10_last:
            best_t10_last = t10_l

        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        with torch.no_grad():
            gamma_std = model.layer_gamma.std(dim=0).mean().item()
            beta_std = model.layer_beta.std(dim=0).mean().item()

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: KL_loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={opt.param_groups[0]['lr']:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    [last-tok] T1={t1_l*100:.1f}%  T10={t10_l*100:.1f}%  "
            f"gamma_std={gamma_std:.4f}  beta_std={beta_std:.4f}  "
            f"best_last_t10={best_t10_last*100:.1f}%"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frozen_block_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

final_path = os.path.join(CHECKPOINT_DIR, 'frozen_block_final.pt')
torch.save(model.state_dict(), final_path)
total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"FROZEN BLOCK + PURE KL COMPLETE")
print(f"  Best T10 (all-pos): {best_t10*100:.1f}% at step {best_step}")
print(f"  Best T10 (last-tok): {best_t10_last*100:.1f}%")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"{'='*70}")
