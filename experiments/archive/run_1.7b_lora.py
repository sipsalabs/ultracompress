"""
1.7B LoRA-AUGMENTED FRR — Per-layer specialization experiment.

Key insight: FRR's single shared block captures the "average" layer computation.
LoRA adapters let each virtual layer deviate slightly from that average.

Design:
  - Load best 100K checkpoint (already well-trained shared block)
  - Enable rank-16 LoRA adapters on all 28 virtual layers (+1.8M params)
  - Phase 1 (20K steps): Train ONLY LoRA params (frozen block)
  - Phase 2 (30K steps): Train ALL params jointly
  - Total: 50K additional steps

Parameter budget:
  - Shared block: 29.4M (frozen in phase 1)
  - LoRA adapters: 1.8M (28 layers × 2 × 2048 × 16)
  - Total: 31.2M trainable, ~49x compression (still excellent)

Expected outcome: If layers need specialization beyond affine modulation,
LoRA should improve T10 significantly. If not, we learn the shared block
is already near-optimal for this architecture.
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
from ultracompress.moonshot import FractalModel

# ── Configuration ─────────────────────────────────────────────────────
DEVICE = 'cuda:1'  # Will use GPU 1 when available
LORA_RANK = 16
PHASE1_STEPS = 20_000   # LoRA only
PHASE2_STEPS = 30_000   # All params
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
BATCH_SIZE = 4
SEQ_LEN = 64
LR_PHASE1 = 1e-3    # Higher LR for LoRA-only (small params, fast adaptation)
LR_PHASE2 = 1e-4    # Lower LR for joint fine-tuning
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_lora'
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"1.7B LoRA-AUGMENTED FRR — rank-{LORA_RANK} per-layer specialization")
print(f"Device: {DEVICE}  |  Temp: {TEMP} (fixed)")
print(f"Phase 1 ({PHASE1_STEPS:,} steps): LoRA only, LR={LR_PHASE1}")
print(f"Phase 2 ({PHASE2_STEPS:,} steps): All params, LR={LR_PHASE2}")
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
N_LAYERS = 28
for li in range(N_LAYERS):
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
    n_layers=N_LAYERS, n_heads=n_heads, n_kv_heads=8,
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

# ── Data (pre-tokenized local file for speed) ────────────────────────
LOCAL_TOKENS_FILE = 'fineweb_edu_100M_tokens.pt'
print(f"Loading pre-tokenized data from {LOCAL_TOKENS_FILE}...")
ALL_TOKENS = torch.load(LOCAL_TOKENS_FILE, weights_only=True).to(torch.long)
print(f"  {ALL_TOKENS.numel():,} tokens loaded")


def get_real_batch(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    starts = torch.randint(0, ALL_TOKENS.numel() - seq_len, (batch_size,))
    batch = torch.stack([ALL_TOKENS[s:s + seq_len] for s in starts])
    return batch.to(DEVICE)


def eval_vs_teacher(model: nn.Module, n: int = 100) -> tuple[float, float]:
    model.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        starts = torch.randint(0, ALL_TOKENS.numel() - 32, (1,))
        tokens = ALL_TOKENS[starts[0]:starts[0] + 32].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=N_LAYERS)
            sl = model(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1_hits / n, t10_hits / n


# ── Build FRR Student with LoRA ───────────────────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load pre-trained weights
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt, strict=False)
del ckpt

# Enable LoRA adapters (starts as identity — no quality change)
model.enable_adapters(rank=LORA_RANK)

# Verify baseline quality hasn't changed
print("Verifying baseline (should match 100K results)...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

trainable_block = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and 'adapter' not in n)
trainable_lora = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and 'adapter' in n)
trainable_total = trainable_block + trainable_lora
teacher_params = N_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
print(f"  Block params: {trainable_block:,}")
print(f"  LoRA params: {trainable_lora:,}")
print(f"  Total trainable: {trainable_total:,}, compression: {teacher_params/trainable_total:.1f}x")

# ── Phase 1: Train LoRA only (freeze shared block) ───────────────────
print(f"\n{'='*70}")
print(f"PHASE 1: LoRA-only training ({PHASE1_STEPS:,} steps, LR={LR_PHASE1})")
print(f"{'='*70}")

# Freeze all non-LoRA params
for name, param in model.named_parameters():
    if 'adapter' not in name:
        param.requires_grad = False

lora_params = [p for p in model.parameters() if p.requires_grad]
lora_count = sum(p.numel() for p in lora_params)
print(f"  Training {lora_count:,} LoRA params only (block frozen)")

opt1 = torch.optim.AdamW(lora_params, lr=LR_PHASE1, weight_decay=0.01)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, PHASE1_STEPS)
t0 = time.time()
best_t10 = t10_base
best_step = -1
loss_history = []

for step in range(PHASE1_STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt1.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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
        print(
            f"  P1 Step {step:>6d}/{PHASE1_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched1.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_lora_p1_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

# Save end of phase 1
p1_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_lora_p1_final.pt')
torch.save(model.state_dict(), p1_path)
p1_t1, p1_t10 = eval_vs_teacher(model, n=200)
print(f"\n  Phase 1 complete: T1={p1_t1*100:.1f}%, T10={p1_t10*100:.1f}% (200-sample)")
print(f"  Best T10 in phase 1: {best_t10*100:.1f}% at step {best_step}")

# ── Phase 2: Train ALL params jointly ─────────────────────────────────
print(f"\n{'='*70}")
print(f"PHASE 2: Joint training ({PHASE2_STEPS:,} steps, LR={LR_PHASE2})")
print(f"{'='*70}")

# Unfreeze everything
for name, param in model.named_parameters():
    if 'embed' not in name and 'lm_head' not in name and 'norm' not in name:
        param.requires_grad = True

all_params = [p for p in model.parameters() if p.requires_grad]
all_count = sum(p.numel() for p in all_params)
print(f"  Training {all_count:,} params (block + modulation + LoRA)")

opt2 = torch.optim.AdamW(all_params, lr=LR_PHASE2, weight_decay=0.01)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, PHASE2_STEPS)
t0_p2 = time.time()

for step in range(PHASE2_STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
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
        print(
            f"  P2 Step {step:>6d}/{PHASE2_STEPS} (global {global_step}): loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched2.get_last_lr()[0]:.6f}  ({total_elapsed:.0f}s){new_best}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_lora_p2_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_lora_final.pt')
torch.save(model.state_dict(), final_path)

t1_final, t10_final = eval_vs_teacher(model, n=200)

print(f"\n{'='*70}")
print(f"LoRA EXPERIMENT COMPLETE")
print(f"  Total time: {total_time/3600:.1f} hours")
print(f"  Baseline (100K no LoRA): T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")
print(f"  Phase 1 final (LoRA only): T1={p1_t1*100:.1f}%, T10={p1_t10*100:.1f}%")
print(f"  Final (200-sample): T1={t1_final*100:.1f}%, T10={t10_final*100:.1f}%")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Improvement over baseline: {(t10_final - t10_base)*100:+.1f}pp T10")
print(f"  Compression: {teacher_params/trainable_total:.1f}x")
print(f"  Saved: {final_path}")
print(f"{'='*70}")
