"""
1.7B CYCLIC TEMPERATURE — Resume from best checkpoint with temperature cycling.

Hypothesis: The T10 decline (62.4% → 60.3%) despite loss decreasing is caused
by monotonic temperature annealing (5.0 → 2.0). As T drops, KL focuses on
fewer top tokens, causing metric-objective misalignment.

Solution: Cycle temperature between T_high=4.0 and T_low=2.0 with warm restarts
(analogous to SGDR for learning rate). This alternates between:
  - High T: broad distribution matching (good for T10 coverage)
  - Low T: sharp token focusing (good for T1 precision)

Loads from the best 1.7B checkpoint (10K steps, T10=62.4%) and continues
for 50K more steps with cyclic temp.

Usage:
  python run_1.7b_cyclic_temp.py                     # Default: cuda:0
  python run_1.7b_cyclic_temp.py --device cuda:1      # Specify GPU
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
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Configuration ─────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--checkpoint', default='checkpoints_1.7b_real_text/frr_1.7b_best.pt')
parser.add_argument('--steps', type=int, default=50_000)
parser.add_argument('--t-high', type=float, default=4.0, help="Max temperature in cycle")
parser.add_argument('--t-low', type=float, default=2.0, help="Min temperature in cycle")
parser.add_argument('--cycle-length', type=int, default=10_000, help="Steps per temp cycle")
cmd_args = parser.parse_args()

DEVICE = cmd_args.device
STEPS = cmd_args.steps
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 2e-4  # Lower LR since we're resuming (was 5e-4 initially)
EVAL_INTERVAL = 2_500       # More frequent eval to see temp effects
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_cyclic_temp'
T_HIGH = cmd_args.t_high
T_LOW = cmd_args.t_low
CYCLE_LEN = cmd_args.cycle_length

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B CYCLIC TEMPERATURE — Resume from best with temp warm restarts")
print(f"Device: {DEVICE}")
print(f"Resume from: {cmd_args.checkpoint}")
print(f"Temperature: {T_LOW} ↔ {T_HIGH}, cycle={CYCLE_LEN} steps")
print(f"LR: {LR} (reduced for fine-tuning)")
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

# ── Data ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT",
    split="train", streaming=True,
)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")


def get_real_batch(batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    global ds_iter
    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                toks = tokenizer.encode(
                    text, max_length=seq_len, truncation=True, return_tensors='pt',
                )[0]
                if len(toks) >= seq_len:
                    tokens_list.append(toks[:seq_len])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(DEVICE)


def eval_vs_teacher(model: nn.Module, n: int = 200) -> tuple[float, float]:
    """T1 and T10 agreement (200 samples for more stability)."""
    t1_correct, t10_scores = 0, []
    model.eval()
    for _ in range(n):
        batch = get_real_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=N_LAYERS)
            sl = model(batch)
            t_top = tl[0, -1].argmax().item()
            s_top = sl[0, -1].argmax().item()
            t_top10 = set(tl[0, -1].topk(10).indices.tolist())
            s_top10 = set(sl[0, -1].topk(10).indices.tolist())
            if t_top == s_top:
                t1_correct += 1
            t10_scores.append(len(t_top10 & s_top10) / 10)
    model.train()
    return t1_correct / n, sum(t10_scores) / len(t10_scores)


def cyclic_temperature(step: int) -> float:
    """Cosine cycling between T_HIGH and T_LOW.

    Uses cosine annealing within each cycle, like SGDR warm restarts.
    At cycle boundaries, temperature resets to T_HIGH.
    """
    cycle_pos = step % CYCLE_LEN
    # Cosine annealing within cycle: T_HIGH → T_LOW
    t = T_LOW + 0.5 * (T_HIGH - T_LOW) * (1 + math.cos(math.pi * cycle_pos / CYCLE_LEN))
    return t


# ── Build FRR Student (from checkpoint) ──────────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load checkpoint
print(f"Loading checkpoint: {cmd_args.checkpoint}")
state = torch.load(cmd_args.checkpoint, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
del state

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FRR: {trainable:,} params (52.0x compression)")

# Initial eval
t1_init, t10_init = eval_vs_teacher(model, n=200)
print(f"Checkpoint quality: T1={t1_init*100:.1f}% T10={t10_init*100:.1f}% (200 samples)")

# ── Training ──────────────────────────────────────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
# Cosine annealing with warm restarts (matching temp cycles)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=CYCLE_LEN, T_mult=1)
t0 = time.time()
best_t10 = t10_init
best_step = -1

print(f"\nTraining {STEPS:,} steps with CYCLIC TEMPERATURE...")
print(f"  Batch: {BATCH_SIZE}×{SEQ_LEN}, LR: {LR}")
print(f"  Temp cycles: {T_LOW}↔{T_HIGH}, period={CYCLE_LEN}")
print(f"  LR schedule: CosineAnnealingWarmRestarts (same period)")
print()

for step in range(STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    T = cyclic_temperature(step)
    loss = F.kl_div(
        F.log_softmax(sl / T, dim=-1),
        F.softmax(tl / T, dim=-1),
        reduction='batchmean',
    ) * T * T

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    # ── Evaluation ────────────────────────────────────────────────
    if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=200)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        cycle_num = step // CYCLE_LEN + 1
        print(
            f"  Step {step:>6d}/{STEPS}: loss={loss.item():.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"T={T:.2f}  cycle={cycle_num}  ({elapsed:.0f}s){new_best}"
        )

    # ── Save Checkpoint ───────────────────────────────────────────
    if step > 0 and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS - 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_cyclic_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_cyclic_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10*100:.1f}%)")

print(f"\n{'=' * 70}")
print(f"CYCLIC TEMPERATURE COMPLETE")
print(f"Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"{'=' * 70}")
