"""
1.7B MULTI-TEMPERATURE KL — Simultaneous KL at multiple temperatures.

HYPOTHESIS: Single-temperature annealing (T: 5→2) causes T10 decline because
the student "forgets" broad distribution matching when T drops. Multi-temp KL
forces the student to match teacher distributions at ALL temperatures
simultaneously, preventing this forgetting.

Loss = Σ_T w(T) * KL(student/T || teacher/T) * T²

With T ∈ {1.0, 2.0, 4.0}:
  - T=1.0: sharp distribution → top-token precision (T1)
  - T=2.0: medium softness → balanced signal
  - T=4.0: soft distribution → broad coverage (T10)

The student must satisfy all three constraints simultaneously.
This is strictly stronger than single-temperature and prevents
the metric-objective misalignment seen in standard annealing.

Usage:
  python run_1.7b_multi_temp.py                       # Default: cuda:0, fresh
  python run_1.7b_multi_temp.py --device cuda:0        # Specify GPU
  python run_1.7b_multi_temp.py --resume checkpoints_1.7b_real_text/frr_1.7b_best.pt
"""
import lib.unbuffered
import torch
import sys
import os
import time
import math
import argparse

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Arguments ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--resume', default=None, help="Resume from checkpoint")
parser.add_argument('--steps', type=int, default=50_000)
cmd_args = parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────
DEVICE = cmd_args.device
STEPS = cmd_args.steps
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 3e-4 if cmd_args.resume is None else 2e-4  # Lower LR for resume
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_multi_temp'

# Multi-temperature config: (temperature, weight)
TEMP_WEIGHTS = [
    (1.0, 0.3),   # Sharp — T1 precision
    (2.0, 0.4),   # Medium — balanced
    (4.0, 0.3),   # Soft — T10 coverage
]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B MULTI-TEMPERATURE KL DISTILLATION")
print(f"Device: {DEVICE}")
print(f"Temperatures: {[(t, f'w={w}') for t, w in TEMP_WEIGHTS]}")
print(f"Resume: {cmd_args.resume or 'fresh start'}")
print(f"Steps: {STEPS:,}, LR: {LR}")
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
    """T1 and T10 agreement (200 samples)."""
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


def multi_temp_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Multi-temperature KL divergence.

    Computes weighted sum of KL at multiple temperatures.
    The student must match the teacher's distribution at ALL temperatures,
    preventing sharp/soft distribution forgetting.
    """
    total_loss = torch.tensor(0.0, device=student_logits.device)
    for T, w in TEMP_WEIGHTS:
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean',
        ) * T * T
        total_loss = total_loss + w * kl
    return total_loss


# ── Build FRR Student ─────────────────────────────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

if cmd_args.resume:
    print(f"Resuming from: {cmd_args.resume}")
    state = torch.load(cmd_args.resume, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    del state

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_teacher = N_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
compression = total_teacher / trainable
print(f"FRR: {trainable:,} params ({compression:.1f}x compression)")

# Initial eval
t1_init, t10_init = eval_vs_teacher(model, n=200)
print(f"Initial quality: T1={t1_init*100:.1f}% T10={t10_init*100:.1f}% (200 samples)")

# ── Training ──────────────────────────────────────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()
best_t10 = t10_init
best_step = -1

print(f"\nTraining {STEPS:,} steps with MULTI-TEMPERATURE KL...")
print(f"  Batch: {BATCH_SIZE}×{SEQ_LEN}, LR: {LR}")
print(f"  Temperatures: " + ", ".join(f"T={t}(w={w})" for t, w in TEMP_WEIGHTS))
print()

for step in range(STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    loss = multi_temp_kl_loss(sl, tl)

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
        print(
            f"  Step {step:>6d}/{STEPS}: loss={loss.item():.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"({elapsed:.0f}s){new_best}"
        )

    # ── Save Checkpoint ───────────────────────────────────────────
    if step > 0 and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS - 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_multitemp_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_multitemp_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10*100:.1f}%)")

print(f"\n{'=' * 70}")
print("MULTI-TEMPERATURE KL COMPLETE")
print(f"Best T10: {best_t10*100:.1f}% at step {best_step}")
print("=" * 70)
