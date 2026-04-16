"""
1.7B EXTENDED TRAINING — Continue from 100K checkpoint for 200K more steps.

Hires evaluation showed T10 is still improving at 100K (60.7% → 63.9% monotonic).
This run continues training to push T10 higher.

Key changes from 100K run:
  - Loads from 100K final checkpoint
  - Lower peak LR (2e-4 vs 5e-4) — fine-tuning, not training from scratch
  - Fixed temperature T=2.0 (already optimal at convergence)
  - 200K additional steps (steps 100K-300K effectively)
  - Saves to checkpoints_1.7b_extended/
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
DEVICE = 'cuda:1'
STEPS = 200_000
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 2e-4  # Lower than original 5e-4 — fine-tuning from good init
TEMP = 2.0  # Fixed at optimal temperature
EVAL_INTERVAL = 5_000
CHECKPOINT_INTERVAL = 10_000
HELLASWAG_STEPS = [49_999, 99_999, 149_999, 199_999]
CHECKPOINT_DIR = 'checkpoints_1.7b_extended'
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B EXTENDED TRAINING — 200K more steps from 100K checkpoint")
print(f"Device: {DEVICE}  |  LR: {LR}  |  Temp: {TEMP} (fixed)")
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

hidden = gd['token_embd.weight'].shape[1]  # 2048
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
    batch = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
            except StopIteration:
                ds_iter = iter(ds)
                sample = next(ds_iter)
            toks = tokenizer.encode(sample['text'], add_special_tokens=False)
            if len(toks) >= seq_len:
                start = torch.randint(0, len(toks) - seq_len + 1, (1,)).item()
                batch.append(torch.tensor(toks[start:start + seq_len]))
                break
    return torch.stack(batch).to(DEVICE)


def eval_vs_teacher(model: nn.Module, n: int = 100) -> tuple[float, float]:
    model.eval()
    t1_hits, t10_hits = 0, 0
    for _ in range(n):
        tokens = get_real_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=N_LAYERS)
            sl = model(tokens)
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t1_hits += int(s_top[0] == t_top[0])
        t10_hits += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
    model.train()
    return t1_hits / n, t10_hits / n


def eval_hellaswag(model_fn, n_samples: int = 300) -> float | None:
    try:
        from datasets import load_dataset as ld
        hs = ld("Rowan/hellaswag", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"  HellaSwag load failed: {e}")
        return None

    correct = 0
    total = 0
    indices = torch.randperm(len(hs))[:n_samples]
    for idx in indices:
        ex = hs[int(idx)]
        ctx = ex.get('ctx', ex.get('ctx_a', ''))
        endings = ex.get('endings', [])
        label = int(ex.get('label', 0))
        if not endings:
            continue

        scores = []
        for end in endings:
            text = ctx + " " + end
            toks = tokenizer.encode(text, add_special_tokens=False)
            ctx_len = len(tokenizer.encode(ctx, add_special_tokens=False))
            if len(toks) > 256:
                toks = toks[:256]
            t = torch.tensor([toks], device=DEVICE)
            with torch.no_grad():
                logits = model_fn(t)
            log_probs = F.log_softmax(logits[0], dim=-1)
            score = 0.0
            for i in range(ctx_len - 1, t.shape[1] - 1):
                score += log_probs[i, toks[i + 1]].item()
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else None


# ── Build FRR Student ─────────────────────────────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

# Load from 100K checkpoint
print(f"Loading checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt, strict=False)
del ckpt
print("  Checkpoint loaded successfully")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_teacher = N_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
compression = total_teacher / trainable
print(f"FRR: {trainable:,} params ({compression:.1f}x compression)")

# ── Training ──────────────────────────────────────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()
best_t10 = 0.0
best_step = 0
loss_history = []

print(f"\nExtended training: {STEPS:,} steps, T={TEMP} fixed, LR={LR}")
print(f"  Checkpoints: every {CHECKPOINT_INTERVAL}")
print(f"  Eval: every {EVAL_INTERVAL}")
print(f"  HellaSwag at steps: {HELLASWAG_STEPS}")
print()

for step in range(STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    loss = F.kl_div(
        F.log_softmax(sl / TEMP, dim=-1),
        F.softmax(tl / TEMP, dim=-1),
        reduction='batchmean',
    ) * TEMP * TEMP

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    loss_history.append(loss.item())

    # ── T1/T10 Evaluation ─────────────────────────────────────────
    if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=100)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        # Effective step = 100K + current step
        eff_step = 100_000 + step
        print(
            f"  Step {step:>6d}/{STEPS} (eff {eff_step:,}): loss={avg_loss:.4f}  "
            f"T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  "
            f"LR={sched.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )

    # ── Save Checkpoint ───────────────────────────────────────────
    if step > 0 and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS - 1):
        eff_step = 100_000 + step
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_ext_step{eff_step}.pt')
        ckpt_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'step': step,
            'effective_step': eff_step,
            'best_t10': best_t10,
            'best_step': best_step,
            'loss_history': loss_history[-5000:],
        }
        torch.save(ckpt_data, ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")
        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_ext_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10 * 100:.1f}%)")

    # ── HellaSwag Evaluation ──────────────────────────────────────
    if step in HELLASWAG_STEPS:
        eff_step = 100_000 + step
        print(f"\n  --- HellaSwag + WikiText-2 at effective step {eff_step:,} ---")
        model.eval()

        # HellaSwag
        frr_hs = eval_hellaswag(lambda t: model(t), n_samples=300)
        if frr_hs is not None:
            print(f"  FRR HellaSwag: {frr_hs * 100:.1f}%")
            teacher_hs = 31.3  # Known from prior eval
            retention = frr_hs * 100 / teacher_hs
            print(f"  HellaSwag retention: {retention:.1f}% (teacher=31.3%)")

        model.train()
        print()

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
t1_final, t10_final = eval_vs_teacher(model, n=200)

final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_ext_300k_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'='*70}")
print(f"EXTENDED TRAINING COMPLETE")
print(f"  Total time: {total_time/3600:.1f} hours ({total_time:.0f}s)")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step} (eff {100_000+best_step:,})")
print(f"  Final (200-sample): T1={t1_final*100:.1f}%  T10={t10_final*100:.1f}%")
print(f"  Saved: {final_path}")
print(f"{'='*70}")
