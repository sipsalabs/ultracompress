"""
8B REAL TEXT DISTILLATION — Streaming teacher, proper training.

Priority #3 from STATUS.md: "Scale to 8B teacher — bigger models compress better."

Key improvements over run_8b_frr.py:
  - Real text from FineWeb-Edu (not random tokens)
  - Checkpoints every 5K steps
  - T1/T10 eval every 2.5K steps
  - HellaSwag at 25K and final step
  - VRAM-conscious: streaming teacher loads one layer at a time
  - Configurable device (default cuda:0, launch when GPU is free)
  - Temperature annealing
  - Longer training (50K steps by default)

VRAM estimate (fp32 streaming):
  - Embed + LM head constant: ~5 GB
  - One teacher layer peak: ~1.9 GB (freed after each layer)
  - FRR student (4096 hidden): ~0.4 GB
  - Optimizer states: ~1.2 GB
  - Activations (batch=2x64): ~0.5 GB
  - Peak total: ~9 GB (fits alongside other training on 32GB GPU)

Usage:
  python run_8b_real_text.py                    # cuda:0, 50K steps
  python run_8b_real_text.py --device cuda:1    # specific GPU
  python run_8b_real_text.py --steps 100000     # longer training
"""
import lib.unbuffered
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer
from datasets import load_dataset

# ── Configuration ─────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-len", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

DEVICE = args.device
STEPS = args.steps
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
LR = args.lr
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 5_000
HELLASWAG_STEPS = [25_000, STEPS - 1]
CHECKPOINT_DIR = 'checkpoints_8b_real_text'

# 8B model config
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
HIDDEN_DIM = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = HIDDEN_DIM // N_HEADS  # 128
INTERMEDIATE = 12288
N_LAYERS = 36
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000
NORM_EPS = 1e-6

# FRR config for 8B
FRR_N_SCALES = 4
FRR_ITERS_PER_SCALE = 9  # 4*9=36 virtual layers matching teacher
FRR_FF_MULT = 2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("8B REAL TEXT DISTILLATION — Streaming teacher + FRR student")
print(f"Device: {DEVICE}")
print(f"Steps: {STEPS:,}  Batch: {BATCH_SIZE}x{SEQ_LEN}  LR: {LR}")
print("=" * 70)

# ── Verify 8B model exists ───────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Qwen3-8B not found at {MODEL_PATH}")
    print("Download with: huggingface-cli download Qwen/Qwen3-8B")
    sys.exit(1)


# ── Weight index ──────────────────────────────────────────────────────
def load_weight_index() -> dict:
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


WEIGHT_INDEX = load_weight_index()


# ── RoPE ──────────────────────────────────────────────────────────────
def rope_embed(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, d, 2, device=x.device).float() / d))
    angles = torch.outer(positions.float(), freqs)
    cos_a = angles.cos().unsqueeze(0).unsqueeze(0)
    sin_a = angles.sin().unsqueeze(0).unsqueeze(0)
    x_r, x_i = x[..., 0::2], x[..., 1::2]
    out_r = x_r * cos_a - x_i * sin_a
    out_i = x_r * sin_a + x_i * cos_a
    return torch.stack([out_r, out_i], dim=-1).reshape(x.shape)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


# ── Teacher layer forward ────────────────────────────────────────────
def teacher_layer_forward(
    x: torch.Tensor, positions: torch.Tensor, weights: dict
) -> torch.Tensor:
    B, T, C = x.shape

    h = rms_norm(x, weights["input_layernorm.weight"])
    q = F.linear(h, weights["self_attn.q_proj.weight"])
    k = F.linear(h, weights["self_attn.k_proj.weight"])
    v = F.linear(h, weights["self_attn.v_proj.weight"])

    q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    if "self_attn.q_norm.weight" in weights:
        q = rms_norm(q, weights["self_attn.q_norm.weight"])
    if "self_attn.k_norm.weight" in weights:
        k = rms_norm(k, weights["self_attn.k_norm.weight"])

    q = rope_embed(q, positions)
    k = rope_embed(k, positions)

    if N_KV_HEADS < N_HEADS:
        repeat = N_HEADS // N_KV_HEADS
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    if T > 1:
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())

    out = out.transpose(1, 2).reshape(B, T, C)
    x = x + F.linear(out, weights["self_attn.o_proj.weight"])

    h = rms_norm(x, weights["post_attention_layernorm.weight"])
    gate = F.linear(h, weights["mlp.gate_proj.weight"])
    up = F.linear(h, weights["mlp.up_proj.weight"])
    x = x + F.linear(F.silu(gate) * up, weights["mlp.down_proj.weight"])

    return x


# ── Streaming weight loading ─────────────────────────────────────────
def load_layer_weights(layer_idx: int, target_device: str) -> dict:
    prefix = f"model.layers.{layer_idx}."
    suffixes = [
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    shard_to_keys: dict[str, list] = {}
    for suffix in suffixes:
        full_name = prefix + suffix
        if full_name in WEIGHT_INDEX:
            shard_file = WEIGHT_INDEX[full_name]
            shard_to_keys.setdefault(shard_file, []).append((full_name, suffix))

    weights = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
            for full_name, suffix in keys:
                weights[suffix] = f.get_tensor(full_name).float()
    return weights


def load_special_weights(target_device: str) -> dict:
    needed = {
        "model.embed_tokens.weight": "embed",
        "model.norm.weight": "norm",
        "lm_head.weight": "lm_head",
    }
    shard_to_keys: dict[str, list] = {}
    for full_name, short in needed.items():
        shard_file = WEIGHT_INDEX[full_name]
        shard_to_keys.setdefault(shard_file, []).append((full_name, short))

    special = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
            for full_name, short in keys:
                special[short] = f.get_tensor(full_name).float()
    return special


@torch.no_grad()
def teacher_forward(tokens: torch.Tensor, special_weights: dict, positions: torch.Tensor):
    x = F.embedding(tokens, special_weights["embed"]).float()
    for layer_idx in range(N_LAYERS):
        layer_w = load_layer_weights(layer_idx, DEVICE)
        x = teacher_layer_forward(x, positions, layer_w)
        del layer_w
        torch.cuda.empty_cache()
    x = rms_norm(x, special_weights["norm"])
    return F.linear(x, special_weights["lm_head"])


# ── Data ──────────────────────────────────────────────────────────────
print("Loading tokenizer and FineWeb-Edu...")
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


# ── Evaluation ────────────────────────────────────────────────────────
def eval_vs_teacher(
    model: nn.Module, special_weights: dict, positions: torch.Tensor, n: int = 50
) -> tuple[float, float]:
    """T1/T10 agreement on real text. Uses small n to save time (8B teacher is slow)."""
    t1_correct, t10_scores = 0, []
    model.eval()
    for _ in range(n):
        batch = get_real_batch(1, 32)
        pos = torch.arange(32, device=batch.device)
        with torch.no_grad():
            tl = teacher_forward(batch, special_weights, pos)
            sl = model(batch)
            t_top = tl[0, -1].argmax().item()
            s_top = sl[0, -1].argmax().item()
            t_top10 = set(tl[0, -1].topk(10).indices.tolist())
            s_top10 = set(sl[0, -1].topk(10).indices.tolist())
            if t_top == s_top:
                t1_correct += 1
            t10_scores.append(len(t_top10 & s_top10) / 10)
        torch.cuda.empty_cache()
    model.train()
    return t1_correct / n, sum(t10_scores) / len(t10_scores)


def eval_hellaswag(model_fn, n_samples: int = 200) -> float | None:
    """HellaSwag accuracy. Uses 200 samples (less than 300 for speed with 8B)."""
    try:
        ds_hs = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as e:
        print(f"  Could not load HellaSwag: {e}")
        return None

    correct, total = 0, 0
    t0 = time.time()

    with torch.no_grad():
        for i, sample in enumerate(ds_hs):
            if i >= n_samples:
                break
            ctx = sample['ctx']
            endings = sample['endings']
            label = int(sample['label'])

            best_score, best_idx = float('-inf'), 0
            for j, ending in enumerate(endings):
                text = ctx + " " + ending
                tokens = tokenizer.encode(
                    text, max_length=128, truncation=True, return_tensors='pt',
                ).to(DEVICE)
                if tokens.shape[1] < 2:
                    continue
                ctx_len = len(tokenizer.encode(ctx, max_length=128, truncation=True))
                if ctx_len >= tokens.shape[1] - 1:
                    continue

                logits = model_fn(tokens)
                log_probs = F.log_softmax(logits[0, ctx_len - 1:-1], dim=-1)
                targets = tokens[0, ctx_len:]
                score = log_probs.gather(1, targets.unsqueeze(1)).mean().item()
                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx == label:
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                print(f"    HellaSwag: {i + 1}/{n_samples}  "
                      f"acc={correct / total * 100:.1f}%  ({time.time() - t0:.0f}s)")

    return correct / total if total > 0 else None


# ── Load teacher special weights ──────────────────────────────────────
print("\nLoading 8B teacher embedding + LM head...")
special_weights = load_special_weights(DEVICE)
embed_mb = special_weights["embed"].numel() * 4 / 1e6
head_mb = special_weights["lm_head"].numel() * 4 / 1e6
print(f"  embed: {special_weights['embed'].shape} ({embed_mb:.0f} MB)")
print(f"  lm_head: {special_weights['lm_head'].shape} ({head_mb:.0f} MB)")
print(f"  norm: {special_weights['norm'].shape}")
if torch.cuda.is_available():
    print(f"  GPU used: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")

# ── Build FRR Student ─────────────────────────────────────────────────
print("\nBuilding FRR student (8B scale)...")
model = FractalModel(
    HIDDEN_DIM, N_HEADS, FRR_N_SCALES, FRR_ITERS_PER_SCALE,
    VOCAB_SIZE, FRR_FF_MULT,
    embed_weight=special_weights["embed"].clone(),
    lm_head_weight=special_weights["lm_head"].clone(),
    norm_weight=special_weights["norm"].clone(),
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
teacher_params = N_LAYERS * (4 * HIDDEN_DIM * HIDDEN_DIM + 3 * HIDDEN_DIM * INTERMEDIATE)
compression = teacher_params / trainable
print(f"FRR: {trainable:,} trainable params ({compression:.1f}x compression)")
print(f"Teacher: ~{teacher_params:,} layer params")
if torch.cuda.is_available():
    print(f"  GPU used after student: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")

# ── Training ──────────────────────────────────────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
positions = torch.arange(SEQ_LEN, device=DEVICE)

t0 = time.time()
best_t10 = 0.0
best_step = 0
loss_history = []

print(f"\nTraining {STEPS:,} steps with REAL TEXT distillation (streaming 8B teacher)...")
print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL}")
print(f"  Eval interval: {EVAL_INTERVAL}")
print(f"  HellaSwag at: {HELLASWAG_STEPS}")
print()

for step in range(STEPS):
    tokens = get_real_batch()

    # Streaming teacher forward (loads/frees each layer)
    teacher_logits = teacher_forward(tokens, special_weights, positions)
    gc.collect()
    torch.cuda.empty_cache()

    # Student forward + loss
    student_logits = model(tokens)
    T_val = max(2.0, 5.0 * (1 - step / STEPS))
    loss = F.kl_div(
        F.log_softmax(student_logits / T_val, dim=-1),
        F.softmax(teacher_logits.detach() / T_val, dim=-1),
        reduction='batchmean',
    ) * T_val * T_val

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    loss_history.append(loss.item())
    del teacher_logits, student_logits
    gc.collect()
    torch.cuda.empty_cache()

    # ── T1/T10 Evaluation ─────────────────────────────────────────
    if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
        t1, t10 = eval_vs_teacher(model, special_weights, positions, n=50)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        print(
            f"  Step {step:>6d}/{STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  "
            f"T={T_val:.1f}  ({elapsed:.0f}s){new_best}"
        )
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(DEVICE) / 1e9
            print(f"    GPU memory: {mem:.2f} GB")

    # ── Save Checkpoint ───────────────────────────────────────────
    if step > 0 and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS - 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_8b_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")
        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_8b_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10 * 100:.1f}%)")

    # ── HellaSwag Evaluation ──────────────────────────────────────
    if step in HELLASWAG_STEPS:
        print(f"\n  --- HellaSwag evaluation at step {step} ---")
        model.eval()

        # Teacher HellaSwag (through streaming)
        if step == HELLASWAG_STEPS[0]:
            print("  Computing teacher HellaSwag (200 samples)...")

            def teacher_hs_fn(t):
                pos = torch.arange(t.shape[1], device=DEVICE)
                return teacher_forward(t, special_weights, pos)

            teacher_hs = eval_hellaswag(teacher_hs_fn, n_samples=200)
            if teacher_hs is not None:
                print(f"  Teacher HellaSwag: {teacher_hs * 100:.1f}%")

        # FRR HellaSwag
        print(f"  Computing FRR HellaSwag (200 samples)...")
        frr_hs = eval_hellaswag(lambda t: model(t), n_samples=200)
        if frr_hs is not None:
            print(f"  FRR HellaSwag: {frr_hs * 100:.1f}%")
            if teacher_hs is not None and teacher_hs > 0:
                retention = frr_hs / teacher_hs * 100
                print(f"  HellaSwag retention: {retention:.1f}%")

        model.train()
        print()

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
t1_final, t10_final = eval_vs_teacher(model, special_weights, positions, n=100)

final_path = os.path.join(CHECKPOINT_DIR, 'frr_8b_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'=' * 70}")
print(f"FINAL RESULTS — 8B Real Text {STEPS // 1000}K")
print(f"{'=' * 70}")
print(f"  T1: {t1_final * 100:.1f}%  T10: {t10_final * 100:.1f}%")
print(f"  Compression: {compression:.1f}x")
print(f"  Best T10: {best_t10 * 100:.1f}% (step {best_step})")
print(f"  Training time: {total_time / 3600:.1f} hours")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}/")
print(f"\n  Previous records (1.7B):")
print(f"    T1=46% (50K), T10=67% (random 100K)")
print(f"    HellaSwag retention: 83.3% (0.6B 100K)")
print(f"\n  HYPOTHESIS: 8B teacher should produce BETTER student")
print(f"  because bigger models have more redundancy (smoother manifold)")
print("Done!")
