"""
Distill Qwen3-8B into a Fractal Residual Recursion (FRR) model.

Teacher: Qwen3-8B (36 layers, 4096 hidden, 32 heads, 8 kv_heads)
Student: FRR (1 shared block, 4 scales x 9 iters = 36 virtual layers)

Streaming inference: loads teacher shards one at a time, never holds
the full 8B in memory. Peak GPU usage ~11GB (embed+head+1 shard+activations+FRR).

Training: 5000 steps, batch_size=2, seq_len=32
Loss: all-position KL divergence on logits
"""

import os
import sys
import gc
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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

# FRR config
FRR_N_SCALES = 4
FRR_ITERS_PER_SCALE = 9  # 4*9 = 36 virtual layers
FRR_FF_MULT = 2

# Training config
BATCH_SIZE = 2
SEQ_LEN = 32
LR = 3e-4
N_STEPS = 5000
EVAL_EVERY = 1000
SAVE_PATH = "checkpoints/frr_8b_distill.pt"

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected, running on CPU (will be very slow)")


# ---------------------------------------------------------------------------
# Weight index: maps each weight name -> shard file
# ---------------------------------------------------------------------------
def load_weight_index():
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]

WEIGHT_INDEX = load_weight_index()


def get_shard_for_weight(weight_name):
    """Return full path to shard containing this weight."""
    shard_file = WEIGHT_INDEX[weight_name]
    return os.path.join(MODEL_PATH, shard_file)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def rope_embed(x, positions):
    """Apply rotary position embeddings. x: (B, n_heads, T, head_dim)"""
    d = x.shape[-1]
    n_pairs = d // 2
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, d, 2, device=x.device).float() / d))
    angles = torch.outer(positions.float(), freqs)  # (T, n_pairs)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    # Broadcast: (1, 1, T, n_pairs)
    cos_a = cos_a.unsqueeze(0).unsqueeze(0)
    sin_a = sin_a.unsqueeze(0).unsqueeze(0)
    x_r = x[..., 0::2]
    x_i = x[..., 1::2]
    out_r = x_r * cos_a - x_i * sin_a
    out_i = x_r * sin_a + x_i * cos_a
    return torch.stack([out_r, out_i], dim=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# RMSNorm (functional)
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=NORM_EPS):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Teacher layer forward (manual, from raw weight tensors)
# ---------------------------------------------------------------------------
def teacher_layer_forward(x, positions, weights_dict):
    """
    Run one Qwen3-8B transformer layer.
    x: (B, T, hidden_dim) float32 on device
    positions: (T,) on device
    weights_dict: dict of weight_name_suffix -> tensor on device
    Returns: x after this layer
    """
    B, T, C = x.shape

    # Pre-attention norm
    h = rms_norm(x, weights_dict["input_layernorm.weight"])

    # QKV projections
    q = F.linear(h, weights_dict["self_attn.q_proj.weight"])  # (B, T, n_heads * head_dim)
    k = F.linear(h, weights_dict["self_attn.k_proj.weight"])  # (B, T, n_kv_heads * head_dim)
    v = F.linear(h, weights_dict["self_attn.v_proj.weight"])  # (B, T, n_kv_heads * head_dim)

    q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)      # (B, nh, T, hd)
    k = k.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)   # (B, nkv, T, hd)
    v = v.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)   # (B, nkv, T, hd)

    # Q/K norms (Qwen3 has these)
    if "self_attn.q_norm.weight" in weights_dict:
        q = rms_norm(q, weights_dict["self_attn.q_norm.weight"])
    if "self_attn.k_norm.weight" in weights_dict:
        k = rms_norm(k, weights_dict["self_attn.k_norm.weight"])

    # RoPE
    q = rope_embed(q, positions)
    k = rope_embed(k, positions)

    # GQA: repeat KV heads
    if N_KV_HEADS < N_HEADS:
        repeat = N_HEADS // N_KV_HEADS
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # Scaled dot-product attention
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    if T > 1:
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())

    out = out.transpose(1, 2).reshape(B, T, C)
    attn_out = F.linear(out, weights_dict["self_attn.o_proj.weight"])
    x = x + attn_out

    # Post-attention norm + FFN (SwiGLU)
    h = rms_norm(x, weights_dict["post_attention_layernorm.weight"])
    gate = F.linear(h, weights_dict["mlp.gate_proj.weight"])
    up = F.linear(h, weights_dict["mlp.up_proj.weight"])
    ffn_out = F.linear(F.silu(gate) * up, weights_dict["mlp.down_proj.weight"])
    x = x + ffn_out

    return x


# ---------------------------------------------------------------------------
# Load layer weights from shards using weight index (efficient: opens only
# the shards that actually contain this layer's weights)
# ---------------------------------------------------------------------------
def load_layer_weights(layer_idx, target_device):
    """Load all weights for a single transformer layer onto target_device."""
    prefix = f"model.layers.{layer_idx}."
    suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]

    # Group weights by shard to minimize file opens
    shard_to_keys = {}
    for suffix in suffixes:
        full_name = prefix + suffix
        if full_name in WEIGHT_INDEX:
            shard_file = WEIGHT_INDEX[full_name]
            if shard_file not in shard_to_keys:
                shard_to_keys[shard_file] = []
            shard_to_keys[shard_file].append((full_name, suffix))

    weights = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
            for full_name, suffix in keys:
                weights[suffix] = f.get_tensor(full_name).float()

    return weights


# ---------------------------------------------------------------------------
# Load special weights (embedding, final norm, LM head)
# ---------------------------------------------------------------------------
def load_special_weights(target_device):
    """Load embed_tokens, model.norm, lm_head onto target_device."""
    special = {}
    needed = {
        "model.embed_tokens.weight": "embed",
        "model.norm.weight": "norm",
        "lm_head.weight": "lm_head",
    }
    # Group by shard
    shard_to_keys = {}
    for full_name, short in needed.items():
        shard_file = WEIGHT_INDEX[full_name]
        if shard_file not in shard_to_keys:
            shard_to_keys[shard_file] = []
        shard_to_keys[shard_file].append((full_name, short))

    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
            for full_name, short in keys:
                special[short] = f.get_tensor(full_name).float()

    return special


# ---------------------------------------------------------------------------
# Full teacher forward pass (streaming, layer by layer)
# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_forward(tokens, special_weights, positions):
    """
    Full Qwen3-8B forward pass via streaming.
    tokens: (B, T) on device
    Returns: logits (B, T, vocab_size) on device
    """
    x = F.embedding(tokens, special_weights["embed"]).float()

    for layer_idx in range(N_LAYERS):
        # Load this layer's weights
        layer_w = load_layer_weights(layer_idx, device)
        x = teacher_layer_forward(x, positions, layer_w)
        # Free layer weights
        del layer_w
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final norm + LM head
    x = rms_norm(x, special_weights["norm"])
    logits = F.linear(x, special_weights["lm_head"])
    return logits


# ---------------------------------------------------------------------------
# FRR Student Model (from moonshot.py, imported inline for self-containment)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.moonshot import FractalModel


# ---------------------------------------------------------------------------
# Build FRR student
# ---------------------------------------------------------------------------
def build_student(special_weights):
    """Build the FRR student model, sharing embed/head from teacher."""
    student = FractalModel(
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_scales=FRR_N_SCALES,
        iters_per_scale=FRR_ITERS_PER_SCALE,
        vocab_size=VOCAB_SIZE,
        ff_mult=FRR_FF_MULT,
        embed_weight=special_weights["embed"].clone(),
        lm_head_weight=special_weights["lm_head"].clone(),
        norm_weight=special_weights["norm"].clone(),
    )
    student = student.to(device)
    student.enable_adapters(rank=16)

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    fractal_params = student.fractal_params()
    print(f"FRR student total params:     {total_params:>12,}")
    print(f"FRR student trainable params: {trainable_params:>12,}")
    print(f"FRR student fractal params:   {fractal_params:>12,}")
    print(f"FRR fractal size:             {fractal_params * 4 / 1e6:.1f} MB (fp32)")
    print(f"Teacher size:                 ~16,000 MB (bf16)")
    print(f"Compression ratio:            ~{16_000_000_000 / (fractal_params * 4):.0f}x (fractal only)")

    return student


# ---------------------------------------------------------------------------
# KL divergence loss (all positions)
# ---------------------------------------------------------------------------
def kl_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL divergence between student and teacher logit distributions.
    Computed over ALL positions (not just last token).
    student_logits, teacher_logits: (B, T, vocab_size)
    """
    # Scale by temperature
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(teacher || student) summed over vocab, averaged over B*T
    loss = F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(student, special_weights, positions, n_eval=4):
    """Run a few eval batches and report metrics."""
    student.eval()
    total_kl = 0.0
    total_topk = {1: 0, 5: 0, 10: 0}
    n_tokens = 0

    for i in range(n_eval):
        torch.manual_seed(999999 + i)
        tokens = torch.randint(100, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        teacher_logits = teacher_forward(tokens, special_weights, positions)
        student_logits = student(tokens)

        # KL
        total_kl += kl_loss(student_logits, teacher_logits, temperature=1.0).item()

        # Top-k accuracy (does student's top-1 match teacher's top-k?)
        teacher_top = teacher_logits.argmax(dim=-1)  # (B, T)
        for k in [1, 5, 10]:
            student_topk = student_logits.topk(k, dim=-1).indices  # (B, T, k)
            match = (student_topk == teacher_top.unsqueeze(-1)).any(dim=-1)  # (B, T)
            total_topk[k] += match.float().sum().item()
        n_tokens += tokens.numel()

    avg_kl = total_kl / n_eval
    student.train()
    return {
        "kl": avg_kl,
        "top1": total_topk[1] / n_tokens * 100,
        "top5": total_topk[5] / n_tokens * 100,
        "top10": total_topk[10] / n_tokens * 100,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FRR Distillation: Qwen3-8B -> Fractal Residual Recursion")
    print("=" * 70)
    print(f"Teacher: Qwen3-8B ({N_LAYERS} layers, {HIDDEN_DIM} hidden)")
    print(f"Student: FRR ({FRR_N_SCALES} scales x {FRR_ITERS_PER_SCALE} iters = "
          f"{FRR_N_SCALES * FRR_ITERS_PER_SCALE} virtual layers)")
    print(f"Batch: {BATCH_SIZE} x {SEQ_LEN} tokens")
    print(f"Steps: {N_STEPS}, LR: {LR}")
    print()

    # --- Load teacher embedding + LM head (stay in GPU memory) ---
    print("Loading teacher embedding + LM head + final norm...")
    special_weights = load_special_weights(device)
    embed_mb = special_weights["embed"].numel() * 4 / 1e6
    head_mb = special_weights["lm_head"].numel() * 4 / 1e6
    print(f"  embed: {special_weights['embed'].shape} ({embed_mb:.0f} MB)")
    print(f"  lm_head: {special_weights['lm_head'].shape} ({head_mb:.0f} MB)")
    print(f"  norm: {special_weights['norm'].shape}")
    if device.type == "cuda":
        print(f"  GPU used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # --- Build FRR student ---
    print("Building FRR student model...")
    student = build_student(special_weights)
    if device.type == "cuda":
        print(f"  GPU used after student: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # --- Optimizer (only train fractal params, not frozen embed/head) ---
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

    # Shared positions tensor
    positions = torch.arange(SEQ_LEN, device=device)

    # --- Checkpoint dir ---
    os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else "checkpoints", exist_ok=True)

    # --- Training ---
    print("Starting training...")
    print("-" * 70)
    start_time = time.time()
    running_loss = 0.0

    for step in range(1, N_STEPS + 1):
        # Generate random tokens
        torch.manual_seed(step)
        tokens = torch.randint(100, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

        # --- Teacher forward (streaming, no grad) ---
        teacher_logits = teacher_forward(tokens, special_weights, positions)

        # Memory cleanup after teacher streaming
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Student forward ---
        student_logits = student(tokens)

        # --- Loss ---
        loss = kl_loss(student_logits, teacher_logits.detach(), temperature=2.0)

        # --- Backprop (student only) ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # --- Progress ---
        if step % 100 == 0:
            avg = running_loss / 100
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (N_STEPS - step) / steps_per_sec if steps_per_sec > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            mem_str = ""
            if device.type == "cuda":
                mem_str = f" | GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB"
            print(f"Step {step:5d}/{N_STEPS} | loss: {avg:.4f} | "
                  f"lr: {lr_now:.2e} | {steps_per_sec:.2f} step/s | "
                  f"ETA: {eta/60:.0f}min{mem_str}")
            running_loss = 0.0

        # --- Evaluation ---
        if step % EVAL_EVERY == 0 or step == 1:
            if step == 1:
                print(f"\n--- Step {step} eval (baseline before training) ---")
            else:
                print(f"\n--- Step {step} eval ---")
            metrics = evaluate(student, special_weights, positions)
            print(f"  KL divergence:  {metrics['kl']:.4f}")
            print(f"  Top-1 accuracy: {metrics['top1']:.1f}%")
            print(f"  Top-5 accuracy: {metrics['top5']:.1f}%")
            print(f"  Top-10 accuracy: {metrics['top10']:.1f}%")
            print()

            # Save checkpoint
            ckpt = {
                "step": step,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "n_heads": N_HEADS,
                    "n_scales": FRR_N_SCALES,
                    "iters_per_scale": FRR_ITERS_PER_SCALE,
                    "ff_mult": FRR_FF_MULT,
                    "vocab_size": VOCAB_SIZE,
                    "teacher": "Qwen3-8B",
                },
            }
            torch.save(ckpt, SAVE_PATH)
            print(f"  Saved checkpoint to {SAVE_PATH}")
            print()

    # --- Final save ---
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training complete! {N_STEPS} steps in {total_time/60:.1f} minutes")
    print(f"Final checkpoint: {SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
