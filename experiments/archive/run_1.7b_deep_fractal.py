"""
1.7B DEEP FRACTAL CONDITIONING (DFC) — Novel internal block modulation.

THE PROBLEM with standard FRR:
  One shared block uses identical Q/K/V weights for all 28 virtual layers.
  Gamma/beta only SCALE the input — this can't produce qualitatively different
  attention patterns. But real transformers need early layers doing local/syntactic
  attention and late layers doing global/semantic attention. One set of attention
  weights fundamentally cannot do both.

THE NOVEL INSIGHT:
  Instead of modulating INPUTS (gamma/beta) or OUTPUTS (LoRA), we modulate
  INSIDE the shared block's computation:

  1. Per-layer attention temperature — controls attention sharpness per head.
     Early layers can have sharp (local) attention, late layers soft (global).
  2. Per-layer head gating — which attention heads matter at each depth.
     Effectively gives each layer a different "attention profile."
  3. Per-layer FFN neuron gating — which feature detectors activate per layer.
     Early layers extract syntax features, late layers extract semantic features.

  Total new params: ~58K (0.2% overhead). Same 52x compression ratio.
  But each virtual layer now computes a QUALITATIVELY different function.

WHY THIS IS DIFFERENT FROM LORA:
  - LoRA adds a residual to the OUTPUT → can only patch errors post-hoc
  - DFC changes HOW the block COMPUTES → fundamentally different attention
    patterns and feature extraction per layer
  - LoRA is 1.8M params of output patching. DFC is 58K params of computation steering.
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
DEVICE = 'cuda:1'
TOTAL_STEPS = 100_000
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 3e-4
TEMP = 2.0
EVAL_INTERVAL = 2_500
CHECKPOINT_INTERVAL = 10_000
CHECKPOINT_DIR = 'checkpoints_1.7b_dfc'
RESUME_FROM = 'checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B DEEP FRACTAL CONDITIONING — Novel internal block modulation")
print(f"Device: {DEVICE}  |  Temp: {TEMP}  |  Steps: {TOTAL_STEPS:,}")
print(f"Novel: per-layer attention temps + head gates + FFN gates")
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


# ── Build FRR Student with Deep Conditioning ──────────────────────────
print("\nBuilding FRR model with Deep Fractal Conditioning...")
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    deep_conditioning=True,  # THE NOVEL PART
).to(DEVICE)

# Load pre-trained weights (DFC params are new, won't be in checkpoint)
print(f"Loading base checkpoint: {RESUME_FROM}")
ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
# strict=False: DFC params (head_log_temps, head_gate_logits, ffn_gate_logits) are new
model.load_state_dict(ckpt, strict=False)
del ckpt

# Verify baseline quality
print("Verifying baseline (DFC params start as near-identity)...")
t1_base, t10_base = eval_vs_teacher(model, n=50)
print(f"  Baseline: T1={t1_base*100:.1f}%, T10={t10_base*100:.1f}%")

# Count params
block_params = sum(p.numel() for p in model.block.parameters())
dfc_params = (model.head_log_temps.numel() +
              model.head_gate_logits.numel() +
              model.ffn_gate_logits.numel())
mod_params = model.scale_gamma.numel() + model.scale_beta.numel() + model.iter_scale.numel()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
teacher_params = N_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)

print(f"\n  Shared block: {block_params:,} params")
print(f"  Scale modulation: {mod_params:,} params (input gamma/beta)")
print(f"  DFC conditioning: {dfc_params:,} params (NOVEL: internal modulation)")
print(f"    - head_temps: {model.head_log_temps.numel():,} (attention sharpness per head)")
print(f"    - head_gates: {model.head_gate_logits.numel():,} (which heads matter)")
print(f"    - ffn_gates: {model.ffn_gate_logits.numel():,} (which FFN neurons fire)")
print(f"  Total trainable: {trainable:,}")
print(f"  Compression: {teacher_params/trainable:.1f}x")

# ── Training ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Training {TOTAL_STEPS:,} steps — all params jointly")
print(f"  LR={LR}, cosine decay, weight_decay=0.01")
print(f"{'='*70}")

# Freeze embed/head/norm (teacher weights)
for name, param in model.named_parameters():
    if 'embed' in name or 'lm_head' in name or 'norm' == name.split('.')[-1]:
        param.requires_grad = False
# Re-enable norm if it's the block norms (not the output norm)
for name, param in model.named_parameters():
    if 'block.norm' in name:
        param.requires_grad = True
# Explicitly freeze output norm
model.norm.weight.requires_grad = False

all_params = [p for p in model.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in all_params)
print(f"  Trainable params: {trainable_count:,}")

opt = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)
t0 = time.time()
best_t10 = t10_base
best_step = -1
loss_history = []

for step in range(TOTAL_STEPS):
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
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    opt.step()
    sched.step()

    loss_history.append(loss.item())

    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=100)
        elapsed = time.time() - t0
        new_best = ""
        if t10 > best_t10:
            best_t10 = t10
            best_step = step
            new_best = " *** NEW BEST ***"
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)

        # Show DFC stats — how much has conditioning diverged from identity
        with torch.no_grad():
            temps = model.head_log_temps.exp()
            hg = torch.sigmoid(model.head_gate_logits)
            fg = torch.sigmoid(model.ffn_gate_logits)
            temp_range = f"[{temps.min():.3f}, {temps.max():.3f}]"
            hg_range = f"[{hg.min():.3f}, {hg.max():.3f}]"
            fg_range = f"[{fg.min():.3f}, {fg.max():.3f}]"

        print(
            f"  Step {step:>6d}/{TOTAL_STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1*100:.1f}%  T10={t10*100:.1f}%  "
            f"LR={sched.get_last_lr()[0]:.6f}  ({elapsed:.0f}s){new_best}"
        )
        print(
            f"    DFC: temps={temp_range}  heads={hg_range}  ffn={fg_range}"
        )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_dfc_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved: {ckpt_path}")

# ── Final Save ────────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_dfc_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"  Final: T1={t1*100:.1f}%, T10={t10*100:.1f}%")
print(f"  Best T10: {best_t10*100:.1f}% at step {best_step}")
print(f"  Total time: {time.time()-t0:.0f}s")
print(f"  Saved: {final_path}")
print(f"{'='*70}")

# Print DFC analysis — what did each layer learn?
print(f"\nDFC LAYER ANALYSIS (what each virtual layer specialized to):")
print(f"{'Layer':>6} | {'Attention Temps (per head)':^50} | {'Active Heads':>12} | {'FFN Active%':>11}")
print("-" * 90)
with torch.no_grad():
    for l in range(model.total_layers):
        temps = model.head_log_temps[l].exp()
        hg = torch.sigmoid(model.head_gate_logits[l])
        fg = torch.sigmoid(model.ffn_gate_logits[l])
        active_heads = (hg > 0.5).sum().item()
        ffn_active = (fg > 0.5).float().mean().item() * 100
        temp_str = ' '.join(f'{t:.2f}' for t in temps)
        print(f"  {l:>4d} | {temp_str:^50} | {active_heads:>5d}/16    | {ffn_active:>8.1f}%")
