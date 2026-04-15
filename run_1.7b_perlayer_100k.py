"""
1.7B PER-LAYER MODULATION EXPERIMENT — 100K steps.

A/B test: per-LAYER modulation (28 individual gamma/beta pairs) vs
per-SCALE modulation (4 groups of shared gamma/beta).

Hypothesis: Per-layer modulation gives each virtual layer its own steering
signal, potentially breaking through the ~67% T10 ceiling.

Parameter impact:
  Per-scale (current): 4*2*2048 + 28 = 16,412 modulation params
  Per-layer (this exp): 28*2*2048 + 28 = 114,716 modulation params
  Total FRR goes from 29.38M to ~29.48M (+0.3%) — negligible.

Baseline: run_1.7b_real_text_100k.py → 66.7% T10 at 80K (per-scale)
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
STEPS = 100_000
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 5e-4
EVAL_INTERVAL = 5_000
CHECKPOINT_INTERVAL = 10_000
HELLASWAG_STEPS = [50_000, 99_999]
CHECKPOINT_DIR = 'checkpoints_1.7b_perlayer'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B PER-LAYER MODULATION — 100K steps + HellaSwag eval")
print(f"Device: {DEVICE}")
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


# ── Evaluation Functions ──────────────────────────────────────────────
def eval_vs_teacher(model: nn.Module, n: int = 100) -> tuple[float, float]:
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


def eval_hellaswag(model_fn, n_samples: int = 300) -> float | None:
    try:
        ds_hs = load_dataset("Rowan/hellaswag", split="validation")
        samples = list(ds_hs)[:n_samples]
    except Exception as e:
        print(f"  Could not load HellaSwag: {e}")
        return None

    correct = 0
    total = 0

    with torch.no_grad():
        for sample in samples:
            ctx = sample['ctx']
            endings = sample['endings']
            label = int(sample['label'])

            scores = []
            for ending in endings:
                text = ctx + " " + ending
                tokens = tokenizer.encode(
                    text, max_length=128, truncation=True, return_tensors='pt',
                ).to(DEVICE)
                ctx_len = len(tokenizer.encode(ctx, max_length=128, truncation=True))

                logits = model_fn(tokens[:, :-1])
                log_probs = F.log_softmax(logits[0], dim=-1)

                score = 0.0
                for i in range(ctx_len - 1, tokens.shape[1] - 1):
                    if i < log_probs.shape[0]:
                        score += log_probs[i, tokens[0, i + 1]].item()
                scores.append(score)

            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == label:
                correct += 1
            total += 1

    return correct / total if total > 0 else None


def eval_wikitext_ppl(model_fn, n_samples: int = 100, seq_len: int = 128) -> float | None:
    try:
        ds_wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds_wt['text'] if len(t) > 100][:n_samples]
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        return None

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(
                text, max_length=seq_len + 1, truncation=True, return_tensors='pt',
            )
            if tokens.shape[1] < 10:
                continue
            tokens = tokens.to(DEVICE)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = model_fn(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ── Build FRR Student (PER-LAYER MODULATION) ─────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    per_layer_mod=True,  # KEY DIFFERENCE: per-layer instead of per-scale
).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frr_params = model.fractal_params()
total_teacher = N_LAYERS * (4 * hidden * hidden + 3 * hidden * hidden * 3)
compression = total_teacher / trainable
print(f"FRR (per-layer mod): {trainable:,} params ({compression:.1f}x compression)")
print(f"  FRR-specific: {frr_params:,} params")
print(f"  Modulation params: {model.layer_gamma.numel() + model.layer_beta.numel() + model.iter_scale.numel():,}")

# ── Training ──────────────────────────────────────────────────────────
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()
best_t10 = 0.0
best_step = 0
loss_history = []

print(f"\nTraining {STEPS:,} steps with REAL TEXT distillation (PER-LAYER MOD)...")
print(f"  Batch: {BATCH_SIZE}x{SEQ_LEN}, LR: {LR}, Checkpoints: every {CHECKPOINT_INTERVAL}")
print(f"  T1/T10 eval: every {EVAL_INTERVAL} steps")
print(f"  HellaSwag eval at steps: {HELLASWAG_STEPS}")
print()

for step in range(STEPS):
    tokens = get_real_batch()

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=N_LAYERS)

    sl = model(tokens)
    T = max(2.0, 5.0 * (1 - step / STEPS))
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
        avg_loss = sum(loss_history[-1000:]) / min(len(loss_history), 1000)
        print(
            f"  Step {step:>6d}/{STEPS}: loss={avg_loss:.4f}  "
            f"T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  "
            f"T={T:.1f}  ({elapsed:.0f}s){new_best}"
        )

    # ── Save Checkpoint ───────────────────────────────────────────
    if step > 0 and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS - 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_perlayer_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_perlayer_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10 * 100:.1f}%)")

    # ── HellaSwag + WikiText-2 Evaluation ─────────────────────────
    if step in HELLASWAG_STEPS:
        print(f"\n  --- HellaSwag + WikiText-2 evaluation at step {step} ---")
        model.eval()

        if step == HELLASWAG_STEPS[0]:
            print("  Computing teacher HellaSwag (300 samples)...")
            teacher_hs = eval_hellaswag(
                lambda t: teacher.forward(t, max_layers=N_LAYERS), n_samples=300,
            )
            if teacher_hs is not None:
                print(f"  Teacher HellaSwag: {teacher_hs * 100:.1f}%")

            print("  Computing teacher WikiText-2 PPL...")
            teacher_ppl = eval_wikitext_ppl(
                lambda t: teacher.forward(t, max_layers=N_LAYERS),
            )
            if teacher_ppl is not None:
                print(f"  Teacher WikiText-2 PPL: {teacher_ppl:.1f}")

        print(f"  Computing FRR HellaSwag (300 samples)...")
        frr_hs = eval_hellaswag(lambda t: model(t), n_samples=300)
        if frr_hs is not None:
            print(f"  FRR HellaSwag: {frr_hs * 100:.1f}%")
            if teacher_hs is not None:
                retention = frr_hs / teacher_hs * 100
                print(f"  HellaSwag retention: {retention:.1f}%")

        print(f"  Computing FRR WikiText-2 PPL...")
        frr_ppl = eval_wikitext_ppl(lambda t: model(t))
        if frr_ppl is not None:
            print(f"  FRR WikiText-2 PPL: {frr_ppl:.1f}")

        model.train()
        print()

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
t1_final, t10_final = eval_vs_teacher(model, n=200)

final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_perlayer_100k_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'=' * 70}")
print(f"FINAL RESULTS -- 1.7B Per-Layer Modulation 100K")
print(f"{'=' * 70}")
print(f"  T1: {t1_final * 100:.1f}%  T10: {t10_final * 100:.1f}%")
print(f"  Compression: {compression:.1f}x")
print(f"  Best T10: {best_t10 * 100:.1f}% (step {best_step})")
print(f"  Training time: {total_time / 3600:.1f} hours")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}/")
print(f"\n  Per-scale baseline: 66.7% T10 at 80K")
print(f"  Target: >67% T10 (beat per-scale record)")
print(f"\n  {'BEAT BASELINE!' if best_t10 > 0.667 else 'Did not beat baseline'}")
print("Done!")
