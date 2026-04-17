"""
1.7B REAL TEXT DISTILLATION — 100K steps with HellaSwag evaluation.

Priority #1 from STATUS.md: "Real text distillation at 1.7B scale, 100K steps
— should push to 75%+ real T10."

Previous records:
  - 1.7B real text 50K: T1=46% T10=62% (best T1 ever)
  - 1.7B random 100K: T10=67% (plateau confirmed)
  - 0.6B 100K: 83.3% HellaSwag retention

Improvements over run_1.7b_real_text.py:
  - Checkpoints saved every 10K steps
  - HellaSwag + WikiText-2 eval at 50K and 100K
  - T1/T10 eval every 5K steps
  - Best model tracking and saving
  - Configurable GPU device (default cuda:1 for parallel training)
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
DEVICE = 'cuda:1'  # Use GPU 1 (GPU 0 may be running other experiments)
STEPS = 100_000
BATCH_SIZE = 4
SEQ_LEN = 64
LR = 5e-4
EVAL_INTERVAL = 5_000       # T1/T10 eval
CHECKPOINT_INTERVAL = 10_000  # Save checkpoint
HELLASWAG_STEPS = [50_000, 99_999]  # HellaSwag + PPL eval (expensive)
CHECKPOINT_DIR = 'checkpoints_1.7b_real_text'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("1.7B REAL TEXT DISTILLATION — 100K steps + HellaSwag eval")
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
del wd  # Free ~7GB RAM

hidden = gd['token_embd.weight'].shape[1]  # 2048
n_heads = 16
head_dim = hidden // n_heads  # 128
vocab_size = gd['token_embd.weight'].shape[0]  # 151936
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
del gd  # Free RAM

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
    """T1 and T10 agreement with teacher on real text."""
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
    """HellaSwag accuracy via log-probability scoring."""
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
    """WikiText-2 perplexity."""
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


# ── Build FRR Student ─────────────────────────────────────────────────
model = FractalModel(
    hidden, n_heads, 4, 7, vocab_size, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)
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

print(f"\nTraining {STEPS:,} steps with REAL TEXT distillation...")
print(f"  Batch: {BATCH_SIZE}×{SEQ_LEN}, LR: {LR}, Checkpoints: every {CHECKPOINT_INTERVAL}")
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
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'frr_1.7b_step{step}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

        # Also save as best if applicable
        if t10 >= best_t10:
            best_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best model: {best_path} (T10={best_t10 * 100:.1f}%)")

    # ── HellaSwag + WikiText-2 Evaluation ─────────────────────────
    if step in HELLASWAG_STEPS:
        print(f"\n  --- HellaSwag + WikiText-2 evaluation at step {step} ---")
        model.eval()

        # Teacher HellaSwag (run once for comparison)
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

        # FRR HellaSwag
        print(f"  Computing FRR HellaSwag (300 samples)...")
        frr_hs = eval_hellaswag(lambda t: model(t), n_samples=300)
        if frr_hs is not None:
            print(f"  FRR HellaSwag: {frr_hs * 100:.1f}%")
            if teacher_hs is not None:
                retention = frr_hs / teacher_hs * 100
                print(f"  HellaSwag retention: {retention:.1f}%")

        # FRR WikiText-2
        print(f"  Computing FRR WikiText-2 PPL...")
        frr_ppl = eval_wikitext_ppl(lambda t: model(t))
        if frr_ppl is not None:
            print(f"  FRR WikiText-2 PPL: {frr_ppl:.1f}")

        model.train()
        print()

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
t1_final, t10_final = eval_vs_teacher(model, n=200)

# Save final model
final_path = os.path.join(CHECKPOINT_DIR, 'frr_1.7b_100k_final.pt')
torch.save(model.state_dict(), final_path)

print(f"\n{'=' * 70}")
print(f"FINAL RESULTS — 1.7B Real Text 100K")
print(f"{'=' * 70}")
print(f"  T1: {t1_final * 100:.1f}%  T10: {t10_final * 100:.1f}%")
print(f"  Compression: {compression:.1f}x")
print(f"  Best T10: {best_t10 * 100:.1f}% (step {best_step})")
print(f"  Training time: {total_time / 3600:.1f} hours")
print(f"  Checkpoint dir: {CHECKPOINT_DIR}/")
print(f"\n  Previous records:")
print(f"    1.7B real text 50K: T1=46% T10=62%")
print(f"    1.7B random 100K:  T10=67%")
print(f"    0.6B 100K HellaSwag: 83.3% retention")
print(f"\n  NEW RECORD?" if t10_final > 0.67 else "\n  Needs more training or scaling")
print("Done!")
