"""
SELECTIVE STUDENT EXPERIMENT — Does learning WHEN to trust the teacher break the barrier?

Previous experiments showed:
  - Real text KL distillation: 62% T10 (1.7B), 60% T10 (0.6B)
  - Standard KL plateaus around 62-67% T10
  - Training from scratch can BEAT teacher PPL (FRR 1614 < teacher 2404)
  - Teacher is sometimes WRONG. Diverging from teacher HELPS.

The hypothesis: a trust gate that detects reliable vs unreliable teacher
predictions will combine the speed of distillation with the ceiling-break
of self-supervised learning.

Tests:
  1. Standard KL (baseline) — 15K steps, real text
  2. Selective Student — trust gate blends KL + NTP
  3. Curriculum: trust teacher early, then gradually switch to self
"""
import lib.unbuffered
import torch, sys, os, time
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel
from ultracompress.selective_student import TrustGate, selective_loss
from transformers import AutoTokenizer
from datasets import load_dataset

# Ensure CWD is project root (so cache files are found)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = 'cuda'
STEPS = 15_000

print("=" * 70)
print("SELECTIVE STUDENT: Does learning WHEN to trust break the barrier?")
print("=" * 70)

# ── Load teacher ──────────────────────────────────────────────────────
print("Loading teacher (Qwen3-0.6B)...")
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
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
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(1024)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(28):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()

config = ModelConfig(
    n_layers=28, n_heads=16, n_kv_heads=8, hidden_size=1024,
    intermediate_size=3072, vocab_size=151936, head_dim=128,
)
teacher = MiniTransformer(config, device)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(device)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(device)

embed_w = gd['token_embd.weight'].to(device)
norm_w = gd['output_norm.weight'].to(device)
lm_head_w = gd['output.weight'].to(device)

# ── Data ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")


def get_batch(batch_size: int = 8, seq_len: int = 64) -> torch.Tensor:
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
    return torch.stack(tokens_list).to(device)


def eval_real_text(model, n: int = 200) -> tuple[float, float]:
    """Evaluate T1 and T10 agreement with teacher on real text."""
    t1_correct, t10_scores = 0, []
    model.eval()
    for _ in range(n):
        batch = get_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=28)
            sl = model(batch)
            # Last position prediction
            t_top = tl[0, -1].argmax().item()
            s_top = sl[0, -1].argmax().item()
            t_top10 = set(tl[0, -1].topk(10).indices.tolist())
            s_top10 = set(sl[0, -1].topk(10).indices.tolist())
            if t_top == s_top:
                t1_correct += 1
            t10_scores.append(len(t_top10 & s_top10) / 10)
    model.train()
    return t1_correct / n, sum(t10_scores) / len(t10_scores)


# ── Training loop ────────────────────────────────────────────────────
def train_experiment(
    name: str,
    model: nn.Module,
    loss_fn,
    steps: int = STEPS,
    extra_params: list | None = None,
) -> tuple[float, float, list]:
    """Train and evaluate. Returns (t1, t10, loss_history)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if extra_params:
        params += extra_params
    trainable = sum(p.numel() for p in params)

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  Trainable params: {trainable:,}")
    print(f"{'=' * 70}")

    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    t0 = time.time()

    for step in range(steps):
        tokens = get_batch(8, 64)

        with torch.no_grad():
            tl = teacher.forward(tokens, max_layers=28)

        sl = model(tokens)
        loss = loss_fn(sl, tl, tokens, step, steps)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())

        if step % 3000 == 0 or step == steps - 1:
            t1, t10 = eval_real_text(model, n=100)
            elapsed = time.time() - t0
            print(
                f"  Step {step:>5d}/{steps}: loss={loss.item():.4f}  "
                f"T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  "
                f"({elapsed:.0f}s)"
            )

    t1, t10 = eval_real_text(model, n=200)
    print(f"  FINAL: T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  ({time.time() - t0:.0f}s)")
    return t1, t10, losses


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Standard KL baseline (real text)
# ═══════════════════════════════════════════════════════════════════════
def standard_kl_loss(sl, tl, tokens, step, total_steps):
    T = max(2.0, 5.0 * (1 - step / total_steps))
    return F.kl_div(
        F.log_softmax(sl / T, dim=-1),
        F.softmax(tl / T, dim=-1),
        reduction='batchmean',
    ) * T * T

frr_baseline = FractalModel(
    1024, 16, 4, 7, 151936, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
t1_baseline, t10_baseline, _ = train_experiment(
    "Experiment 1: Standard KL (baseline)", frr_baseline, standard_kl_loss,
)
del frr_baseline
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Selective Student (trust gate)
# ═══════════════════════════════════════════════════════════════════════
trust_gate = TrustGate(vocab_size=151936, hidden_dim=64).to(device)

def selective_student_loss(sl, tl, tokens, step, total_steps):
    return selective_loss(sl, tl, tokens, trust_gate, step, total_steps)

frr_selective = FractalModel(
    1024, 16, 4, 7, 151936, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
t1_selective, t10_selective, _ = train_experiment(
    "Experiment 2: Selective Student (trust gate)",
    frr_selective,
    selective_student_loss,
    extra_params=list(trust_gate.parameters()),
)

# Log trust gate statistics at the end
with torch.no_grad():
    test_batch = get_batch(4, 64)
    tl = teacher.forward(test_batch, max_layers=28)
    sl = frr_selective(test_batch)
    trust_scores = trust_gate(sl, tl)
    print(f"  Trust gate stats: mean={trust_scores.mean():.3f}  "
          f"std={trust_scores.std():.3f}  "
          f"min={trust_scores.min():.3f}  max={trust_scores.max():.3f}")

del frr_selective, trust_gate
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Curriculum — teacher first, then self
# ═══════════════════════════════════════════════════════════════════════
def curriculum_loss(sl, tl, tokens, step, total_steps):
    """Hard curriculum: 100% KL first half, linear blend to 50% NTP second half."""
    T = max(2.0, 5.0 * (1 - step / total_steps))
    kl = F.kl_div(
        F.log_softmax(sl / T, dim=-1),
        F.softmax(tl / T, dim=-1),
        reduction='batchmean',
    ) * T * T

    B, S, V = sl.shape
    ntp = F.cross_entropy(
        sl[:, :-1].reshape(-1, V),
        tokens[:, 1:].reshape(-1),
    )

    # First half: pure KL. Second half: blend in NTP
    progress = step / total_steps
    if progress < 0.5:
        alpha = 0.0
    else:
        alpha = (progress - 0.5) * 2.0  # 0->1 over second half

    return (1 - alpha) * kl + alpha * ntp

frr_curriculum = FractalModel(
    1024, 16, 4, 7, 151936, 1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)
t1_curriculum, t10_curriculum, _ = train_experiment(
    "Experiment 3: Curriculum (KL → NTP)", frr_curriculum, curriculum_loss,
)
del frr_curriculum
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"RESULTS: Does selective trust break the barrier?")
print(f"{'=' * 70}")
print(f"  {'Approach':<35} {'T1':>8} {'T10':>8}")
print(f"  {'-' * 51}")
print(f"  {'Standard KL (baseline)':<35} {t1_baseline * 100:>7.1f}% {t10_baseline * 100:>7.1f}%")
print(f"  {'Selective Student (trust gate)':<35} {t1_selective * 100:>7.1f}% {t10_selective * 100:>7.1f}%")
print(f"  {'Curriculum (KL → NTP)':<35} {t1_curriculum * 100:>7.1f}% {t10_curriculum * 100:>7.1f}%")
print(f"\n  Previous best (0.6B, 15K): ~35% T1, ~55% T10")
print(f"  Record (0.6B, 100K):        44% T1,  62% T10")
print(f"\n  If Selective > Standard → trust gate works, scale to 100K")
print(f"  If Curriculum > Standard → curriculum works, test at 1.7B")
print("Done!")
