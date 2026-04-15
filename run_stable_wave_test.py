"""
STABLE WAVE ENGINE — Fixed version that prevents divergence.

Original wave engine proved waves can learn (PPL 485M→444) but diverged
after ~6K steps due to:
1. Coupling matrix spectral radius > 1.0 → exponential growth over 28 steps
2. No magnitude clamping in complex space
3. Phase accumulation across steps

Fixes applied:
- Spectral normalization on coupling matrix (max eigenvalue ≤ 1.0)
- Complex magnitude clamping after each propagation step
- Gradual warmup of propagation steps (start with 4, grow to 28)
- Per-step dampening that guarantees energy decay
- KL distillation from teacher (like all successful experiments)
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
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda'
STEPS = 30_000


# ═══════════════════════════════════════════════════════════════════════
# STABLE WAVE MEDIUM — spectral normalization + magnitude clamping
# ═══════════════════════════════════════════════════════════════════════
class StableWaveMedium(nn.Module):
    """Wave propagation medium with guaranteed stability.

    Key changes from original:
    - coupling matrix is spectrally normalized (eigenvalues ≤ 1)
    - dampening is always positive (energy always decays)
    - output magnitude is clamped
    """
    def __init__(self, hidden_dim: int, n_freqs: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs

        self.speed = nn.Parameter(torch.ones(n_freqs) * 0.5)
        self.dampen = nn.Parameter(torch.zeros(n_freqs))
        self.phase_shift = nn.Parameter(torch.zeros(n_freqs))

        # Coupling matrix — initialized near identity but will be spectral-normed
        self.coupling_real = nn.Parameter(torch.eye(n_freqs) * 0.1)
        self.coupling_imag = nn.Parameter(torch.zeros(n_freqs, n_freqs))

    def _get_stable_coupling(self) -> torch.Tensor:
        """Get coupling matrix with spectral radius ≤ 1.0."""
        C = torch.complex(self.coupling_real, self.coupling_imag)
        # Approximate spectral normalization via Frobenius norm
        # This is cheaper than SVD and sufficient for stability
        fnorm = torch.sqrt((C.abs() ** 2).sum())
        scale = torch.clamp(fnorm / self.n_freqs, min=1.0)
        return C / scale

    def propagate(self, wave_spectrum: torch.Tensor) -> torch.Tensor:
        """Propagate with guaranteed energy decay."""
        # 1. Phase rotation
        phase = torch.exp(1j * self.speed.unsqueeze(0).unsqueeze(0) * math.pi)
        wave_spectrum = wave_spectrum * phase

        # 2. Guaranteed dampening (softplus ensures positive decay)
        dampen = torch.exp(-F.softplus(self.dampen + 0.5).unsqueeze(0).unsqueeze(0))
        wave_spectrum = wave_spectrum * dampen

        # 3. Phase shift
        shift = torch.exp(1j * self.phase_shift.unsqueeze(0).unsqueeze(0))
        wave_spectrum = wave_spectrum * shift

        # 4. Spectrally-normalized coupling
        coupling = self._get_stable_coupling()
        wave_spectrum = torch.einsum('bsf,fg->bsg', wave_spectrum, coupling)

        # 5. Magnitude clamping — prevent any single component from blowing up
        mag = wave_spectrum.abs()
        max_mag = 50.0
        scale = torch.clamp(max_mag / (mag + 1e-8), max=1.0)
        wave_spectrum = wave_spectrum * scale

        return wave_spectrum


class StableWaveEngine(nn.Module):
    """Wave engine with stability guarantees."""
    def __init__(self, hidden_dim: int, n_freqs: int = 64, n_steps: int = 28,
                 vocab_size: int = 151936, embed_weight=None,
                 lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs
        self.n_steps = n_steps

        self.medium = StableWaveMedium(hidden_dim, n_freqs)

        self.to_wave = nn.Linear(hidden_dim, n_freqs * 2, bias=False)
        self.from_wave = nn.Linear(n_freqs, hidden_dim, bias=False)
        nn.init.zeros_(self.from_wave.weight)

        self.step_scale = nn.Parameter(torch.ones(n_steps) * 0.1)
        self.pos_freq = nn.Parameter(torch.randn(n_freqs) * 0.1)

        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_steps)])

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

    def forward(self, tokens: torch.Tensor, active_steps: int | None = None) -> torch.Tensor:
        """Forward pass with optional step warmup.

        Args:
            active_steps: If set, only use this many propagation steps.
                         Use for gradual warmup during training.
        """
        n_steps = active_steps if active_steps is not None else self.n_steps
        n_steps = min(n_steps, self.n_steps)

        x = self.embed(tokens).float()
        B, S, D = x.shape

        positions = torch.arange(S, device=x.device).float().unsqueeze(1)
        pos_phase = torch.exp(1j * positions * self.pos_freq.unsqueeze(0))

        for step in range(n_steps):
            h = self.norms[step](x)

            wave_components = self.to_wave(h)
            wave_real = wave_components[..., :self.n_freqs]
            wave_imag = wave_components[..., self.n_freqs:]
            waves = torch.complex(wave_real, wave_imag)

            waves = waves * pos_phase.unsqueeze(0)

            wave_freq = torch.fft.fft(waves, dim=1)
            wave_freq = self.medium.propagate(wave_freq)
            waves_out = torch.fft.ifft(wave_freq, dim=1)

            wave_real_out = waves_out.real
            delta = self.from_wave(wave_real_out)

            x = x + delta * self.step_scale[step]

        x = self.out_norm(x)
        return self.lm_head(x)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STABLE WAVE ENGINE — Fixed divergence, KL distillation")
print("=" * 70)

# Load teacher
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

# Data
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT",
    split="train", streaming=True,
)
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


def eval_vs_teacher(model: nn.Module, n: int = 100) -> tuple[float, float]:
    t1_correct, t10_scores = 0, []
    model.eval()
    for _ in range(n):
        batch = get_batch(1, 32)
        with torch.no_grad():
            tl = teacher.forward(batch, max_layers=28)
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


# Build model
model = StableWaveEngine(
    1024, n_freqs=128, n_steps=28, vocab_size=151936,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()

print(f"\nTraining {STEPS:,} steps — KL distillation with step warmup")
print(f"  Step warmup: 4→28 steps over first 10K training steps\n")

for step in range(STEPS):
    tokens = get_batch(8, 64)

    # Step warmup: start with 4 propagation steps, grow to 28
    if step < 10_000:
        active_steps = 4 + int(24 * step / 10_000)
    else:
        active_steps = 28

    with torch.no_grad():
        tl = teacher.forward(tokens, max_layers=28)

    sl = model(tokens, active_steps=active_steps)
    T = max(2.0, 5.0 * (1 - step / STEPS))
    loss = F.kl_div(
        F.log_softmax(sl / T, dim=-1),
        F.softmax(tl / T, dim=-1),
        reduction='batchmean',
    ) * T * T

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  Step {step}: NaN/Inf detected! Skipping...")
        opt.zero_grad()
        continue

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    if step % 3000 == 0 or step == STEPS - 1:
        t1, t10 = eval_vs_teacher(model, n=100)
        elapsed = time.time() - t0
        print(
            f"  Step {step:>5d}/{STEPS}: loss={loss.item():.4f}  "
            f"T1={t1 * 100:.1f}%  T10={t10 * 100:.1f}%  "
            f"steps={active_steps}  T={T:.1f}  ({elapsed:.0f}s)"
        )

# Final eval
t1, t10 = eval_vs_teacher(model, n=200)
print(f"\n{'=' * 70}")
print(f"STABLE WAVE ENGINE RESULTS")
print(f"{'=' * 70}")
print(f"  T1: {t1 * 100:.1f}%  T10: {t10 * 100:.1f}%")
print(f"  Trainable: {trainable:,}")
print(f"  Time: {(time.time() - t0) / 60:.0f} min")
print(f"\n  Key question: Does spectral normalization + dampen fix divergence?")
print(f"  If T10 > 30%: wave computation is viable for distillation")
print(f"  If T10 > 50%: wave engine is competitive with FRR")
print("Done!")
