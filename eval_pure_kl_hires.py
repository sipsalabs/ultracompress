"""
HIGH-RESOLUTION EVAL of pure_kl_step5000.pt

100-sample eval has ~±5pp noise. Run 1000 samples to pin down true T10.
Uses GPU 0 alongside 8B training (low contention, eval is fast).
"""
import lib.unbuffered
import torch
import sys
import os
import time

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = 1000
CKPT_PATH = 'checkpoints_1.7b_pure_kl/pure_kl_step5000.pt'

print("=" * 70)
print(f"HIGH-RES EVAL: {CKPT_PATH}")
print(f"Samples: {N_SAMPLES}  Device: {DEVICE}")
print("=" * 70)


class PerLayerFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        layer_idx = 0
        for scale in range(self.n_scales):
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = self.block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                layer_idx += 1
        x = self.norm(x)
        return self.lm_head(x)


# ── Load teacher ──────────────────────────────────────────────────────
print("\nLoading teacher...")
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
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd

hidden = gd['token_embd.weight'].shape[1]
n_heads = 16
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(
    n_layers=N_TEACHER_LAYERS, n_heads=n_heads, n_kv_heads=8,
    hidden_size=hidden, intermediate_size=hidden * 3,
    vocab_size=vocab_size, head_dim=hidden // n_heads,
)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

# ── Build model and load checkpoint ───────────────────────────────────
print(f"\nLoading checkpoint {CKPT_PATH}...")
model = PerLayerFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

# ── Data ──────────────────────────────────────────────────────────────
print("Loading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
print(f"  {N_TOKENS:,} tokens")

# Use held-out portion (last 10% to match past hires eval convention)
HELD_OUT_START = int(N_TOKENS * 0.9)
print(f"  Held-out start: {HELD_OUT_START:,}")


@torch.no_grad()
def eval_comprehensive(n):
    """All positions + last-token + variance across chunks."""
    torch.manual_seed(42)  # reproducible
    t1_all, t10_all, n_all = 0, 0, 0
    t1_last, t10_last = 0, 0
    t1_last_run = []
    t10_last_run = []
    for i in range(n):
        start = torch.randint(HELD_OUT_START, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[start:start + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        # all positions
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1_all += int(s_top[0] == t_top[0])
            t10_all += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            n_all += 1
        # last token
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        is_t1 = int(s_top[0] == t_top[0])
        is_t10 = len(set(t_top.tolist()) & set(s_top.tolist())) / 10
        t1_last += is_t1
        t10_last += is_t10
        t1_last_run.append(is_t1)
        t10_last_run.append(is_t10)

        if (i + 1) % 100 == 0:
            # Running stats
            t10_mean_so_far = t10_last / (i + 1)
            print(f"  [{i+1}/{n}] last-tok T10 running mean: {t10_mean_so_far*100:.2f}%")

    return {
        'all_pos_t1': t1_all / n_all,
        'all_pos_t10': t10_all / n_all,
        'last_tok_t1': t1_last / n,
        'last_tok_t10': t10_last / n,
        't1_last_samples': t1_last_run,
        't10_last_samples': t10_last_run,
    }


print(f"\nRunning {N_SAMPLES}-sample high-res eval...")
t0 = time.time()
results = eval_comprehensive(N_SAMPLES)
elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"RESULTS ({N_SAMPLES} samples, {elapsed:.0f}s)")
print(f"{'='*70}")
print(f"All positions:  T1={results['all_pos_t1']*100:.2f}%  T10={results['all_pos_t10']*100:.2f}%")
print(f"Last token:     T1={results['last_tok_t1']*100:.2f}%  T10={results['last_tok_t10']*100:.2f}%")

# Compute 95% CI via bootstrapping the last-tok T10
import numpy as np
t10_samples = np.array(results['t10_last_samples'])
rng = np.random.default_rng(42)
boot = np.array([rng.choice(t10_samples, size=len(t10_samples), replace=True).mean()
                 for _ in range(1000)])
ci_lo = np.percentile(boot, 2.5) * 100
ci_hi = np.percentile(boot, 97.5) * 100
print(f"Last-tok T10 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"{'='*70}")
