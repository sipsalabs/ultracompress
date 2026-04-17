"""
HIGH-RES EVAL of LoRA per-layer checkpoints.

Training showed last-tok T10=69.7% at step 12500 (200-sample eval), which
would beat the 68.23% hires record from pure_kl_step5000 (1000 samples).

Need 1000-sample hires confirmation on the best LoRA checkpoint(s).
"""
import lib.unbuffered
import torch
import sys
import os
import time
import math

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = 1000
LORA_RANK = 4

CKPT_DIR = 'checkpoints_1.7b_lora_perlayer'

print("=" * 70)
print("HIGH-RES EVAL: LoRA per-layer checkpoints")
print(f"Samples: {N_SAMPLES}  Device: {DEVICE}")
print("=" * 70)


class PerLayerLoRAFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, lora_rank,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.layer_gamma = nn.Parameter(torch.ones(self.total_layers, hidden_dim))
        self.layer_beta = nn.Parameter(torch.zeros(self.total_layers, hidden_dim))
        self.iter_scale = nn.Parameter(torch.ones(n_scales, iters_per_scale))
        self.lora_down = nn.Parameter(
            torch.randn(self.total_layers, hidden_dim, lora_rank) / math.sqrt(hidden_dim)
        )
        self.lora_up = nn.Parameter(torch.zeros(self.total_layers, lora_rank, hidden_dim))
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
                Wd = self.lora_down[layer_idx]
                Wu = self.lora_up[layer_idx]
                lora_out = (x @ Wd) @ Wu
                x = x + (block_out - x) * iter_s + lora_out
                layer_idx += 1
        x = self.norm(x)
        return self.lm_head(x)


# Load teacher
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

# Data
print("Loading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
HELD_OUT_START = int(N_TOKENS * 0.9)


@torch.no_grad()
def hires_eval(model, n=1000, seed=42):
    torch.manual_seed(seed)
    t1_all, t10_all, n_all = 0, 0, 0
    t10_last_samples = []
    for i in range(n):
        start = torch.randint(HELD_OUT_START, all_tokens.numel() - SEQ_LEN, (1,)).item()
        tokens = all_tokens[start:start + SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(tokens, max_layers=N_TEACHER_LAYERS)
        sl = model(tokens)
        for pos in range(SEQ_LEN):
            t_top = tl[0, pos].topk(10).indices
            s_top = sl[0, pos].topk(10).indices
            t1_all += int(s_top[0] == t_top[0])
            t10_all += len(set(t_top.tolist()) & set(s_top.tolist())) / 10
            n_all += 1
        t_top = tl[0, -1].topk(10).indices
        s_top = sl[0, -1].topk(10).indices
        t10_last_samples.append(len(set(t_top.tolist()) & set(s_top.tolist())) / 10)
        if (i + 1) % 200 == 0:
            mean_so_far = sum(t10_last_samples) / len(t10_last_samples)
            print(f"  [{i+1}/{n}] last-tok T10 running: {mean_so_far*100:.2f}%")
    return {
        'all_pos_t1': t1_all / n_all,
        'all_pos_t10': t10_all / n_all,
        'last_tok_t10': sum(t10_last_samples) / n,
        't10_samples': t10_last_samples,
    }


# Which checkpoints exist?
ckpts = []
for f in sorted(os.listdir(CKPT_DIR)):
    if f.endswith('.pt'):
        ckpts.append(os.path.join(CKPT_DIR, f))
print(f"\nFound {len(ckpts)} LoRA checkpoints:")
for c in ckpts:
    print(f"  {c}")


def build_and_eval(ckpt_path):
    print(f"\n{'='*70}\n  Evaluating: {ckpt_path}\n{'='*70}")
    model = PerLayerLoRAFRR(
        hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
        vocab_size=vocab_size, ff_mult=1, lora_rank=LORA_RANK,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    t0 = time.time()
    res = hires_eval(model, n=N_SAMPLES)
    el = time.time() - t0
    print(f"\n  Results ({el:.0f}s):")
    print(f"    All-pos: T1={res['all_pos_t1']*100:.2f}%  T10={res['all_pos_t10']*100:.2f}%")
    print(f"    Last-tok T10: {res['last_tok_t10']*100:.2f}%")
    # 95% CI bootstrap
    samples = np.array(res['t10_samples'])
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(samples, size=len(samples), replace=True).mean()
                     for _ in range(1000)])
    lo = np.percentile(boot, 2.5) * 100
    hi = np.percentile(boot, 97.5) * 100
    print(f"    95% CI: [{lo:.2f}, {hi:.2f}]")
    vs_record = (res['last_tok_t10'] * 100) - 68.23
    print(f"    vs pure_kl record (68.23%): {vs_record:+.2f}pp")
    del model
    torch.cuda.empty_cache()
    return res


# Evaluate best-looking checkpoints: step10000, step15000 (final)
targets = [c for c in ckpts if 'step10000' in c or 'step15000' in c
           or 'final' in c or 'best' in c]
if not targets:
    targets = ckpts  # fall back: all
print(f"\nWill evaluate {len(targets)} checkpoints: {targets}")

for c in targets:
    try:
        build_and_eval(c)
    except Exception as e:
        print(f"  FAILED on {c}: {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 70)
print("HIRES EVAL COMPLETE")
print("=" * 70)
