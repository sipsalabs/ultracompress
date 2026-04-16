"""
HIGH-RES EVAL of dual_block_step5000.pt

Verify if dual-block broke hires T10 all-pos ceiling of 63.9%.
Training eval showed 64.2% at step 5000.
"""
import lib.unbuffered
import torch
import sys
import os
import time
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

DEVICE = 'cuda:1'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = 1000
CKPT_PATH = 'checkpoints_1.7b_dual_block/dual_block_step5000.pt'
SPLIT_SCALE = 2

print("=" * 70)
print(f"HIGH-RES EVAL: {CKPT_PATH}")
print(f"Samples: {N_SAMPLES}  Device: {DEVICE}")
print("=" * 70)


class DualBlockFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, split_scale=2,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.split_scale = split_scale
        self.total_layers = n_scales * iters_per_scale
        self.block_early = FractalBlock(hidden_dim, n_heads, ff_mult)
        self.block_late = FractalBlock(hidden_dim, n_heads, ff_mult)
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
            block = self.block_early if scale < self.split_scale else self.block_late
            for it in range(self.iters_per_scale):
                gamma = self.layer_gamma[layer_idx]
                beta = self.layer_beta[layer_idx]
                iter_s = self.iter_scale[scale, it]
                block_out = block(x, gamma, beta)
                x = x + (block_out - x) * iter_s
                layer_idx += 1
        x = self.norm(x)
        return self.lm_head(x)


# Teacher load
print("Loading teacher...")
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

print(f"Loading {CKPT_PATH}...")
model = DualBlockFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1, split_scale=SPLIT_SCALE,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

print("Loading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
N_TOKENS = all_tokens.shape[0]
HELD_OUT_START = int(N_TOKENS * 0.9)


@torch.no_grad()
def eval_comprehensive(n):
    torch.manual_seed(42)
    t1_all, t10_all, n_all = 0, 0, 0
    t1_last, t10_last = 0, 0
    t10_last_run = []
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
        t1_last += int(s_top[0] == t_top[0])
        is_t10 = len(set(t_top.tolist()) & set(s_top.tolist())) / 10
        t10_last += is_t10
        t10_last_run.append(is_t10)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] all-pos T10: {t10_all/n_all*100:.2f}%  last-tok T10: {t10_last/(i+1)*100:.2f}%")
    return {
        'all_pos_t1': t1_all / n_all,
        'all_pos_t10': t10_all / n_all,
        'last_tok_t1': t1_last / n,
        'last_tok_t10': t10_last / n,
        't10_last_samples': np.array(t10_last_run),
    }


t0 = time.time()
results = eval_comprehensive(N_SAMPLES)
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"DUAL-BLOCK HIRES RESULTS ({N_SAMPLES} samples, {elapsed:.0f}s)")
print(f"{'='*70}")
print(f"All positions:  T1={results['all_pos_t1']*100:.2f}%  T10={results['all_pos_t10']*100:.2f}%")
print(f"Last token:     T1={results['last_tok_t1']*100:.2f}%  T10={results['last_tok_t10']*100:.2f}%")

rng = np.random.default_rng(42)
boot = np.array([rng.choice(results['t10_last_samples'], size=N_SAMPLES, replace=True).mean()
                 for _ in range(1000)])
ci_lo, ci_hi = np.percentile(boot, 2.5) * 100, np.percentile(boot, 97.5) * 100
print(f"Last-tok T10 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

# Compare to baseline hires (pure KL step5000): all-pos 62.80%, last-tok 68.23%
print(f"\nCOMPARISON:")
print(f"  Single-block (pure_kl step5000): all-pos 62.80%, last-tok 68.23%")
print(f"  Dual-block  (step5000):           all-pos {results['all_pos_t10']*100:.2f}%, last-tok {results['last_tok_t10']*100:.2f}%")
delta_all = (results['all_pos_t10'] - 0.6280) * 100
delta_last = (results['last_tok_t10'] - 0.6823) * 100
print(f"  Delta:                            all-pos {delta_all:+.2f}pp, last-tok {delta_last:+.2f}pp")
print(f"{'='*70}")
