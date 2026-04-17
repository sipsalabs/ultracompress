"""Quick hires eval of lora_final.pt only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reuse the eval script's machinery
# Just override which checkpoints to eval
import lib.unbuffered
import torch, time, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalBlock

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = 1000
LORA_RANK = 4


class PerLayerLoRAFRR(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_scales, iters_per_scale,
                 vocab_size, ff_mult, lora_rank,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
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
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        layer_idx = 0
        for s in range(self.n_scales):
            for i in range(self.iters_per_scale):
                g = self.layer_gamma[layer_idx]
                b = self.layer_beta[layer_idx]
                it = self.iter_scale[s, i]
                bo = self.block(x, g, b)
                lo = (x @ self.lora_down[layer_idx]) @ self.lora_up[layer_idx]
                x = x + (bo - x) * it + lo
                layer_idx += 1
        x = self.norm(x)
        return self.lm_head(x)


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

print("Loading data...")
all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
HELD_OUT_START = int(all_tokens.numel() * 0.9)

model = PerLayerLoRAFRR(
    hidden_dim=hidden, n_heads=n_heads, n_scales=4, iters_per_scale=7,
    vocab_size=vocab_size, ff_mult=1, lora_rank=LORA_RANK,
    embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
).to(DEVICE)

CKPTS = [
    'checkpoints_1.7b_lora_perlayer/lora_final.pt',
    'checkpoints_1.7b_lora_perlayer/lora_step5000.pt',
]

for ckpt_path in CKPTS:
    if not os.path.exists(ckpt_path):
        print(f"SKIP (not found): {ckpt_path}")
        continue
    print(f"\n{'='*70}\n{ckpt_path}\n{'='*70}")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    torch.manual_seed(42)
    t1_all, t10_all, n_all = 0, 0, 0
    t10_last = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_SAMPLES):
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
            t10_last.append(len(set(t_top.tolist()) & set(s_top.tolist())) / 10)
            if (i+1) % 250 == 0:
                print(f"  [{i+1}/{N_SAMPLES}] last-tok T10: {sum(t10_last)/len(t10_last)*100:.2f}%")

    ap_t10 = t10_all / n_all
    lt_t10 = sum(t10_last) / N_SAMPLES
    arr = np.array(t10_last)
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(1000)])
    lo, hi = np.percentile(boot, 2.5)*100, np.percentile(boot, 97.5)*100
    print(f"  ({time.time()-t0:.0f}s)  all-pos T10={ap_t10*100:.2f}%  last-tok T10={lt_t10*100:.2f}%  CI=[{lo:.2f},{hi:.2f}]")
    print(f"  vs pure_kl record (68.23%): {(lt_t10*100 - 68.23):+.2f}pp")
