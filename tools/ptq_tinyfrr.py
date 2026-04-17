"""
Post-training quantization on TinyFRR checkpoints.
Weight-only per-output-channel symmetric quantization at INT8 and INT4.

Compression stack:
  h=128 architectural:  2201x  -> 68.44%
  + INT8 (2x):           4402x -> expect ~67-68%
  + INT4 (4x):           8804x -> expect ~65-66%
For 100T model at h=128 + INT4: 100T / 8804 = 11.36B params
  11.36B * 0.5 byte(INT4) = 5.68 GB. Fits on a phone.

Runs on whichever GPU is free. Auto-detects.
"""
import lib.unbuffered
import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--tags', nargs='+', default=['h128_long', 'h48_long', 'h512'])
ap.add_argument('--bits', nargs='+', type=int, default=[8, 4])
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--device', type=str, default='cuda:0')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = args.n

print(f"PTQ eval: tags={args.tags}  bits={args.bits}  n={N_SAMPLES}  device={DEVICE}")

# ==================== Teacher ====================
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
hf_to_gguf = {
    'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
    'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
    'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
    'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
    'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}
gd = {}
gd['token_embd.weight'] = wd['model.embed_tokens.weight'].float()
gd['output_norm.weight'] = wd.get('model.norm.weight', torch.ones(2048)).float()
gd['output.weight'] = wd.get('lm_head.weight', gd['token_embd.weight']).float()
for li in range(N_TEACHER_LAYERS):
    for h, g in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd: gd[f'blk.{li}.{g}'] = wd[k].float()
del wd
H_OUTER = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=H_OUTER, intermediate_size=H_OUTER*3,
                  vocab_size=vocab_size, head_dim=H_OUTER//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd

all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)
torch.manual_seed(42)
starts = torch.randint(0, all_tokens.numel() - SEQ_LEN, (N_SAMPLES,))


# ==================== Student ====================
class TinyFRR(nn.Module):
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7,
            vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)

    def forward(self, tokens):
        x_outer = self.embed(tokens).float()
        x = self.proj_in(x_outer)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x_outer = self.proj_out(x)
        x_outer = self.norm_outer(x_outer)
        return F.linear(x_outer, self.lm_head_w)


def load_ckpt(tag):
    path = f'checkpoints_1.7b_tinyfrr_{tag}/best.pt'
    if not os.path.exists(path): return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h_inner = ck['h_inner']
    n_heads_inner = ck.get('n_heads_inner', 16)
    m = TinyFRR(H_OUTER, h_inner, n_heads_inner, vocab_size,
                embed_w, lm_head_w, norm_w).to(DEVICE)
    m.load_state_dict(ck['state_dict'], strict=False)
    m.eval()
    return m, ck


# ==================== Quantization ====================
@torch.no_grad()
def quantize_per_channel_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric per-output-channel quantize-dequantize.
    w: [out, in]. Returns same-dtype tensor with quantization noise baked in.
    """
    assert w.dim() == 2
    qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4
    # Per-output-row absmax
    absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = absmax / qmax
    q = (w / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_model_weights(model: TinyFRR, bits: int) -> int:
    """In-place quantize all 2D weight tensors on trainable params. Returns count."""
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.requires_grad:
            # Skip teacher embed/lm_head which are frozen buffers anyway
            mod.weight.data = quantize_per_channel_symmetric(mod.weight.data, bits)
            count += 1
    # Also quantize the inner.block.qkv/o_proj/gate/up/down explicitly
    # (they ARE nn.Linear with requires_grad=True, so already handled above)
    return count


# ==================== Eval ====================
@torch.no_grad()
def eval_model(model, tag):
    all_t1 = all_t10 = 0
    last_t1 = last_t10 = 0
    n = 0
    for i, s in enumerate(starts):
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = model(toks)
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            h1 = int(st[0] == tt[0])
            h10 = len(set(tt.tolist()) & set(st.tolist())) / 10
            all_t1 += h1; all_t10 += h10; n += 1
            if pos == SEQ_LEN - 1:
                last_t1 += h1; last_t10 += h10
        if (i+1) % 100 == 0:
            print(f"  [{tag}] {i+1}/{N_SAMPLES}", flush=True)
    return {
        'all_t1': all_t1/n, 'all_t10': all_t10/n,
        'last_t1': last_t1/N_SAMPLES, 'last_t10': last_t10/N_SAMPLES,
    }


# ==================== Main ====================
results = {}
for tag in args.tags:
    loaded = load_ckpt(tag)
    if loaded is None:
        print(f"\n!!! {tag}: checkpoint missing, skipping", flush=True)
        continue
    model, ck = loaded
    h_inner = ck['h_inner']
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n>>> {tag}  h_inner={h_inner}  trainable={trainable_params}", flush=True)

    # FP32/BF16 baseline
    t0 = time.time()
    base = eval_model(model, tag + '_fp')
    print(f"  {tag} FP    : last-T10={base['last_t10']*100:.2f}%  all-T10={base['all_t10']*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    results[(tag, 'fp')] = base

    for bits in args.bits:
        # Reload fresh copy
        model, _ = load_ckpt(tag)
        n_q = quantize_model_weights(model, bits)
        print(f"  [{tag}] INT{bits}: quantized {n_q} linear layers", flush=True)
        t0 = time.time()
        q = eval_model(model, f'{tag}_int{bits}')
        delta = (q['last_t10'] - base['last_t10']) * 100
        print(f"  {tag} INT{bits}: last-T10={q['last_t10']*100:.2f}%  all-T10={q['all_t10']*100:.2f}%  delta={delta:+.2f}pp  ({time.time()-t0:.0f}s)", flush=True)
        results[(tag, f'int{bits}')] = q
        del model
        torch.cuda.empty_cache()

# ==================== Summary ====================
print("\n" + "="*78)
print("PTQ SUMMARY")
print("="*78)
print(f"{'tag':<14}{'precision':<10}{'last-T10':<12}{'all-T10':<12}{'delta-vs-fp':<14}")
for tag in args.tags:
    fp = results.get((tag, 'fp'))
    if fp is None: continue
    print(f"{tag:<14}{'fp':<10}{fp['last_t10']*100:>8.2f}%   {fp['all_t10']*100:>8.2f}%")
    for bits in args.bits:
        r = results.get((tag, f'int{bits}'))
        if r is None: continue
        d = (r['last_t10'] - fp['last_t10']) * 100
        print(f"{tag:<14}{'int'+str(bits):<10}{r['last_t10']*100:>8.2f}%   {r['all_t10']*100:>8.2f}%   {d:+.2f}pp")

torch.save(results, 'tinyfrr_ptq_results.pt')
print("\nSaved: tinyfrr_ptq_results.pt")
