"""
Plug-and-play evaluation: combine separately-trained FRR body + ASVD head.

Tests whether FRR body output (proj_out → norm) feeds well into ASVD head
that was trained on teacher's actual latents.

Pipeline: tokens → teacher_embed → proj_in → FRR → proj_out → norm → ASVD_proj → ASVD_out → logits

Usage:
  python eval_plugplay.py --body h128_long --heads r1024 r512 --n 200 --device cuda:1
"""
import lib.unbuffered
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--body', type=str, default='h128_long', help='body checkpoint tag')
ap.add_argument('--heads', nargs='+', default=['asvd_r1024_ft', 'asvd_r512_ft'],
                help='ASVD head checkpoint tags')
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--device', type=str, default='cuda:1')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_TEACHER_LAYERS = 28

print(f"Plug-and-play eval: body={args.body}  heads={args.heads}  n={args.n}  device={DEVICE}")

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
starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (args.n,))

# ==================== Load FRR Body ====================
body_path = f'checkpoints_1.7b_tinyfrr_{args.body}/best.pt'
print(f"\nLoading body: {body_path}")
body_ck = torch.load(body_path, map_location='cpu', weights_only=False)
h_inner = body_ck['h_inner']
n_heads_inner = body_ck.get('n_heads_inner', next(
    (h for h in [16, 8, 12, 4] if h_inner % h == 0), 4))
print(f"  h_inner={h_inner}  n_heads={n_heads_inner}")

# Build body components
proj_in = nn.Linear(H_OUTER, h_inner, bias=False).to(DEVICE)
proj_out = nn.Linear(h_inner, H_OUTER, bias=False).to(DEVICE)
norm_outer = nn.RMSNorm(H_OUTER).to(DEVICE)
inner = FractalModel(
    hidden_dim=h_inner, n_heads=n_heads_inner,
    n_scales=4, iters_per_scale=7, vocab_size=vocab_size, ff_mult=1,
    embed_weight=None, lm_head_weight=None, norm_weight=None,
).to(DEVICE)

# Load body weights from checkpoint
sd = body_ck['state_dict']
proj_in.weight.data.copy_(sd['proj_in.weight'])
proj_out.weight.data.copy_(sd['proj_out.weight'])
if 'norm_outer.weight' in sd:
    norm_outer.weight.data.copy_(sd['norm_outer.weight'])
else:
    norm_outer.weight.data.copy_(norm_w.cpu())

# Load inner FRR weights
for k, v in sd.items():
    if k.startswith('inner.'):
        inner_key = k[6:]  # strip 'inner.'
        obj = inner
        parts = inner_key.split('.')
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        final = parts[-1]
        if final.isdigit():
            obj[int(final)].data.copy_(v)
        elif hasattr(getattr(obj, final, None), 'data'):
            getattr(obj, final).data.copy_(v)
        else:
            setattr(obj, final, nn.Parameter(v.to(DEVICE)))

body_params = sum(p.numel() for n, p in list(proj_in.named_parameters()) +
                  list(proj_out.named_parameters()) + list(inner.named_parameters())
                  if 'embed' not in n and 'lm_head' not in n and 'norm.' not in n)
print(f"  Body params: {body_params/1e6:.2f}M")

# ==================== Evaluate Function ====================
@torch.no_grad()
def run_body(tokens):
    """Run FRR body: embed → proj_in → FRR → proj_out → norm → latent [B, T, H_OUTER]"""
    x = F.embedding(tokens, embed_w).float()
    x = proj_in(x)
    for scale in range(inner.n_scales):
        gamma = inner.scale_gamma[scale]
        beta = inner.scale_beta[scale]
        for it in range(inner.iters_per_scale):
            iter_s = inner.iter_scale[scale, it]
            x = x + (inner.block(x, gamma, beta, None, None, None) - x) * iter_s
    x = proj_out(x)
    return norm_outer(x)

@torch.no_grad()
def evaluate(logits_fn, tag, n):
    teacher_nll = student_nll = 0.0
    top1_agree = top10_overlap = kl_total = 0
    n_tokens = 0

    for i in range(n):
        s = int(starts[i].item())
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]; tgt = toks[0, 1:SEQ_LEN+1]

        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = logits_fn(inp)

        t_logp = F.log_softmax(tl[0].float(), dim=-1)
        s_logp = F.log_softmax(sl[0].float(), dim=-1)
        teacher_nll += -t_logp[torch.arange(SEQ_LEN), tgt].sum().item()
        student_nll += -s_logp[torch.arange(SEQ_LEN), tgt].sum().item()

        t_top1 = tl[0].argmax(-1); s_top1 = sl[0].argmax(-1)
        top1_agree += (t_top1 == s_top1).sum().item()

        t_prob = F.softmax(tl[0].float(), dim=-1)
        kl = (t_prob * (t_prob.clamp_min(1e-12).log() - s_logp)).sum(-1)
        kl_total += kl.sum().item()

        for pos in range(SEQ_LEN):
            tt = set(tl[0, pos].topk(10).indices.tolist())
            st = set(sl[0, pos].topk(10).indices.tolist())
            top10_overlap += len(tt & st) / 10.0

        n_tokens += SEQ_LEN
        if (i + 1) % 50 == 0:
            print(f"    [{tag}] {i+1}/{n}", flush=True)

    t_ppl = math.exp(teacher_nll / n_tokens)
    s_ppl = math.exp(student_nll / n_tokens)
    ppl_ratio = s_ppl / t_ppl
    return {
        'teacher_ppl': t_ppl, 'student_ppl': s_ppl,
        'ppl_ratio': ppl_ratio, 'quality_pct': 100.0 / ppl_ratio,
        'top1_pct': 100.0 * top1_agree / n_tokens,
        'top10_pct': 100.0 * top10_overlap / n_tokens,
        'kl_per_token': kl_total / n_tokens,
    }

# ==================== Evaluate body + full lm_head (baseline) ====================
print(f"\n{'='*100}")
print("Evaluating body + FULL lm_head (baseline)...")
body_full_fn = lambda inp: F.linear(run_body(inp), lm_head_w)
r_full = evaluate(body_full_fn, f'{args.body}+full_head', args.n)
body_full_params = body_params + lm_head_w.numel()
r_full['total_params'] = body_full_params
print(f"  {args.body}+full_head: T1={r_full['top1_pct']:.2f}%  T10={r_full['top10_pct']:.2f}%  "
      f"PPL={r_full['ppl_ratio']:.3f}  params={body_full_params/1e6:.2f}M")

# ==================== Evaluate body + ASVD heads ====================
results = {'full_head': r_full}
teacher_total = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + lm_head_w.numel()

for head_tag in args.heads:
    head_dir = f'checkpoints_1.7b_{head_tag}'
    head_path = os.path.join(head_dir, 'best.pt')
    if not os.path.exists(head_path):
        print(f"\n  SKIP {head_tag}: no checkpoint at {head_path}")
        continue

    print(f"\nLoading ASVD head: {head_path}")
    hck = torch.load(head_path, map_location='cpu', weights_only=False)

    # Handle both checkpoint formats: direct weights or nested state_dict
    if 'state_dict' in hck:
        hsd = hck['state_dict']
        asvd_proj_w = hsd['asvd_proj.weight'].to(DEVICE)
        asvd_out_w = hsd['asvd_out.weight'].to(DEVICE)
    else:
        asvd_proj_w = hck['asvd_proj'].to(DEVICE)   # [r, 2048]
        asvd_out_w = hck['asvd_out'].to(DEVICE)     # [V, r]
    rank = asvd_proj_w.shape[0]
    head_params = asvd_proj_w.numel() + asvd_out_w.numel()

    print(f"  ASVD r={rank}  head_params={head_params/1e6:.2f}M")

    def make_asvd_fn(proj_w, out_w):
        def fn(inp):
            latent = run_body(inp)
            z = F.linear(latent, proj_w)
            return F.linear(z, out_w)
        return fn

    asvd_fn = make_asvd_fn(asvd_proj_w, asvd_out_w)
    tag = f'{args.body}+{head_tag}'
    r = evaluate(asvd_fn, tag, args.n)
    total = body_params + head_params
    r['total_params'] = total
    r['head_params'] = head_params
    r['body_params'] = body_params
    r['compression'] = teacher_total / total
    results[head_tag] = r

    print(f"  {tag}: T1={r['top1_pct']:.2f}%  T10={r['top10_pct']:.2f}%  "
          f"PPL={r['ppl_ratio']:.3f}  params={total/1e6:.2f}M  comp={r['compression']:.1f}x")

    del asvd_proj_w, asvd_out_w, hck
    torch.cuda.empty_cache()

# ==================== Summary ====================
print(f"\n{'='*100}")
print(f"PLUG-AND-PLAY COMBINED RESULTS (body={args.body}, h_inner={h_inner})")
print(f"{'='*100}")
print(f"{'config':35s} {'total_params':>12s} {'compression':>12s} {'top1':>8s} {'top10':>8s} {'ppl':>8s} {'kl':>8s}")
print(f"{'-'*100}")

# Full head baseline
print(f"{args.body+'+full_head':35s} {body_full_params/1e6:>10.2f}M {'1.0x':>12s} "
      f"{r_full['top1_pct']:>7.2f}% {r_full['top10_pct']:>7.2f}% "
      f"{r_full['ppl_ratio']:>7.3f} {r_full['kl_per_token']:>7.3f}")

for head_tag, r in sorted(((k, v) for k, v in results.items() if k != 'full_head'),
                           key=lambda x: -x[1]['top10_pct']):
    comp = r.get('compression', 0)
    print(f"{args.body+'+'+head_tag:35s} {r['total_params']/1e6:>10.2f}M {comp:>11.1f}x "
          f"{r['top1_pct']:>7.2f}% {r['top10_pct']:>7.2f}% "
          f"{r['ppl_ratio']:>7.3f} {r['kl_per_token']:>7.3f}")

print(f"{'='*100}")

# Head-only reference (from ASVD fine-tune results)
print(f"\nReference (head-only, teacher latents):")
print(f"  ASVD r=1024 ft: T1=91.66%  T10=92.57%  (2.0x head shrink)")
print(f"  ASVD r=512 ft:  T1=87.73%  T10=88.93%  (3.9x head shrink)")
print(f"\nDONE eval_plugplay: body={args.body} heads={args.heads}")
