"""
Combined-stack eval: HQ5 body + ASVD head, end-to-end.

Measures what HAPPENS WHEN YOU ACTUALLY USE THE COMPRESSED MODEL to generate
next-token logits — the number every reviewer, investor, and patent examiner
will ask for first.

Pipeline:  tokens -> teacher_embed -> proj_in -> FRR body -> proj_out
                  -> norm_outer -> ASVD proj -> ASVD out -> logits

1000 samples, seed 42, SEQ_LEN=128, eval positions from tail 50M of
fineweb_edu_500M_tokens.pt (least-touched during training).
Reports T1/T10/PPL-ratio/quality with 95% bootstrap CIs + total param count +
effective compression ratio vs teacher.

Usage:
    python combined_stack_eval.py --body hq5_h256 --heads asvd_r1024_ft asvd_r512_ft asvd_r256_ft \
        --n 1000 --device cuda:1
"""
import sys
import os
import math
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--body', type=str, required=True,
                help='TinyFRR body tag, e.g. hq5_h256')
ap.add_argument('--heads', nargs='+', required=True,
                help='ASVD head tags, e.g. asvd_r1024_ft asvd_r512_ft')
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--device', type=str, default='cuda:1')
ap.add_argument('--data', type=str, default='fineweb_edu_500M_tokens.pt')
ap.add_argument('--tail_tokens', type=int, default=50_000_000)
ap.add_argument('--out_prefix', type=str, default='combined_stack_results')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_TEACHER_LAYERS = 28

print(f"COMBINED STACK EVAL  body={args.body}  heads={args.heads}  n={args.n}")

# ---------- teacher ----------
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
        if k in wd:
            gd[f'blk.{li}.{g}'] = wd[k].float()
del wd
H_OUTER = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=H_OUTER, intermediate_size=H_OUTER * 3,
                  vocab_size=vocab_size, head_dim=H_OUTER // 16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)
embed_w = gd['token_embd.weight'].to(DEVICE)
norm_w = gd['output_norm.weight'].to(DEVICE)
lm_head_w = gd['output.weight'].to(DEVICE)
del gd
teacher_total_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + lm_head_w.numel() + embed_w.numel()

# ---------- data ----------
all_tokens = torch.load(args.data, weights_only=True)
total = all_tokens.numel()
tail_start = max(0, total - args.tail_tokens)
print(f"  data: tail_start={tail_start/1e6:.0f}M  total={total/1e6:.0f}M")

g = torch.Generator().manual_seed(args.seed)
starts = torch.randint(tail_start, total - SEQ_LEN - 1, (args.n,), generator=g)


# ---------- body ----------
body_path = f'checkpoints_1.7b_tinyfrr_{args.body}/best.pt'
if not os.path.exists(body_path):
    raise SystemExit(f"body checkpoint missing: {body_path}")
print(f"\nLoading body: {body_path}")
body_ck = torch.load(body_path, map_location='cpu', weights_only=False)
h_inner = body_ck['h_inner']
n_heads_inner = body_ck.get('n_heads_inner', next((h for h in [16, 8, 12, 4] if h_inner % h == 0), 4))
print(f"  h_inner={h_inner}  n_heads={n_heads_inner}  best_step={body_ck.get('step', '?')}")

proj_in = nn.Linear(H_OUTER, h_inner, bias=False).to(DEVICE)
proj_out = nn.Linear(h_inner, H_OUTER, bias=False).to(DEVICE)
norm_outer = nn.RMSNorm(H_OUTER).to(DEVICE)
inner = FractalModel(
    hidden_dim=h_inner, n_heads=n_heads_inner,
    n_scales=4, iters_per_scale=7, vocab_size=vocab_size, ff_mult=1,
    embed_weight=None, lm_head_weight=None, norm_weight=None,
).to(DEVICE)

sd = body_ck['state_dict']
proj_in.weight.data.copy_(sd['proj_in.weight'])
proj_out.weight.data.copy_(sd['proj_out.weight'])
if 'norm_outer.weight' in sd:
    norm_outer.weight.data.copy_(sd['norm_outer.weight'])
else:
    norm_outer.weight.data.copy_(norm_w.cpu())

for k, v in sd.items():
    if k.startswith('inner.'):
        inner_key = k[6:]
        obj = inner
        parts = inner_key.split('.')
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        final = parts[-1]
        if final.isdigit():
            obj[int(final)].data.copy_(v)
        elif hasattr(getattr(obj, final, None), 'data'):
            getattr(obj, final).data.copy_(v)
        else:
            setattr(obj, final, nn.Parameter(v.to(DEVICE)))

body_params = sum(p.numel() for n, p in list(proj_in.named_parameters())
                  + list(proj_out.named_parameters()) + list(inner.named_parameters())
                  if 'embed' not in n and 'lm_head' not in n and 'norm.' not in n)
print(f"  body trainable params: {body_params/1e6:.3f}M")


@torch.no_grad()
def run_body(tokens):
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


def bootstrap_ci(values, n_boot=1000, conf=0.95):
    v = torch.tensor(values, dtype=torch.float32)
    n = v.numel()
    g2 = torch.Generator().manual_seed(0)
    boot = torch.empty(n_boot)
    for i in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g2)
        boot[i] = v[idx].mean()
    return torch.quantile(boot, (1 - conf) / 2).item(), torch.quantile(boot, 1 - (1 - conf) / 2).item()


@torch.no_grad()
def eval_stack(logits_fn, tag, total_params):
    t0 = time.time()
    all_t1_list, all_t10_list = [], []
    last_t1_list, last_t10_list = [], []
    teacher_nll_sum = student_nll_sum = 0.0
    n_tok = 0
    for i, s in enumerate(starts):
        s = int(s.item())
        toks = all_tokens[s:s + SEQ_LEN + 1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]
        tgt = toks[0, 1:SEQ_LEN + 1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = logits_fn(inp)

        samp_t1 = 0
        samp_t10 = 0.0
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            samp_t1 += int(st[0] == tt[0])
            samp_t10 += len(set(tt.tolist()) & set(st.tolist())) / 10.0
        all_t1_list.append(samp_t1 / SEQ_LEN)
        all_t10_list.append(samp_t10 / SEQ_LEN)

        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        last_t1_list.append(int(st[0] == tt[0]))
        last_t10_list.append(len(set(tt.tolist()) & set(st.tolist())) / 10.0)

        teacher_nll_sum += F.cross_entropy(tl[0], tgt, reduction='sum').item()
        student_nll_sum += F.cross_entropy(sl[0], tgt, reduction='sum').item()
        n_tok += SEQ_LEN

        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  [{tag}] {i+1}/{args.n}  ({(i+1)/dt:.1f}/s)", flush=True)

    all_t1 = sum(all_t1_list) / len(all_t1_list)
    all_t10 = sum(all_t10_list) / len(all_t10_list)
    last_t1 = sum(last_t1_list) / len(last_t1_list)
    last_t10 = sum(last_t10_list) / len(last_t10_list)
    t_ppl = math.exp(teacher_nll_sum / n_tok)
    s_ppl = math.exp(student_nll_sum / n_tok)
    ppl_ratio = s_ppl / t_ppl
    quality = 0.5 * all_t10 * 100 + 0.5 * (1.0 / ppl_ratio) * 100

    r = {
        'tag': tag,
        'total_params': int(total_params),
        'compression': teacher_total_params / total_params,
        'all_t1': all_t1, 'all_t1_ci': bootstrap_ci(all_t1_list),
        'all_t10': all_t10, 'all_t10_ci': bootstrap_ci(all_t10_list),
        'last_t1': last_t1, 'last_t1_ci': bootstrap_ci(last_t1_list),
        'last_t10': last_t10, 'last_t10_ci': bootstrap_ci(last_t10_list),
        'teacher_ppl': t_ppl, 'student_ppl': s_ppl, 'ppl_ratio': ppl_ratio,
        'quality': quality,
    }
    print(f"\n  [{tag}]")
    print(f"    total_params = {total_params/1e6:.2f}M   comp = {r['compression']:.1f}x (vs teacher)")
    print(f"    all T1   = {all_t1*100:.2f}%  95% CI [{r['all_t1_ci'][0]*100:.2f}, {r['all_t1_ci'][1]*100:.2f}]")
    print(f"    all T10  = {all_t10*100:.2f}%  95% CI [{r['all_t10_ci'][0]*100:.2f}, {r['all_t10_ci'][1]*100:.2f}]")
    print(f"    last T1  = {last_t1*100:.2f}%  95% CI [{r['last_t1_ci'][0]*100:.2f}, {r['last_t1_ci'][1]*100:.2f}]")
    print(f"    last T10 = {last_t10*100:.2f}%  95% CI [{r['last_t10_ci'][0]*100:.2f}, {r['last_t10_ci'][1]*100:.2f}]")
    print(f"    ppl_ratio = {ppl_ratio:.3f}   quality = {quality:.2f}%")
    return r


# ---------- baseline: body + full lm_head ----------
print(f"\n{'='*80}\nbody + FULL lm_head (baseline, head uncompressed)")
full_fn = lambda inp: F.linear(run_body(inp), lm_head_w)
baseline = eval_stack(full_fn, f'{args.body}+full_head', body_params + lm_head_w.numel())
results = {'baseline': baseline}

# ---------- body + each ASVD head ----------
for head_tag in args.heads:
    head_path = f'checkpoints_1.7b_{head_tag}/best.pt'
    if not os.path.exists(head_path):
        print(f"\nSKIP {head_tag}: no checkpoint at {head_path}")
        continue
    print(f"\n{'='*80}\nLoading head: {head_path}")
    hck = torch.load(head_path, map_location='cpu', weights_only=False)
    if 'state_dict' in hck:
        hsd = hck['state_dict']
        asvd_proj_w = hsd['asvd_proj.weight'].to(DEVICE)
        asvd_out_w = hsd['asvd_out.weight'].to(DEVICE)
    else:
        asvd_proj_w = hck['asvd_proj'].to(DEVICE)
        asvd_out_w = hck['asvd_out'].to(DEVICE)
    rank = asvd_proj_w.shape[0]
    head_params = asvd_proj_w.numel() + asvd_out_w.numel()
    print(f"  ASVD rank={rank}  head_params={head_params/1e6:.2f}M")

    def make_fn(pw, ow):
        def fn(inp):
            latent = run_body(inp)
            return F.linear(F.linear(latent, pw), ow)
        return fn

    tag = f'{args.body}+{head_tag}'
    r = eval_stack(make_fn(asvd_proj_w, asvd_out_w), tag, body_params + head_params)
    r['rank'] = rank
    r['body_params'] = body_params
    r['head_params'] = head_params
    results[head_tag] = r
    del asvd_proj_w, asvd_out_w, hck
    torch.cuda.empty_cache()

# ---------- summary ----------
print(f"\n{'='*100}")
print(f"COMBINED STACK SUMMARY  (body={args.body}, h_inner={h_inner}, n={args.n})")
print(f"{'='*100}")
print(f"{'config':38s} {'params':>9s} {'comp':>7s} {'all T1':>8s} {'all T10':>9s} {'lastT10':>9s} {'pplR':>7s} {'Q':>7s}")
print('-' * 100)
print(f"{'teacher (Qwen3-1.7B)':38s} {teacher_total_params/1e6:>7.1f}M {'1.0x':>7s} {'100.00%':>8s} {'100.00%':>9s} {'100.00%':>9s} {'1.000':>7s} {'100.00%':>7s}")
ordered = [('baseline', results['baseline'])] + [(h, results[h]) for h in args.heads if h in results]
for tag, r in ordered:
    label = r['tag']
    print(f"{label:38s} {r['total_params']/1e6:>7.2f}M {r['compression']:>6.1f}x "
          f"{r['all_t1']*100:>7.2f}% {r['all_t10']*100:>8.2f}% {r['last_t10']*100:>8.2f}% "
          f"{r['ppl_ratio']:>7.3f} {r['quality']:>6.2f}%")
print('=' * 100)

torch.save(results, f'{args.out_prefix}.pt')
json_r = {}
for k, r in results.items():
    json_r[k] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in r.items()}
with open(f'{args.out_prefix}.json', 'w') as f:
    json.dump(json_r, f, indent=2)
print(f"\nSaved: {args.out_prefix}.pt  {args.out_prefix}.json")
