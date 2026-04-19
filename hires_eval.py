"""
Hires eval for HQ5 / HQ6 / HQ7 flagship checkpoints.

1000 stratified samples, seed 42, SEQ_LEN=128 matching training protocol.
Eval positions are drawn from the TAIL of fineweb_edu_500M_tokens.pt
(last 50M tokens) — the least-touched region during training, giving a
quasi-held-out read on flagship model quality.

Reports:
  - Top-1 agreement with teacher (all positions, last position)
  - Top-10 agreement with teacher (all positions, last position)
  - Quality = 0.5 * all_T10 + 0.5 * (1 / ppl_ratio) * 100, matching training EVAL
  - Entropy-stratified buckets (conf <2 bits, mid 2-5, unsure >5)
  - 95% bootstrap confidence intervals on primary metrics
  - Saved to hires_results.pt and hires_results.json

Usage:
    python hires_eval.py --tags hq5_h256 hq5_h128 --n 1000
    python hires_eval.py --tags hq6_h256 hq6_h384 --n 1000 --device cuda:0
"""
import sys
import os
import math
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--tags', nargs='+', required=True,
                help='TinyFRR tags to eval, e.g. hq5_h256 hq5_h128')
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--data', type=str, default='fineweb_edu_500M_tokens.pt')
ap.add_argument('--tail_tokens', type=int, default=50_000_000,
                help='Draw eval starts from the last N tokens of the data file')
ap.add_argument('--out_prefix', type=str, default='hires_results')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_TEACHER_LAYERS = 28
N_SAMPLES = args.n

print(f"HIRES EVAL  tags={args.tags}  n={N_SAMPLES}  seq_len={SEQ_LEN}  seed={args.seed}")

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

# ---------- data ----------
all_tokens = torch.load(args.data, weights_only=True)
total = all_tokens.numel()
tail_start = max(0, total - args.tail_tokens)
print(f"  data: {args.data}  total={total/1e6:.0f}M  tail_start={tail_start/1e6:.0f}M")

g = torch.Generator().manual_seed(args.seed)
starts = torch.randint(tail_start, total - SEQ_LEN - 1, (N_SAMPLES,), generator=g)


# ---------- student ----------
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
        for p in self.inner.embed.parameters():
            p.requires_grad = False
        for p in self.inner.lm_head.parameters():
            p.requires_grad = False
        for p in self.inner.norm.parameters():
            p.requires_grad = False
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
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h_inner = ck['h_inner']
    n_heads_inner = ck.get('n_heads_inner', 16)
    m = TinyFRR(H_OUTER, h_inner, n_heads_inner, vocab_size,
                embed_w, lm_head_w, norm_w).to(DEVICE)
    m.load_state_dict(ck['state_dict'], strict=False)
    m.eval()
    return m, ck


def bootstrap_ci(values, n_boot=1000, conf=0.95):
    v = torch.tensor(values, dtype=torch.float32)
    n = v.numel()
    g2 = torch.Generator().manual_seed(0)
    boot = torch.empty(n_boot)
    for i in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g2)
        boot[i] = v[idx].mean()
    lo = torch.quantile(boot, (1 - conf) / 2).item()
    hi = torch.quantile(boot, 1 - (1 - conf) / 2).item()
    return lo, hi


@torch.no_grad()
def eval_one(model, tag):
    print(f"\n>>> {tag}")
    all_t1_list = []   # per-sample all-position T1 rate
    all_t10_list = []
    last_t1_list = []  # per-sample last-token T1 hit (0 or 1)
    last_t10_list = [] # per-sample last-token T10 overlap
    teacher_nll_sum = 0.0
    student_nll_sum = 0.0
    n_tok = 0
    buckets = {'conf(<2)': {'t1': 0, 't10': 0, 'n': 0},
               'mid(2-5)': {'t1': 0, 't10': 0, 'n': 0},
               'unsure(>5)': {'t1': 0, 't10': 0, 'n': 0}}

    t0 = time.time()
    for i, s in enumerate(starts):
        s = int(s.item())
        toks = all_tokens[s:s + SEQ_LEN + 1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]
        tgt = toks[0, 1:SEQ_LEN + 1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = model(inp)

        t_prob = F.softmax(tl[0], dim=-1)
        ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(-1) / 0.6931  # bits

        samp_t1 = 0
        samp_t10 = 0.0
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            h1 = int(st[0] == tt[0])
            h10 = len(set(tt.tolist()) & set(st.tolist())) / 10.0
            samp_t1 += h1
            samp_t10 += h10
            e = ent[pos].item()
            k = 'conf(<2)' if e < 2 else ('mid(2-5)' if e < 5 else 'unsure(>5)')
            buckets[k]['t1'] += h1
            buckets[k]['t10'] += h10
            buckets[k]['n'] += 1
        all_t1_list.append(samp_t1 / SEQ_LEN)
        all_t10_list.append(samp_t10 / SEQ_LEN)

        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        last_t1_list.append(int(st[0] == tt[0]))
        last_t10_list.append(len(set(tt.tolist()) & set(st.tolist())) / 10.0)

        # Perplexity on ground-truth next-token
        teacher_nll_sum += F.cross_entropy(tl[0], tgt, reduction='sum').item()
        student_nll_sum += F.cross_entropy(sl[0], tgt, reduction='sum').item()
        n_tok += SEQ_LEN

        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (N_SAMPLES - i - 1) / rate
            print(f"  {i+1}/{N_SAMPLES}  ({rate:.1f}/s, eta {eta/60:.1f}m)")

    all_t1 = sum(all_t1_list) / len(all_t1_list)
    all_t10 = sum(all_t10_list) / len(all_t10_list)
    last_t1 = sum(last_t1_list) / len(last_t1_list)
    last_t10 = sum(last_t10_list) / len(last_t10_list)
    teacher_ppl = math.exp(teacher_nll_sum / n_tok)
    student_ppl = math.exp(student_nll_sum / n_tok)
    ppl_ratio = student_ppl / teacher_ppl
    quality = 0.5 * all_t10 * 100 + 0.5 * (1.0 / ppl_ratio) * 100

    all_t1_lo, all_t1_hi = bootstrap_ci(all_t1_list)
    all_t10_lo, all_t10_hi = bootstrap_ci(all_t10_list)
    last_t1_lo, last_t1_hi = bootstrap_ci(last_t1_list)
    last_t10_lo, last_t10_hi = bootstrap_ci(last_t10_list)

    print(f"\n  [{tag}] RESULTS  (n={N_SAMPLES}, {SEQ_LEN*N_SAMPLES} tokens)")
    print(f"    all T1   = {all_t1*100:.2f}%  95% CI [{all_t1_lo*100:.2f}, {all_t1_hi*100:.2f}]")
    print(f"    all T10  = {all_t10*100:.2f}%  95% CI [{all_t10_lo*100:.2f}, {all_t10_hi*100:.2f}]")
    print(f"    last T1  = {last_t1*100:.2f}%  95% CI [{last_t1_lo*100:.2f}, {last_t1_hi*100:.2f}]")
    print(f"    last T10 = {last_t10*100:.2f}%  95% CI [{last_t10_lo*100:.2f}, {last_t10_hi*100:.2f}]")
    print(f"    teacher ppl = {teacher_ppl:.3f}  student ppl = {student_ppl:.3f}  ratio = {ppl_ratio:.3f}")
    print(f"    quality  = {quality:.2f}%")
    for k, b in buckets.items():
        if b['n']:
            print(f"    {k:12s} T1={b['t1']/b['n']*100:.2f}%  T10={b['t10']/b['n']*100:.2f}%  (n={b['n']})")

    return {
        'tag': tag,
        'n': N_SAMPLES,
        'seq_len': SEQ_LEN,
        'seed': args.seed,
        'all_t1': all_t1, 'all_t1_ci': (all_t1_lo, all_t1_hi),
        'all_t10': all_t10, 'all_t10_ci': (all_t10_lo, all_t10_hi),
        'last_t1': last_t1, 'last_t1_ci': (last_t1_lo, last_t1_hi),
        'last_t10': last_t10, 'last_t10_ci': (last_t10_lo, last_t10_hi),
        'teacher_ppl': teacher_ppl,
        'student_ppl': student_ppl,
        'ppl_ratio': ppl_ratio,
        'quality': quality,
        'buckets': {k: {'t1': b['t1'] / max(b['n'], 1),
                        't10': b['t10'] / max(b['n'], 1),
                        'n': b['n']} for k, b in buckets.items()},
    }


results = {}
for tag in args.tags:
    r = load_ckpt(tag)
    if r is None:
        print(f"\nSKIP {tag} — no checkpoint at checkpoints_1.7b_tinyfrr_{tag}/best.pt")
        continue
    model, ck = r
    trainable = ck.get('trainable', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"  {tag}  h_inner={ck['h_inner']}  trainable={trainable}  step={ck.get('step', '?')}")
    results[tag] = eval_one(model, tag)
    results[tag]['trainable'] = int(trainable)
    results[tag]['best_step'] = int(ck.get('step', 0))
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'tag':14s} {'trainable':>10s} {'allT1':>7s} {'allT10':>7s} {'lastT1':>7s} {'lastT10':>8s} {'pplR':>6s} {'Q':>6s}")
for tag, r in results.items():
    print(f"{tag:14s} {r['trainable']:>10d} "
          f"{r['all_t1']*100:6.2f}% {r['all_t10']*100:6.2f}% "
          f"{r['last_t1']*100:6.2f}% {r['last_t10']*100:7.2f}% "
          f"{r['ppl_ratio']:6.3f} {r['quality']:5.2f}%")

torch.save(results, f'{args.out_prefix}.pt')
json_results = {tag: {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in r.items() if k != 'buckets'}
                for tag, r in results.items()}
for tag, r in results.items():
    json_results[tag]['buckets'] = r['buckets']
with open(f'{args.out_prefix}.json', 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"\nSaved: {args.out_prefix}.pt  {args.out_prefix}.json")
