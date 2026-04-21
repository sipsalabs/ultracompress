"""
TRULY held-out eval on WikiText-103 test split.

Why this file exists:
  hires_eval.py evaluates on the TAIL of fineweb_edu_500M_tokens.pt. The
  training script (run_hq4_ceiling_break.py etc) samples training windows
  uniformly from the FULL 500M-token range, so the "tail" region is
  technically in-distribution for training (quasi-held-out). That is fine
  for relative comparison between runs, but it is not defensible for
  external technical due-diligence.

  WikiText-103 test split has ~245K tokens and was never touched during
  training. It is a standard, fully-public, fully-disjoint corpus. Any
  reviewer can reproduce our numbers on it in < 5 minutes.

Protocol: identical to hires_eval.py (seed 42, SEQ_LEN=128, bootstrap
95% CIs, entropy-stratified buckets) so numbers are directly comparable.

Usage:
    python wikitext_eval.py --tags hq5_h256 hq5_h128 --n 1000 --device cuda:1
"""
import sys
import os
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
ap.add_argument('--tags', nargs='+', required=True)
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--out_prefix', type=str, default='wikitext_results')
ap.add_argument('--cache', type=str, default='wikitext103_test_qwen3.pt')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_TEACHER_LAYERS = 28
N_SAMPLES = args.n


# ---------- WikiText-103 test tokens (cached) ----------
if os.path.exists(args.cache):
    print(f"Loading cached WikiText-103 test tokens: {args.cache}")
    all_tokens = torch.load(args.cache, weights_only=True)
else:
    print("Tokenizing WikiText-103 test split with Qwen3 tokenizer (one-time cache)...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    text = '\n\n'.join([row['text'] for row in ds if row['text'].strip()])
    ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    all_tokens = ids.to(torch.int32)
    torch.save(all_tokens, args.cache)
    print(f"  cached {all_tokens.numel()/1e3:.1f}K tokens -> {args.cache}")

total = all_tokens.numel()
print(f"WIKITEXT EVAL  tags={args.tags}  n={N_SAMPLES}  seq_len={SEQ_LEN}  seed={args.seed}  total={total/1e3:.1f}K")

# seed-42 random starts, same protocol as hires_eval
g = torch.Generator().manual_seed(args.seed)
starts = torch.randint(0, total - SEQ_LEN - 1, (N_SAMPLES,), generator=g)


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
    for h, g2 in hf_to_gguf.items():
        k = f'model.layers.{li}.{h}'
        if k in wd:
            gd[f'blk.{li}.{g2}'] = wd[k].float()
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
    all_t1_list, all_t10_list = [], []
    last_t1_list, last_t10_list = [], []
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
            if e < 2:
                b = 'conf(<2)'
            elif e < 5:
                b = 'mid(2-5)'
            else:
                b = 'unsure(>5)'
            buckets[b]['t1'] += h1
            buckets[b]['t10'] += h10
            buckets[b]['n'] += 1
        all_t1_list.append(samp_t1 / SEQ_LEN)
        all_t10_list.append(samp_t10 / SEQ_LEN)

        # last position metrics
        tt_last = tl[0, -1].topk(10).indices
        st_last = sl[0, -1].topk(10).indices
        last_t1_list.append(int(st_last[0] == tt_last[0]))
        last_t10_list.append(len(set(tt_last.tolist()) & set(st_last.tolist())) / 10.0)

        # perplexity contributions
        t_lp = F.log_softmax(tl[0], dim=-1)
        s_lp = F.log_softmax(sl[0], dim=-1)
        teacher_nll_sum += -t_lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        student_nll_sum += -s_lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        n_tok += SEQ_LEN

        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{tag}] {i+1}/{N_SAMPLES}  ({rate:.1f}/s)")

    all_t1 = sum(all_t1_list) / len(all_t1_list)
    all_t10 = sum(all_t10_list) / len(all_t10_list)
    last_t1 = sum(last_t1_list) / len(last_t1_list)
    last_t10 = sum(last_t10_list) / len(last_t10_list)
    t_ppl = float(torch.tensor(teacher_nll_sum / n_tok).exp())
    s_ppl = float(torch.tensor(student_nll_sum / n_tok).exp())
    ppl_ratio = s_ppl / t_ppl
    quality = 0.5 * all_t10 + 0.5 * (1.0 / ppl_ratio)

    # bootstrap CIs
    at1_lo, at1_hi = bootstrap_ci(all_t1_list)
    at10_lo, at10_hi = bootstrap_ci(all_t10_list)
    lt1_lo, lt1_hi = bootstrap_ci(last_t1_list)
    lt10_lo, lt10_hi = bootstrap_ci(last_t10_list)

    print(f"\n  [{tag}] WIKITEXT-103 held-out:")
    print(f"    all T1   = {all_t1*100:5.2f}%  95% CI [{at1_lo*100:.2f}, {at1_hi*100:.2f}]")
    print(f"    all T10  = {all_t10*100:5.2f}%  95% CI [{at10_lo*100:.2f}, {at10_hi*100:.2f}]")
    print(f"    last T1  = {last_t1*100:5.2f}%  95% CI [{lt1_lo*100:.2f}, {lt1_hi*100:.2f}]")
    print(f"    last T10 = {last_t10*100:5.2f}%  95% CI [{lt10_lo*100:.2f}, {lt10_hi*100:.2f}]")
    print(f"    teacher_ppl = {t_ppl:.3f}  student_ppl = {s_ppl:.3f}  ratio = {ppl_ratio:.3f}")
    print(f"    quality = {quality*100:.2f}")
    for k, b in buckets.items():
        if b['n'] > 0:
            print(f"    bucket {k}: T1={b['t1']/b['n']*100:5.2f}%  T10={b['t10']/b['n']*100:5.2f}%  (n={b['n']})")

    return {
        'tag': tag,
        'all_t1': all_t1, 'all_t10': all_t10,
        'last_t1': last_t1, 'last_t10': last_t10,
        'at1_ci': [at1_lo, at1_hi], 'at10_ci': [at10_lo, at10_hi],
        'lt1_ci': [lt1_lo, lt1_hi], 'lt10_ci': [lt10_lo, lt10_hi],
        'teacher_ppl': t_ppl, 'student_ppl': s_ppl, 'ppl_ratio': ppl_ratio,
        'quality': quality,
        'buckets': {k: {kk: (vv / b['n']) if kk != 'n' else vv
                        for kk, vv in b.items()}
                    for k, b in buckets.items()},
    }


# ---------- run ----------
results = {}
for tag in args.tags:
    out = load_ckpt(tag)
    if out is None:
        print(f"!!! skip {tag}: checkpoint not found")
        continue
    m, ck = out
    r = eval_one(m, tag)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    r['trainable_params'] = n_params
    r['step'] = ck.get('step', None)
    r['h_inner'] = ck['h_inner']
    results[tag] = r
    del m
    torch.cuda.empty_cache()

torch.save(results, f'{args.out_prefix}.pt')
with open(f'{args.out_prefix}.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved -> {args.out_prefix}.pt and {args.out_prefix}.json")
