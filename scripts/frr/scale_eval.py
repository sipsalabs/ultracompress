"""
Model-agnostic FRR eval.

Auto-detects the teacher from the checkpoint's 'teacher_cache' field.
Supports both:
  - 1000-sample in-domain eval from fineweb tail (protocol matches hires_eval)
  - 1000-sample truly-disjoint eval on WikiText-103 test

Usage:
    python scale_eval.py --tags generic_0.6b_h64 --corpus wikitext
    python scale_eval.py --tags generic_1.7b_h256 generic_0.6b_h64 \
        --corpus fineweb_tail --n 1000 --device cuda:1
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultracompress.moonshot import FractalModel
from scaling.teacher_loader import load_qwen3_teacher

ap = argparse.ArgumentParser()
ap.add_argument('--tags', nargs='+', required=True)
ap.add_argument('--corpus', choices=['fineweb_tail', 'wikitext'], default='wikitext',
                help='wikitext = truly-disjoint WT103 test; fineweb_tail = in-domain')
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--data', type=str, default='fineweb_edu_500M_tokens.pt')
ap.add_argument('--tail_tokens', type=int, default=50_000_000)
ap.add_argument('--wikitext_cache', type=str, default='wikitext103_test_qwen3.pt')
ap.add_argument('--out_prefix', type=str, default='scale_eval_results')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_SAMPLES = args.n


# ---------- data ----------
if args.corpus == 'wikitext':
    if os.path.exists(args.wikitext_cache):
        all_tokens = torch.load(args.wikitext_cache, weights_only=True)
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        text = '\n\n'.join([r['text'] for r in ds if r['text'].strip()])
        ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        all_tokens = ids.to(torch.int32)
        torch.save(all_tokens, args.wikitext_cache)
    tail_start = 0
    total = all_tokens.numel()
    print(f"CORPUS=wikitext103-test  total={total/1e3:.1f}K tokens")
else:
    all_tokens = torch.load(args.data, weights_only=True)
    total = all_tokens.numel()
    tail_start = max(0, total - args.tail_tokens)
    print(f"CORPUS=fineweb_tail  total={total/1e6:.0f}M  tail_start={tail_start/1e6:.0f}M")

g = torch.Generator().manual_seed(args.seed)
starts = torch.randint(tail_start, total - SEQ_LEN - 1, (N_SAMPLES,), generator=g)


# ---------- student class (must match training) ----------
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
        x = self.embed(tokens).float()
        x = self.proj_in(x)
        fr = self.inner
        for s in range(fr.n_scales):
            gamma, beta = fr.scale_gamma[s], fr.scale_beta[s]
            for it in range(fr.iters_per_scale):
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * fr.iter_scale[s, it]
        x = self.proj_out(x)
        x = self.norm_outer(x)
        return F.linear(x, self.lm_head_w)


def bootstrap_ci(values, n_boot=1000, conf=0.95):
    v = torch.tensor(values, dtype=torch.float32)
    n = v.numel()
    g = torch.Generator().manual_seed(0)
    boot = torch.empty(n_boot)
    for i in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        boot[i] = v[idx].mean()
    lo = torch.quantile(boot, (1 - conf) / 2).item()
    hi = torch.quantile(boot, 1 - (1 - conf) / 2).item()
    return lo, hi


def find_ckpt(tag):
    """Support both legacy and new ckpt directory layouts."""
    for candidate in [
        f'checkpoints_1.7b_tinyfrr_{tag}/best.pt',
        f'checkpoints_1.7b_baseline_kd_{tag}/best.pt',
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


@torch.no_grad()
def eval_one(tag):
    ckpt_path = find_ckpt(tag)
    if ckpt_path is None:
        print(f"!!! skip {tag}: no checkpoint found")
        return None
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    teacher_cache = ck.get('teacher_cache', 'qwen3_1.7b_cache.pt')
    tb = load_qwen3_teacher(teacher_cache, device=DEVICE, verbose=True)
    teacher = tb.teacher
    H_OUTER = tb.h_outer
    vocab_size = tb.vocab_size
    N_TEACHER_LAYERS = tb.n_layers

    h_inner = ck['h_inner']
    n_heads_inner = ck.get('n_heads_inner', 16)
    model = TinyFRR(H_OUTER, h_inner, n_heads_inner, vocab_size,
                    tb.embed_w, tb.lm_head_w, tb.norm_w).to(DEVICE)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.eval()

    print(f"\n>>> {tag}  ckpt={ckpt_path}  teacher={teacher_cache}  "
          f"h_inner={h_inner}  h_outer={H_OUTER}")

    all_t1_list, all_t10_list = [], []
    last_t1_list, last_t10_list = [], []
    teacher_nll_sum = student_nll_sum = 0.0
    n_tok = 0
    t0 = time.time()
    for i, s in enumerate(starts):
        s = int(s.item())
        toks = all_tokens[s:s + SEQ_LEN + 1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]
        tgt = toks[0, 1:SEQ_LEN + 1]
        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = model(inp)
        samp_t1 = 0
        samp_t10 = 0.0
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            samp_t1 += int(st[0] == tt[0])
            samp_t10 += len(set(tt.tolist()) & set(st.tolist())) / 10.0
        all_t1_list.append(samp_t1 / SEQ_LEN)
        all_t10_list.append(samp_t10 / SEQ_LEN)
        tt_last = tl[0, -1].topk(10).indices
        st_last = sl[0, -1].topk(10).indices
        last_t1_list.append(int(st_last[0] == tt_last[0]))
        last_t10_list.append(len(set(tt_last.tolist()) & set(st_last.tolist())) / 10.0)
        teacher_nll_sum += -F.log_softmax(tl[0], -1).gather(-1, tgt.unsqueeze(-1)).sum().item()
        student_nll_sum += -F.log_softmax(sl[0], -1).gather(-1, tgt.unsqueeze(-1)).sum().item()
        n_tok += SEQ_LEN
        if (i + 1) % 100 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"  [{tag}] {i+1}/{N_SAMPLES}  ({r:.1f}/s)")

    all_t1 = sum(all_t1_list) / len(all_t1_list)
    all_t10 = sum(all_t10_list) / len(all_t10_list)
    last_t1 = sum(last_t1_list) / len(last_t1_list)
    last_t10 = sum(last_t10_list) / len(last_t10_list)
    t_ppl = float(torch.tensor(teacher_nll_sum / n_tok).exp())
    s_ppl = float(torch.tensor(student_nll_sum / n_tok).exp())
    ratio = s_ppl / t_ppl
    quality = 0.5 * all_t10 + 0.5 * (1.0 / ratio)
    at1_ci = bootstrap_ci(all_t1_list)
    at10_ci = bootstrap_ci(all_t10_list)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    comp = ck.get('compression_ratio',
                  ck.get('teacher_total_params', 0) / n_params if n_params else 0)

    print(f"\n  [{tag}] corpus={args.corpus}  teacher={os.path.basename(teacher_cache)}")
    print(f"    trainable={n_params/1e6:.3f}M  compression={comp:.1f}x")
    print(f"    all T1   = {all_t1*100:.2f}%  95% CI [{at1_ci[0]*100:.2f}, {at1_ci[1]*100:.2f}]")
    print(f"    all T10  = {all_t10*100:.2f}%  95% CI [{at10_ci[0]*100:.2f}, {at10_ci[1]*100:.2f}]")
    print(f"    last T1  = {last_t1*100:.2f}%")
    print(f"    last T10 = {last_t10*100:.2f}%")
    print(f"    teacher_ppl={t_ppl:.3f}  student_ppl={s_ppl:.3f}  ratio={ratio:.3f}")
    print(f"    quality = {quality*100:.2f}")

    del model, teacher, tb
    torch.cuda.empty_cache()

    return {
        'tag': tag, 'ckpt': ckpt_path,
        'teacher_cache': teacher_cache,
        'corpus': args.corpus,
        'trainable_params': n_params,
        'compression_ratio': comp,
        'all_t1': all_t1, 'all_t10': all_t10,
        'last_t1': last_t1, 'last_t10': last_t10,
        'at1_ci': at1_ci, 'at10_ci': at10_ci,
        'teacher_ppl': t_ppl, 'student_ppl': s_ppl, 'ppl_ratio': ratio,
        'quality': quality,
    }


results = {}
for tag in args.tags:
    r = eval_one(tag)
    if r is not None:
        results[tag] = r

outpath = f'{args.out_prefix}_{args.corpus}.json'
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved -> {outpath}")
