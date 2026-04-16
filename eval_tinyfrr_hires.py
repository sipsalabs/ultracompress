"""
Hires eval for TinyFRR checkpoints.
Evaluates all available tags on 1000 stratified samples.
Identical to eval_hmk_hires.py but reconstructs the TinyFRR wrapper.
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
ap.add_argument('--tags', nargs='+', default=['h1024', 'h768', 'h512', 'h384', 'h256', 'h192', 'h128'])
ap.add_argument('--n', type=int, default=500)
args = ap.parse_args()

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = args.n

print(f"TinyFRR hires eval: tags={args.tags}  n={N_SAMPLES}")

# Teacher
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


@torch.no_grad()
def eval_model(model, tag):
    all_t1 = all_t10 = 0
    last_t1 = last_t10 = 0
    buckets = {'conf(<2)': {'t1': 0, 't10': 0, 'n': 0},
               'mid(2-5)': {'t1': 0, 't10': 0, 'n': 0},
               'unsure(>5)': {'t1': 0, 't10': 0, 'n': 0}}
    tot = 0
    for i, s in enumerate(starts):
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        tl = teacher.forward(toks, max_layers=N_TEACHER_LAYERS)
        sl = model(toks)
        t_prob = F.softmax(tl[0], dim=-1)
        ent = -(t_prob * t_prob.clamp_min(1e-12).log()).sum(-1) / 0.6931
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            h1 = int(st[0] == tt[0])
            h10 = len(set(tt.tolist()) & set(st.tolist())) / 10
            all_t1 += h1; all_t10 += h10; tot += 1
            e = ent[pos].item()
            k = 'conf(<2)' if e < 2 else ('mid(2-5)' if e < 5 else 'unsure(>5)')
            buckets[k]['t1'] += h1; buckets[k]['t10'] += h10; buckets[k]['n'] += 1
        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        last_t1 += int(st[0] == tt[0])
        last_t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
        if (i+1) % 100 == 0:
            print(f"  [{tag}] {i+1}/{N_SAMPLES}")
    print(f"\n  {tag}  all: T1={all_t1/tot*100:.2f}% T10={all_t10/tot*100:.2f}% | last: T1={last_t1/N_SAMPLES*100:.2f}% T10={last_t10/N_SAMPLES*100:.2f}%")
    for k, b in buckets.items():
        if b['n']:
            print(f"    {k:12s} T1={b['t1']/b['n']*100:.2f}% T10={b['t10']/b['n']*100:.2f}% (n={b['n']})")
    return {
        'all': (all_t1/tot, all_t10/tot),
        'last': (last_t1/N_SAMPLES, last_t10/N_SAMPLES),
        'buckets': {k: (b['t1']/max(b['n'],1), b['t10']/max(b['n'],1), b['n']) for k, b in buckets.items()}
    }


results = {}
for tag in args.tags:
    r = load_ckpt(tag)
    if r is None:
        print(f"skip {tag} (no checkpoint)")
        continue
    model, ck = r
    print(f"\n>>> {tag}  h_inner={ck['h_inner']}  trainable={ck.get('trainable','?')}")
    results[tag] = eval_model(model, tag)
    del model; torch.cuda.empty_cache()

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"{'tag':8s} {'all-T10':>8s} {'last-T10':>9s} {'conf-T10':>9s} {'mid-T10':>9s} {'unsure-T10':>11s}")
for tag, r in results.items():
    b = r['buckets']
    print(f"{tag:8s} {r['all'][1]*100:7.2f}% {r['last'][1]*100:8.2f}% {b['conf(<2)'][1]*100:8.2f}% {b['mid(2-5)'][1]*100:8.2f}% {b['unsure(>5)'][1]*100:10.2f}%")

torch.save(results, 'tinyfrr_hires_results.pt')
print("\nSaved: tinyfrr_hires_results.pt")
