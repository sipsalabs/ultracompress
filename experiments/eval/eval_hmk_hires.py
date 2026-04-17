"""
Hires eval for HMK. Waits for checkpoint, then runs 1000-sample stratified eval
against baseline FRR-100K. Report deltas by entropy bucket.

Usage: python eval_hmk_hires.py [--ckpt path]
Default ckpt: checkpoints_1.7b_hmk/hmk_best.pt
"""
import lib.unbuffered
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer
from ultracompress.moonshot import FractalModel

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='checkpoints_1.7b_hmk/hmk_best.pt')
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--base', default='checkpoints_1.7b_real_text/frr_1.7b_100k_final.pt')
args = ap.parse_args()

DEVICE = 'cuda:0'
SEQ_LEN = 64
N_TEACHER_LAYERS = 28
N_SAMPLES = args.n

print("=" * 72)
print(f"HMK hires eval: {args.ckpt} vs {args.base}")
print(f"  {N_SAMPLES} samples stratified")
print("=" * 72)

# Teacher
print("Loading teacher...")
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
hidden = gd['token_embd.weight'].shape[1]
vocab_size = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=hidden, intermediate_size=hidden*3,
                  vocab_size=vocab_size, head_dim=hidden//16)
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


def make_student(ckpt_path):
    raw = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    if isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']; ff_mult = raw.get('ff_mult', 1)
    else:
        sd = raw; ff_mult = 1
    m = FractalModel(
        hidden_dim=hidden, n_heads=16,
        n_scales=4, iters_per_scale=7,
        vocab_size=vocab_size, ff_mult=ff_mult,
        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w,
    ).to(DEVICE)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


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
        ent = -(t_prob * (t_prob.clamp_min(1e-12)).log()).sum(-1) / 0.6931
        for pos in range(SEQ_LEN):
            tt = tl[0, pos].topk(10).indices
            st = sl[0, pos].topk(10).indices
            h1 = int(st[0] == tt[0])
            h10 = len(set(tt.tolist()) & set(st.tolist())) / 10
            all_t1 += h1; all_t10 += h10; tot += 1
            e = ent[pos].item()
            k = 'conf(<2)' if e < 2 else ('mid(2-5)' if e < 5 else 'unsure(>5)')
            buckets[k]['t1'] += h1
            buckets[k]['t10'] += h10
            buckets[k]['n'] += 1
        tt = tl[0, -1].topk(10).indices
        st = sl[0, -1].topk(10).indices
        last_t1 += int(st[0] == tt[0])
        last_t10 += len(set(tt.tolist()) & set(st.tolist())) / 10
        if (i+1) % 200 == 0:
            print(f"  [{tag}] {i+1}/{N_SAMPLES}")
    print(f"\n  {tag}")
    print(f"    all-pos  T1={all_t1/tot*100:5.2f}%  T10={all_t10/tot*100:5.2f}%  (n={tot})")
    print(f"    last-tok T1={last_t1/N_SAMPLES*100:5.2f}%  T10={last_t10/N_SAMPLES*100:5.2f}%")
    for k, b in buckets.items():
        if b['n']:
            print(f"    {k:12s} T1={b['t1']/b['n']*100:5.2f}%  T10={b['t10']/b['n']*100:5.2f}%  (n={b['n']})")
    return {
        'all': (all_t1/tot, all_t10/tot),
        'last': (last_t1/N_SAMPLES, last_t10/N_SAMPLES),
        'buckets': {k: (b['t1']/max(b['n'],1), b['t10']/max(b['n'],1), b['n']) for k, b in buckets.items()}
    }


print("\n>>> baseline FRR-100K...")
base = make_student(args.base)
rb = eval_model(base, 'base')
del base; torch.cuda.empty_cache()

print(f"\n>>> HMK ({args.ckpt})...")
hmk = make_student(args.ckpt)
rh = eval_model(hmk, 'HMK')
del hmk; torch.cuda.empty_cache()

print("\n" + "=" * 72)
print("DELTA (HMK - baseline):")
print("=" * 72)
print(f"  all-pos  deltaT1={(rh['all'][0]-rb['all'][0])*100:+.2f}pp  deltaT10={(rh['all'][1]-rb['all'][1])*100:+.2f}pp")
print(f"  last-tok deltaT1={(rh['last'][0]-rb['last'][0])*100:+.2f}pp  deltaT10={(rh['last'][1]-rb['last'][1])*100:+.2f}pp")
for k in rb['buckets']:
    bb = rb['buckets'][k]; bc = rh['buckets'][k]
    print(f"  {k:12s} deltaT1={(bc[0]-bb[0])*100:+.2f}pp  deltaT10={(bc[1]-bb[1])*100:+.2f}pp")

torch.save({'base': rb, 'hmk': rh, 'ckpt': args.ckpt}, 'hmk_hires_results.pt')
print("\nSaved: hmk_hires_results.pt")
