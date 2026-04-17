"""
Evaluate TinyFRR + ASVD joint model checkpoints.

Supports both:
  - Regular TinyFRR (full lm_head): checkpoints_1.7b_tinyfrr_h{H}/best.pt
  - Joint body+ASVD head: checkpoints_1.7b_tinyfrr_h{H}_r{R}_joint/best.pt

Reports: PPL ratio, top-1 agreement, top-10 overlap, KL/token, total params,
         body params, head params, compression ratio.
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
ap.add_argument('--tags', nargs='+', required=True)
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seq_len', type=int, default=128)
ap.add_argument('--device', type=str, default='cuda:0')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = args.seq_len
N_TEACHER_LAYERS = 28

print(f"Combined evaluator: tags={args.tags}  n={args.n}  seq_len={SEQ_LEN}  device={DEVICE}")

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


# ==================== Model Classes ====================
class TinyFRR(nn.Module):
    """Regular TinyFRR with full lm_head."""
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7, vocab_size=vocab, ff_mult=1,
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
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x = self.proj_out(x)
        x = self.norm_outer(x)
        return F.linear(x, self.lm_head_w)


class TinyFRR_ASVD(nn.Module):
    """TinyFRR with ASVD factored lm_head."""
    def __init__(self, h_outer, h_inner, n_heads, vocab, rank, embed_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7, vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)
        self.asvd_proj = nn.Linear(h_outer, rank, bias=False)
        self.asvd_out = nn.Linear(rank, vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        x = self.proj_in(x)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x = self.proj_out(x)
        latent = self.norm_outer(x)
        z = self.asvd_proj(latent)
        return self.asvd_out(z)


class TinyFRR_HQ3(nn.Module):
    """TinyFRR with intermediate projections (HQ3)."""
    def __init__(self, h_outer, h_inner, n_heads, vocab, embed_w, lm_head_w, norm_w):
        super().__init__()
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.inner = FractalModel(
            hidden_dim=h_inner, n_heads=n_heads,
            n_scales=4, iters_per_scale=7, vocab_size=vocab, ff_mult=1,
            embed_weight=None, lm_head_weight=None, norm_weight=None,
        )
        for p in self.inner.embed.parameters(): p.requires_grad = False
        for p in self.inner.lm_head.parameters(): p.requires_grad = False
        for p in self.inner.norm.parameters(): p.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(embed_w, freeze=True)
        self.register_buffer('lm_head_w', lm_head_w, persistent=False)
        self.norm_outer = nn.RMSNorm(h_outer)
        self.norm_outer.weight = nn.Parameter(norm_w, requires_grad=False)
        self.inter_projs = nn.ModuleList([
            nn.Linear(h_inner, h_outer, bias=False) for _ in range(4)
        ])

    def forward(self, tokens):
        x = self.embed(tokens).float()
        x = self.proj_in(x)
        fr = self.inner
        for scale in range(fr.n_scales):
            gamma = fr.scale_gamma[scale]
            beta = fr.scale_beta[scale]
            for it in range(fr.iters_per_scale):
                iter_s = fr.iter_scale[scale, it]
                x = x + (fr.block(x, gamma, beta, None, None, None) - x) * iter_s
        x = self.proj_out(x)
        x = self.norm_outer(x)
        return F.linear(x, self.lm_head_w)


def load_model(tag):
    """Auto-detect model type from checkpoint and build correct architecture."""
    path = f'checkpoints_1.7b_tinyfrr_{tag}/best.pt'
    if not os.path.exists(path):
        print(f"  SKIP {tag}: no checkpoint at {path}")
        return None, None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    h_inner = ck['h_inner']
    n_heads_inner = ck.get('n_heads_inner', next(
        (h for h in [16, 8, 12, 4] if h_inner % h == 0), 4))

    # Detect type from state_dict keys
    has_asvd = any(k.startswith('asvd_') for k in ck['state_dict'])
    has_inter = any(k.startswith('inter_projs') for k in ck['state_dict'])

    if has_asvd:
        rank = ck.get('rank', None)
        if rank is None:
            # infer from weights
            rank = ck['state_dict']['asvd_proj.weight'].shape[0]
        model = TinyFRR_ASVD(H_OUTER, h_inner, n_heads_inner, vocab_size, rank,
                              embed_w, norm_w).to(DEVICE)
        model_type = f'joint(h={h_inner},r={rank})'
    elif has_inter:
        model = TinyFRR_HQ3(H_OUTER, h_inner, n_heads_inner, vocab_size,
                             embed_w, lm_head_w, norm_w).to(DEVICE)
        model_type = f'hq3(h={h_inner})'
    else:
        model = TinyFRR(H_OUTER, h_inner, n_heads_inner, vocab_size,
                        embed_w, lm_head_w, norm_w).to(DEVICE)
        model_type = f'body(h={h_inner})'

    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Count frozen lm_head if it's there
    if has_asvd:
        total_compressed = ck.get('total_train', trainable)
    else:
        total_compressed = trainable  # body only (lm_head is teacher's, frozen)

    print(f"  Loaded {tag}: {model_type}  params={total_compressed/1e6:.2f}M  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return model, {'type': model_type, 'params': total_compressed, 'ck': ck}


@torch.no_grad()
def evaluate(model, tag, n):
    teacher_nll = student_nll = 0.0
    top1_agree = top10_overlap = kl_total = 0
    n_tokens = 0

    for i in range(n):
        s = int(starts[i].item())
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]; tgt = toks[0, 1:SEQ_LEN+1]

        tl = teacher.forward(inp, max_layers=N_TEACHER_LAYERS)
        sl = model(inp)

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
        'teacher_ppl': t_ppl,
        'student_ppl': s_ppl,
        'ppl_ratio': ppl_ratio,
        'quality_pct': 100.0 / ppl_ratio,
        'top1_pct': 100.0 * top1_agree / n_tokens,
        'top10_pct': 100.0 * top10_overlap / n_tokens,
        'kl_per_token': kl_total / n_tokens,
    }


# ==================== Run ====================
results = {}
teacher_full_params = N_TEACHER_LAYERS * 4 * H_OUTER * H_OUTER + lm_head_w.numel()

print(f"\nTeacher: {teacher_full_params/1e6:.1f}M core+head params")
print(f"{'='*100}")

for tag in args.tags:
    model, info = load_model(tag)
    if model is None:
        continue
    r = evaluate(model, tag, args.n)
    r.update({'type': info['type'], 'params': info['params']})
    r['compression'] = teacher_full_params / info['params']
    results[tag] = r
    del model
    torch.cuda.empty_cache()

# Print summary table
print(f"\n{'='*100}")
print(f"{'tag':25s} {'type':20s} {'params':>8s} {'comp':>8s} {'quality':>8s} {'top1':>8s} {'top10':>8s} {'ppl':>8s} {'kl':>8s}")
print(f"{'-'*100}")
for tag, r in sorted(results.items(), key=lambda x: -x[1]['quality_pct']):
    print(f"{tag:25s} {r['type']:20s} {r['params']/1e6:>7.2f}M {r['compression']:>7.1f}x "
          f"{r['quality_pct']:>7.1f}% {r['top1_pct']:>7.1f}% {r['top10_pct']:>7.1f}% "
          f"{r['ppl_ratio']:>7.3f} {r['kl_per_token']:>7.3f}")
print(f"{'='*100}")
