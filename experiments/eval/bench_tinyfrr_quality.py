"""
Proper quality benchmark for TinyFRR checkpoints.

Metrics that matter (beyond top-10 overlap):
  1. Perplexity ratio vs teacher — the STANDARD compression quality metric
  2. KL divergence per token (information-theoretic distance)
  3. Top-1 agreement rate (student top-1 == teacher top-1)
  4. Expected top-1 probability assigned to teacher's top-1 token
  5. Sequence-level log-likelihood correlation

These give a much richer picture than "top-10 overlap".
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
ap.add_argument('--tags', nargs='+',
                default=['h128_long', 'h512', 'h48_long', 'h128', 'h16'])
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--device', type=str, default='cuda:0')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = 64
N_TEACHER_LAYERS = 28

print(f"Quality benchmark: tags={args.tags}  n={args.n}  device={DEVICE}")

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
def bench(model, tag):
    """
    For each sequence of len SEQ_LEN, compute next-token prediction at each position.
    Position i predicts token i+1 (where i < SEQ_LEN-1). We use the true next token
    (from tokens[i+1]) for perplexity, and teacher logits for fidelity.
    """
    # Perplexity: -log p(true next token)
    teacher_nll = 0.0
    student_nll = 0.0
    n_tokens = 0

    # Fidelity
    top1_agree = 0
    p_teacher_top1_on_student = 0.0  # student's prob mass on teacher's top-1
    kl_total = 0.0
    pearson_num = 0.0; pearson_den_t = 0.0; pearson_den_s = 0.0

    for i, s in enumerate(starts):
        s = int(s.item())
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        input_toks = toks[:, :SEQ_LEN]
        target_toks = toks[0, 1:SEQ_LEN+1]  # shifted

        tl = teacher.forward(input_toks, max_layers=N_TEACHER_LAYERS)  # [1,S,V]
        sl = model(input_toks)

        # NLL on true next tokens
        t_logp = F.log_softmax(tl[0].float(), dim=-1)
        s_logp = F.log_softmax(sl[0].float(), dim=-1)
        teacher_nll += -t_logp[torch.arange(SEQ_LEN), target_toks].sum().item()
        student_nll += -s_logp[torch.arange(SEQ_LEN), target_toks].sum().item()
        n_tokens += SEQ_LEN

        # Top-1 agreement and confidence mass
        t_top1 = tl[0].argmax(dim=-1)  # [S]
        s_top1 = sl[0].argmax(dim=-1)
        top1_agree += (t_top1 == s_top1).sum().item()

        s_prob = F.softmax(sl[0].float(), dim=-1)
        p_teacher_top1_on_student += s_prob[torch.arange(SEQ_LEN), t_top1].sum().item()

        # KL(teacher || student) per-token average over vocab
        t_prob = F.softmax(tl[0].float(), dim=-1)
        kl = (t_prob * (t_prob.clamp_min(1e-12).log() - s_logp)).sum(-1)
        kl_total += kl.sum().item()

        # Pearson on LOGITS (centered)
        tc = (tl[0].float() - tl[0].float().mean(dim=-1, keepdim=True)).flatten()
        sc = (sl[0].float() - sl[0].float().mean(dim=-1, keepdim=True)).flatten()
        pearson_num += (tc * sc).sum().item()
        pearson_den_t += (tc * tc).sum().item()
        pearson_den_s += (sc * sc).sum().item()

        if (i+1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{args.n}", flush=True)

    teacher_ppl = math.exp(teacher_nll / n_tokens)
    student_ppl = math.exp(student_nll / n_tokens)
    ppl_ratio = student_ppl / teacher_ppl
    # "Quality %" defined as: how close student PPL is to teacher PPL
    # ppl_ratio=1.0 → 100% quality, 1.5 → 66.7%, 2.0 → 50%
    ppl_quality_pct = 100.0 / ppl_ratio

    return {
        'teacher_ppl': teacher_ppl,
        'student_ppl': student_ppl,
        'ppl_ratio': ppl_ratio,
        'ppl_quality_pct': ppl_quality_pct,
        'top1_agree_pct': 100.0 * top1_agree / n_tokens,
        'p_teacher_top1_pct': 100.0 * p_teacher_top1_on_student / n_tokens,
        'kl_per_token': kl_total / n_tokens,
        'logit_pearson': pearson_num / math.sqrt(pearson_den_t * pearson_den_s),
    }


# Teacher vs itself reference (theoretical limit)
print("\n===================== TEACHER SELF-REFERENCE =====================")
print("(teacher_ppl is the theoretical lower bound; student_ppl approaches it)")

results = {}
for tag in args.tags:
    loaded = load_ckpt(tag)
    if loaded is None:
        print(f"\n!!! {tag}: checkpoint missing, skipping", flush=True)
        continue
    model, ck = loaded
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n>>> {tag}  h={ck['h_inner']}  trainable={trainable/1e6:.2f}M", flush=True)
    r = bench(model, tag)
    results[tag] = r
    print(f"  teacher PPL: {r['teacher_ppl']:.3f}  student PPL: {r['student_ppl']:.3f}  ratio: {r['ppl_ratio']:.3f}")
    print(f"  QUALITY (100/ratio):   {r['ppl_quality_pct']:.2f}%")
    print(f"  Top-1 agreement:       {r['top1_agree_pct']:.2f}%")
    print(f"  Student-prob on T-top1:{r['p_teacher_top1_pct']:.2f}%")
    print(f"  KL per token:          {r['kl_per_token']:.4f} nats")
    print(f"  Logit Pearson r:       {r['logit_pearson']:.4f}")
    del model
    torch.cuda.empty_cache()

print("\n" + "="*78)
print(f"{'tag':<14}{'PPL-ratio':>10}{'Quality%':>10}{'Top1-agr%':>11}{'P(T-top1)%':>12}{'KL/tok':>10}{'Pearson':>10}")
print("="*78)
for tag, r in results.items():
    print(f"{tag:<14}{r['ppl_ratio']:>10.3f}{r['ppl_quality_pct']:>10.2f}{r['top1_agree_pct']:>11.2f}{r['p_teacher_top1_pct']:>12.2f}{r['kl_per_token']:>10.4f}{r['logit_pearson']:>10.4f}")

torch.save(results, 'tinyfrr_quality_bench.pt')
print("\nSaved: tinyfrr_quality_bench.pt")
