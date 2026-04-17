"""
Activation-aware factorization of lm_head.

Naive SVD(W) destroys quality because the lm_head spectrum has a single
dominant singular value then a flat tail — truncating throws away the
vocabulary-discriminating directions that the teacher's actual latent
distribution uses heavily.

Fix: factor W with a sample-weighting matrix C = E[latent @ latent.T]^(1/2).
Let M = chol(C). Then W @ L is the "whitened" lm_head; SVD of that
preserves the teacher's *effective* behavior, not its worst-case behavior.
Equivalent: logits ≈ U_r @ diag(S_r) @ V_r^T @ L^{-1} @ latent.

This is ASVD / OWQ-style weighted low-rank.
"""
import lib.unbuffered
import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultracompress.inference import ModelConfig, MiniTransformer

ap = argparse.ArgumentParser()
ap.add_argument('--ranks', nargs='+', type=int, default=[64, 128, 256, 384, 512, 768, 1024])
ap.add_argument('--n_calib', type=int, default=64)
ap.add_argument('--n_eval', type=int, default=100)
ap.add_argument('--device', type=str, default='cuda:0')
args = ap.parse_args()

DEVICE = args.device
SEQ_LEN = 64
N_TEACHER_LAYERS = 28

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

H = gd['token_embd.weight'].shape[1]
V = gd['token_embd.weight'].shape[0]
cfg = ModelConfig(n_layers=N_TEACHER_LAYERS, n_heads=16, n_kv_heads=8,
                  hidden_size=H, intermediate_size=H*3, vocab_size=V, head_dim=H//16)
teacher = MiniTransformer(cfg, DEVICE)
teacher.load_weights(gd)
teacher.embed_weight = teacher.embed_weight.to(DEVICE)
if teacher.lm_head is not None:
    teacher.lm_head = teacher.lm_head.to(DEVICE)

W = gd['output.weight'].to(DEVICE).float()
print(f"lm_head: {W.shape}  params = {W.numel()/1e6:.1f}M")

all_tokens = torch.load('fineweb_edu_100M_tokens.pt', weights_only=True)

print(f"calibrating activation covariance on {args.n_calib} sequences...")
Mcov = torch.zeros(H, H, device=DEVICE, dtype=torch.float32)
n_obs = 0
starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (args.n_calib,))
with torch.no_grad():
    for s in starts:
        s = int(s.item())
        toks = all_tokens[s:s+SEQ_LEN].unsqueeze(0).long().to(DEVICE)
        _, hs = teacher.forward(toks, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        latent = teacher.final_norm(hs[-1]).float()
        flat = latent.reshape(-1, H)
        Mcov += flat.T @ flat
        n_obs += flat.shape[0]
Mcov /= n_obs
eps = 1e-3 * Mcov.diagonal().mean().item()
Mcov = Mcov + eps * torch.eye(H, device=DEVICE)
print(f"  calibration tokens={n_obs}  diag ratio={Mcov.diagonal().max()/Mcov.diagonal().min():.1f}")

Lchol = torch.linalg.cholesky(Mcov)
Linv = torch.linalg.solve_triangular(Lchol, torch.eye(H, device=DEVICE), upper=False)

print("computing activation-weighted SVD...")
WL = W @ Lchol
U, S, Vt = torch.linalg.svd(WL, full_matrices=False)
print(f"  top-10 weighted sv: {[f'{x:.2f}' for x in S[:10].tolist()]}")

@torch.no_grad()
def bench(head_fn, n=args.n_eval):
    top1 = 0.0; top10 = 0.0; kl = 0.0; s_nll = 0.0; t_nll = 0.0
    n_tok = 0
    starts = torch.randint(0, all_tokens.numel() - SEQ_LEN - 1, (n,))
    for s in starts:
        s = int(s.item())
        toks = all_tokens[s:s+SEQ_LEN+1].unsqueeze(0).long().to(DEVICE)
        inp = toks[:, :SEQ_LEN]; tgt = toks[0, 1:SEQ_LEN+1]
        _, hs = teacher.forward(inp, max_layers=N_TEACHER_LAYERS, return_hidden=True)
        latent = teacher.final_norm(hs[-1])
        t_logits = F.linear(latent, W)
        s_logits = head_fn(latent)
        t_lp = F.log_softmax(t_logits[0].float(), -1)
        s_lp = F.log_softmax(s_logits[0].float(), -1)
        t_nll += -t_lp[torch.arange(SEQ_LEN), tgt].sum().item()
        s_nll += -s_lp[torch.arange(SEQ_LEN), tgt].sum().item()
        kl += (t_lp.exp() * (t_lp - s_lp)).sum(-1).sum().item()
        n_tok += SEQ_LEN
        t_top1 = t_logits[0].argmax(-1); s_top1 = s_logits[0].argmax(-1)
        top1 += (t_top1 == s_top1).float().sum().item()
        for pos in range(SEQ_LEN):
            tt = set(t_logits[0, pos].topk(10).indices.tolist())
            st = set(s_logits[0, pos].topk(10).indices.tolist())
            top10 += len(tt & st) / 10
    t_ppl = math.exp(t_nll / n_tok); s_ppl = math.exp(s_nll / n_tok)
    return {
        'ppl_ratio': s_ppl / t_ppl,
        'quality_pct': 100.0 * t_ppl / s_ppl,
        'top1_agree': top1 / n_tok,
        'top10_agree': top10 / n_tok,
        'kl_per_tok': kl / n_tok,
    }


print("\n=== Naive SVD vs Activation-Aware SVD (ASVD) ===")
results = {'naive': {}, 'asvd': {}}
Un, Sn, Vtn = torch.linalg.svd(W, full_matrices=False)

for r in args.ranks:
    r = min(r, H)
    Un_r = Un[:, :r].contiguous(); Sn_r = Sn[:r]; Vtn_r = Vtn[:r, :].contiguous()
    def naive_head(z, V=Vtn_r, U=Un_r, S=Sn_r):
        return F.linear(F.linear(z, V), U * S.unsqueeze(0))
    mn = bench(naive_head)

    U_r = U[:, :r].contiguous(); S_r = S[:r]; Vt_r = Vt[:r, :].contiguous()
    Proj = Vt_r @ Linv
    Out = U_r * S_r.unsqueeze(0)
    def asvd_head(z, P=Proj, O=Out):
        return F.linear(F.linear(z, P), O)
    ma = bench(asvd_head)

    params = r * (V + H)
    comp = W.numel() / params
    print(f"  r={r:4d}  params={params/1e6:6.2f}M  head_compress={comp:5.1f}x")
    print(f"    naive:  ppl={mn['ppl_ratio']:10.3f}  quality={mn['quality_pct']:6.2f}%  top1={mn['top1_agree']*100:5.1f}%  top10={mn['top10_agree']*100:5.1f}%")
    print(f"    asvd:   ppl={ma['ppl_ratio']:10.3f}  quality={ma['quality_pct']:6.2f}%  top1={ma['top1_agree']*100:5.1f}%  top10={ma['top10_agree']*100:5.1f}%")
    results['naive'][r] = {**mn, 'params': params, 'compression': comp}
    results['asvd'][r] = {**ma, 'params': params, 'compression': comp}

m0 = bench(lambda z: F.linear(z, W))
print(f"  r=FULL  params={W.numel()/1e6:.1f}M  1.0x")
print(f"    exact:  ppl={m0['ppl_ratio']:.3f}  quality={m0['quality_pct']:.2f}%  top1={m0['top1_agree']*100:.1f}%  top10={m0['top10_agree']*100:.1f}%")
results['full'] = m0

torch.save(results, 'lmhead_factor_results.pt')
print("\nsaved lmhead_factor_results.pt")

print("\n=== lm_head memory at large scale (fp16) ===")
for (H_ex, V_ex, label) in [(2048, 152000, 'Qwen3-1.7B'),
                              (4096, 128000, 'Llama-3-8B'),
                              (8192, 128000, 'Llama-3-70B'),
                              (16384, 200000, '1T-class H=16k')]:
    full_gb = H_ex * V_ex * 2 / 1e9
    print(f"  {label}: H={H_ex} V={V_ex}  full fp16 = {full_gb:.2f} GB")
    for r in [128, 256, 512]:
        f_gb = r * (V_ex + H_ex) * 2 / 1e9
        print(f"     r={r:4d}: {f_gb:.3f} GB  ({full_gb/f_gb:4.0f}x shrink)")
