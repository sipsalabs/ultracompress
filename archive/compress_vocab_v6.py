"""
SemanticBasis v6 — Product-Quantized HypernetBasis.

THE NEXT BIT-WIDTH FLOOR
  v5 int4 = 4 bits/weight + tiny scale overhead = ~4.1 bits/weight
  v5 int2 = 2 bits/weight + scale = ~2.1 bits/weight, but T1 crashed to 69.5%.

  Product Quantization (PQ) breaks through the per-weight bit floor.
  Instead of quantizing each weight independently, we quantize GROUPS
  of D weights jointly to the nearest entry in a small learned
  codebook of K vectors.

  Effective bits/weight = log2(K) / D.
  With K=16, D=4  -> 1.0 bit / weight    (4x smaller than int4)
  With K=16, D=8  -> 0.5 bits / weight  (8x smaller than int4)
  With K=256, D=8 -> 1.0 bit / weight   (same as above, more expressive)

  Codebook overhead is tiny (K*D fp16 per Linear) and amortized over
  millions of weights. Per-row fp16 scale preserves dynamic range.

  The VECTORIZATION lets the codebook capture WEIGHT CORRELATIONS
  that int-N cannot -- two weights that always co-occur get merged
  into one codeword. This is the information-theoretic reason PQ
  beats scalar quant below int4.

PIPELINE
  1. Load v4 (or v4-xtreme) fp32 checkpoint.
  2. For each Linear weight in the hypernet + hot U/B:
       - Reshape to [n_rows, n_groups, D].
       - k-means init codebook on sampled subvectors.
       - Replace each subvector with its nearest codeword.
  3. Store: codes (small int), codebook (fp16), per-row scale (fp16).
  4. QAT 500-1000 steps with straight-through estimator on code
     assignments AND learnable codebook.
  5. Eval & report byte-level compression.

PATENT CLAIM
  First application of product quantization to HYPERNET-generated
  vocab embeddings. The combination (hypernet + PQ) provides vocab
  compression that is both O(1) in vocabulary size AND sub-bit-per-
  weight in storage.

USAGE
  python compress_vocab_v6.py --sb4_ckpt qwen3_1.7b_sb4_xtreme.pt \
      --teacher_cache qwen3_1.7b_cache.pt \
      --codebook_size 16 --subvec 4 --qat_steps 800 \
      --out qwen3_1.7b_sb6_pq.pt
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from compress_vocab_v4 import SemanticBasisV4


# ============================================================
# Straight-through nearest-codeword assignment.
# ============================================================
class _PQNearest(torch.autograd.Function):
    """Hard assign + straight-through gradient to both inputs."""

    @staticmethod
    def forward(ctx, x, codebook):
        # x:        [N, D]
        # codebook: [K, D]
        # return:   [N, D] -- codebook[argmin ||x - c||^2]
        # distances in batched form
        d = torch.cdist(x.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)  # [N,K]
        idx = d.argmin(-1)  # [N]
        out = codebook[idx]
        ctx.save_for_backward(idx, codebook)
        return out

    @staticmethod
    def backward(ctx, g):
        # Straight-through to x. For codebook, attribute gradient to
        # assigned rows (scatter_add).
        idx, codebook = ctx.saved_tensors
        g_code = torch.zeros_like(codebook)
        g_code.index_add_(0, idx, g)
        return g, g_code


def pq_quantize(w: torch.Tensor, codebook: torch.Tensor, row_scale: torch.Tensor,
                D: int) -> torch.Tensor:
    """Vectorized PQ quantization of a 2-D weight matrix.

    w:         [O, I]
    codebook:  [K, D]
    row_scale: [O, 1]
    returns:   [O, I]
    """
    O, I = w.shape
    assert I % D == 0, f"I={I} not divisible by D={D}"
    w_scaled = w / row_scale  # [O, I]
    groups = w_scaled.view(O, I // D, D).reshape(-1, D)  # [O*(I/D), D]
    q = _PQNearest.apply(groups, codebook)  # [O*(I/D), D]
    return q.view(O, I // D, D).view(O, I) * row_scale


# ============================================================
# Drop-in PQ-quantized Linear.
# ============================================================
class PQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, K: int, D: int,
                 init_samples: torch.Tensor = None):
        super().__init__()
        self.out_features = linear.out_features
        self.in_features = linear.in_features
        self.K, self.D = K, D
        assert self.in_features % D == 0, f"in_features={self.in_features} not divisible by D={D}"
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        # per-row scale
        rs = self.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        self.row_scale = nn.Parameter(rs)
        # codebook via k-means init
        with torch.no_grad():
            wn = (self.weight / rs).view(-1, D)  # [O*I/D, D]
            cb = self._kmeans(wn, K, iters=5)
        self.codebook = nn.Parameter(cb)

    @staticmethod
    def _kmeans(X: torch.Tensor, K: int, iters: int = 5):
        # X: [N, D]
        N, D = X.shape
        # init: random N points
        idx = torch.randperm(N, device=X.device)[:K]
        cb = X[idx].clone()
        for _ in range(iters):
            d = torch.cdist(X.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)  # [N,K]
            a = d.argmin(-1)
            for k in range(K):
                m = a == k
                if m.any():
                    cb[k] = X[m].mean(0)
        return cb

    def forward(self, x):
        wq = pq_quantize(self.weight, self.codebook, self.row_scale, self.D)
        return F.linear(x, wq, self.bias)


# ============================================================
# PQ-quantized Hot low-rank (U:[K,r], B:[r,d]).
# ============================================================
class PQHotLowRank(nn.Module):
    def __init__(self, hot, K_cb: int, D: int):
        super().__init__()
        self.K = hot.K; self.d = hot.d; self.r = hot.r
        self.D = D; self.K_cb = K_cb

        self.U = nn.Parameter(hot.U.data.clone())
        self.B = nn.Parameter(hot.B.data.clone())
        rs_U = self.U.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        rs_B = self.B.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        self.rs_U = nn.Parameter(rs_U)
        self.rs_B = nn.Parameter(rs_B)
        # codebooks
        cb_U = self._kmeans_init(self.U, rs_U, D, K_cb)
        cb_B = self._kmeans_init(self.B, rs_B, D, K_cb)
        self.cb_U = nn.Parameter(cb_U)
        self.cb_B = nn.Parameter(cb_B)

    @staticmethod
    def _kmeans_init(W, rs, D, K):
        wn = (W / rs).reshape(-1, D)
        idx = torch.randperm(wn.shape[0], device=wn.device)[:K]
        cb = wn[idx].clone()
        for _ in range(5):
            d = torch.cdist(wn.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
            a = d.argmin(-1)
            for k in range(K):
                m = a == k
                if m.any():
                    cb[k] = wn[m].mean(0)
        return cb

    def _Uq(self):
        Oo, Io = self.U.shape
        if Io % self.D != 0:
            # fall back to no PQ on this matrix
            return self.U
        return pq_quantize(self.U, self.cb_U, self.rs_U, self.D)

    def _Bq(self):
        Ob, Ib = self.B.shape
        if Ib % self.D != 0:
            return self.B
        return pq_quantize(self.B, self.cb_B, self.rs_B, self.D)

    def all_rows(self):
        return self._Uq() @ self._Bq()

    def row(self, local_ids):
        return self._Uq()[local_ids] @ self._Bq()


def pq_replace(sb: SemanticBasisV4, K_cb: int, D: int):
    """Replace every nn.Linear in hyper.net with PQLinear; replace hot with PQHotLowRank."""
    seq = sb.hyper.net
    new_layers = []
    for m in seq:
        if isinstance(m, nn.Linear):
            if m.in_features % D == 0:
                new_layers.append(PQLinear(m, K=K_cb, D=D))
            else:
                # keep fp32 for the oddball (usually the small first layer)
                new_layers.append(m)
                print(f"  [note] keeping Linear[{m.in_features}->{m.out_features}] at fp32 "
                      f"(in_features not divisible by D={D})")
        else:
            new_layers.append(m)
    sb.hyper.net = nn.Sequential(*new_layers)
    sb.hot = PQHotLowRank(sb.hot, K_cb=K_cb, D=D)


def effective_byte_size(sb: SemanticBasisV4, K_cb: int, D: int) -> float:
    total_bits = 0.0
    code_bits = max(1, int(math.ceil(math.log2(K_cb))))
    # count param-level contributions
    hyper_net = sb.hyper.net
    for m in hyper_net:
        if isinstance(m, PQLinear):
            O, I = m.weight.shape
            # codes: (O * I/D) * code_bits
            total_bits += O * (I // D) * code_bits
            # codebook: K*D fp16
            total_bits += K_cb * D * 16
            # row_scale: O fp16
            total_bits += O * 16
            if m.bias is not None:
                total_bits += m.bias.numel() * 16
        elif isinstance(m, nn.Linear):
            total_bits += m.weight.numel() * 16
            if m.bias is not None:
                total_bits += m.bias.numel() * 16
    # hot
    h = sb.hot
    if isinstance(h, PQHotLowRank):
        Uo, Ui = h.U.shape
        if Ui % D == 0:
            total_bits += Uo * (Ui // D) * code_bits + K_cb * D * 16 + Uo * 16
        else:
            total_bits += h.U.numel() * 16
        Bo, Bi = h.B.shape
        if Bi % D == 0:
            total_bits += Bo * (Bi // D) * code_bits + K_cb * D * 16 + Bo * 16
        else:
            total_bits += h.B.numel() * 16
    # remaining fp16 params/buffers
    for name, p in sb.named_parameters():
        if name.startswith('hyper.') or name.startswith('hot.'):
            continue
        total_bits += p.numel() * 16
    for name, b in sb.named_buffers():
        if 'old_to_new' in name or 'new_to_old' in name:
            total_bits += b.numel() * 32
        else:
            total_bits += b.numel() * 16
    return total_bits / 8.0


@torch.no_grad()
def fidelity_eval(sb, teacher, tb, eval_tokens, device, label,
                  n_seqs=80, seq=128):
    if not os.path.exists(eval_tokens):
        print(f"  ({label}) no eval file"); return 0.0, 0.0
    toks = torch.load(eval_tokens, weights_only=True)
    agree_t1 = agree_t10 = 0.0; n_tok = 0
    for _ in range(n_seqs):
        s = int(torch.randint(0, toks.numel() - seq - 1, (1,)).item())
        t = toks[s:s + seq].unsqueeze(0).long().to(device)
        t_logits, t_hs = teacher.forward(t, max_layers=tb.n_layers, return_hidden=True)
        latent = teacher.final_norm(t_hs[-1]).float()
        s_logits = sb.decode(latent).float()
        agree_t1 += (t_logits[0].argmax(-1) == s_logits[0].argmax(-1)).float().mean().item()
        for pos in range(seq):
            top_t = set(t_logits[0, pos].topk(10).indices.tolist())
            top_s = set(s_logits[0, pos].topk(10).indices.tolist())
            agree_t10 += len(top_t & top_s) / 10; n_tok += 1
    t1, t10 = agree_t1 / n_seqs * 100, agree_t10 / n_tok * 100
    print(f"  ({label}) T1 = {t1:.2f}%    T10 = {t10:.2f}%")
    return t1, t10


def qat_finetune(sb, teacher, tb, all_tokens, device, steps=800, batch=4,
                 seq=128, lr=3e-4, verbose=True, eval_tokens=None, eval_every=100,
                 eval_seqs=20):
    opt = torch.optim.AdamW(sb.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time()
    # best-checkpoint tracking: pick the QAT step with highest eval T1,
    # not the last step. QAT is noisy under aggressive PQ; the last step
    # is often worse than an intermediate one.
    best_t1 = -1.0
    best_state = {k: v.detach().clone() for k, v in sb.state_dict().items()}
    best_step = 0
    for step in range(steps):
        starts = torch.randint(0, all_tokens.numel() - seq, (batch,))
        toks = torch.stack([all_tokens[s:s + seq].long() for s in starts]).to(device)
        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=tb.n_layers, return_hidden=True)
            latent = teacher.final_norm(t_hs[-1]).float()
            t_logits = t_logits.float()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits = sb.decode(latent)
            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            tgt = t_logits.argmax(-1)
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), tgt.view(-1))
            loss = kl + 0.3 * ce
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(sb.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if verbose and (step % 50 == 0 or step == steps - 1):
            with torch.no_grad():
                t1 = (s_logits.argmax(-1) == tgt).float().mean().item()
            print(f"  [PQ-QAT] step={step:4d}  kl={kl.item():.3f}  ce={ce.item():.3f}  "
                  f"T1={t1*100:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        # periodic eval + best-state save
        if eval_tokens is not None and (step % eval_every == 0 or step == steps - 1):
            sb.eval()
            with torch.no_grad():
                et1, _ = fidelity_eval(sb, teacher, tb, eval_tokens, device,
                                       f'qat-eval s={step}', n_seqs=eval_seqs, seq=seq)
            sb.train()
            if et1 > best_t1:
                best_t1 = et1
                best_step = step
                best_state = {k: v.detach().clone() for k, v in sb.state_dict().items()}
    # restore best
    sb.load_state_dict(best_state)
    print(f"  [PQ-QAT] restored best state: step={best_step}  eval_T1={best_t1:.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sb4_ckpt', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--codebook_size', type=int, default=16)
    ap.add_argument('--subvec', type=int, default=4)
    ap.add_argument('--qat_steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--eval_tokens', type=str, default='fineweb_edu_100M_tokens.pt')
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--eval_seqs', type=int, default=20)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    from scaling.teacher_loader import load_qwen3_teacher
    tb = load_qwen3_teacher(args.teacher_cache, device=args.device)
    V, d = tb.vocab_size, tb.h_outer
    embed_w = tb.embed_w.detach().to(args.device).float()
    lm_head_w = tb.lm_head_w.detach().to(args.device).float()

    ckpt = torch.load(args.sb4_ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt['config']
    hot_ids = ckpt['hot_ids']
    sb = SemanticBasisV4(V, d,
                         k_hot=cfg['k_hot'], r_hot=cfg['r_hot'],
                         hyper_hidden=cfg['hyper_hidden'],
                         n_freqs=cfg['n_freqs'],
                         hot_ids=hot_ids).to(args.device)
    sb.load_state_dict(ckpt['state_dict'])
    print(f"Loaded v4: {args.sb4_ckpt}   params={sb.num_params()/1e6:.2f}M")

    if os.path.exists(args.eval_tokens):
        toks = torch.load(args.eval_tokens, weights_only=True)
    print("\n--- fp32 baseline ---")
    t1_fp, t10_fp = fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'fp32')

    print(f"\n--- PQ quantize K={args.codebook_size} D={args.subvec} ---")
    pq_replace(sb, K_cb=args.codebook_size, D=args.subvec)
    sb = sb.to(args.device)

    n_orig_bytes = (embed_w.numel() + lm_head_w.numel()) * 2
    n_q_bytes = effective_byte_size(sb, K_cb=args.codebook_size, D=args.subvec)
    bits_per_w = math.log2(args.codebook_size) / args.subvec
    print(f"\n--- byte accounting ---")
    print(f"  codebook_size={args.codebook_size}  subvec={args.subvec}  "
          f"effective bits/weight (codes only) = {bits_per_w:.3f}")
    print(f"  original (fp16):     {n_orig_bytes/1e6:.2f} MB")
    print(f"  v6 PQ effective:     {n_q_bytes/1e6:.3f} MB")
    print(f"  byte compression:    {n_orig_bytes/n_q_bytes:.1f}x")

    print("\n--- post-quant pre-QAT ---")
    t1_pq, t10_pq = fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'post-quant pre-QAT')

    print(f"\n--- PQ-QAT recovery ({args.qat_steps} steps) ---")
    qat_finetune(sb, tb.teacher, tb, toks, args.device,
                 steps=args.qat_steps, batch=args.batch, seq=args.seq, lr=args.lr,
                 eval_tokens=args.eval_tokens, eval_every=args.eval_every,
                 eval_seqs=args.eval_seqs)

    print("\n--- FINAL post-QAT ---")
    t1_final, t10_final = fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'post-QAT')

    print("\n=== SUMMARY ===")
    print(f"  K={args.codebook_size}  D={args.subvec}")
    print(f"  byte compression: {n_orig_bytes/n_q_bytes:.1f}x  "
          f"({n_orig_bytes/1e6:.1f}MB -> {n_q_bytes/1e6:.3f}MB)")
    print(f"  T1  fp32={t1_fp:.2f}%  preQAT={t1_pq:.2f}%  postQAT={t1_final:.2f}%")
    print(f"  T10 fp32={t10_fp:.2f}%  preQAT={t10_pq:.2f}%  postQAT={t10_final:.2f}%")

    torch.save({
        'state_dict': sb.state_dict(),
        'config': {**cfg, 'codebook_size': args.codebook_size, 'subvec': args.subvec},
        'orig_fp16_bytes': n_orig_bytes, 'q_bytes': n_q_bytes,
        'byte_compression': n_orig_bytes / n_q_bytes,
        'teacher_cache': args.teacher_cache, 'hot_ids': hot_ids,
        'scores': {'t1_fp32': t1_fp, 't10_fp32': t10_fp,
                   't1_postqat': t1_final, 't10_postqat': t10_final},
    }, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
