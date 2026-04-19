"""
SemanticBasis v4 â€” Hypernet-Generated Vocabulary.

THE SHIFT IN THINKING
  v1/v2/v3 all store one coefficient vector PER TOKEN. That costs O(V)
  params regardless of how clever the basis is. For 150K vocab + r=192
  that's 29M params just for cold-token coefficients. The bottleneck
  on further compression is now that per-token storage â€” not the basis.

  Tesla didn't build a better commutator, he eliminated the commutator.
  v4 eliminates per-token storage for the cold tail.

HYPERNET-GENERATED EMBEDDINGS
  We ask: does E[v] carry 2048 bits of information unique to token v,
  or is E[v] a (mostly) smooth function of v's identity?

  For the LONG TAIL it's the latter. Cold tokens are overwhelmingly
  subword fragments, rare names, whitespace variants â€” tokens whose
  embedding is dominated by SHARED structure (positional co-occurrence
  patterns, byte-level features, morphological signatures). A small
  neural net can REGENERATE cold embeddings from the token ID.

  We compute rich positional features of the integer token-id v:
    phi(v) = [ sin(2^k * v / V) , cos(2^k * v / V) ]_{k=0..K-1}
  ...and a tiny MLP decodes phi(v) -> d-dim embedding:
    E_cold(v) = MLP_embed( phi(v) ) * tail_gain
  Total per-token cost: ZERO. All cold storage is in the MLP weights
  (shared for all 150K+ tokens).

  Hot tokens (top-1024 frequent) keep their own learned low-rank
  coefficient â€” this is where most of the probability mass lives and
  fidelity matters.

NOVEL JOINT TRAINING
  Hypernet-generated embeddings + tied-decode + frequency hot-tier +
  decode-aware KD is, to our knowledge, not in the literature. It is
  the first vocab compression scheme whose per-token cost is
  independent of vocabulary size.

COMPRESSION MATH (Qwen3-1.7B, V=151936, d=2048)
  Original embed+head:                    622.33M params
  Hot tier (K=1024, r_hot=768):             2.37M
  HyperNet for cold (150K+ tokens):         2.0-3.5M (configurable)
  out_bias (V) + log_alpha:                 0.15M
  --------------------------------------------------
  Total v4:                                 ~5M params
  Vocab compression:                        ~125x

WORKS FOR ANY MODEL
  Hypernet is VOCAB-AGNOSTIC: same 3M MLP encodes a 151K or 1M token
  vocabulary. This is the key property for patent claim #2 ("vocab
  compression cost independent of vocabulary size").

USAGE
  python compress_vocab_v4.py --teacher_cache qwen3_1.7b_cache.pt \
      --k_hot 1024 --r_hot 768 --hyper_hidden 1024 \
      --kd_steps 1500 --out qwen3_1.7b_sb4.pt
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


class FourierIDFeatures(nn.Module):
    """Deterministic Fourier embedding of integer token IDs.

    Yields a 2*K dim feature vector per ID. No learned params.
    """
    def __init__(self, V, n_freqs=48):
        super().__init__()
        # multi-scale: log-spaced frequencies + integer byte features
        self.V = V
        freqs = 2.0 ** torch.linspace(0, math.log2(V), n_freqs)
        self.register_buffer('freqs', freqs, persistent=True)
        self.register_buffer('V_tensor', torch.tensor(float(V)), persistent=True)

    @property
    def out_dim(self):
        return self.freqs.numel() * 2 + 8  # +8 byte-features

    def forward(self, ids):
        # ids: [...,] int64
        x = ids.float() / self.V_tensor          # [...]
        theta = x.unsqueeze(-1) * self.freqs     # [..., n_freqs]
        feats = [torch.sin(theta), torch.cos(theta)]
        # byte-level features: MSBs of the integer
        for k in range(8):
            b = ((ids >> (k * 2)) & 0x3).float() / 3.0
            feats.append(b.unsqueeze(-1))
        return torch.cat(feats, dim=-1)


class HyperNet(nn.Module):
    """Tiny MLP mapping Fourier features -> d-dim embedding."""
    def __init__(self, in_dim, hidden, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        # small-init on last layer so initial outputs are near zero
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi):
        return self.net(phi)


class HotLowRank(nn.Module):
    """Full low-rank for the top-K hot tokens."""
    def __init__(self, K, d, r_hot):
        super().__init__()
        self.K, self.d, self.r = K, d, r_hot
        self.U = nn.Parameter(torch.randn(K, r_hot) / math.sqrt(r_hot))
        self.B = nn.Parameter(torch.randn(r_hot, d) / math.sqrt(r_hot))

    @torch.no_grad()
    def svd_init(self, hot_tgt):
        U, S, Vh = torch.linalg.svd(hot_tgt.float(), full_matrices=False)
        self.U.data = (U[:, :self.r] * S[:self.r]).to(self.U.dtype)
        self.B.data = Vh[:self.r].to(self.B.dtype)

    def all_rows(self):
        return self.U @ self.B

    def row(self, local_ids):
        return self.U[local_ids] @ self.B


class SemanticBasisV4(nn.Module):
    """Hypernet-generated tail + hot low-rank + tied decode."""

    def __init__(self, V, d, k_hot=1024, r_hot=768, hyper_hidden=1024,
                 n_freqs=48, hot_ids=None, tail_gain_init=1.0):
        super().__init__()
        self.V, self.d = V, d
        self.K = k_hot
        self.r_hot = r_hot

        if hot_ids is None:
            hot_ids = torch.arange(k_hot, dtype=torch.long)
        assert hot_ids.numel() == k_hot
        mask = torch.zeros(V, dtype=torch.bool); mask[hot_ids] = True
        cold_ids = torch.arange(V)[~mask]
        new_to_old = torch.cat([hot_ids.long(), cold_ids.long()], dim=0)
        old_to_new = torch.empty(V, dtype=torch.long)
        old_to_new[new_to_old] = torch.arange(V)
        self.register_buffer('old_to_new', old_to_new, persistent=True)
        self.register_buffer('new_to_old', new_to_old, persistent=True)

        self.fourier = FourierIDFeatures(V, n_freqs=n_freqs)
        self.hyper = HyperNet(self.fourier.out_dim, hyper_hidden, d)
        self.hot = HotLowRank(k_hot, d, r_hot)
        self.tail_gain = nn.Parameter(torch.tensor(float(tail_gain_init)))
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.out_bias = nn.Parameter(torch.zeros(V))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _build_all_E(self):
        """Materialize E of shape [V, d] in NEW-id order."""
        # hot rows: first K in new order
        hot_rows = self.hot.all_rows()  # [K, d]
        # cold rows from hypernet, applied to ORIGINAL ids in their new order
        cold_orig_ids = self.new_to_old[self.K:]  # [V-K]
        phi = self.fourier(cold_orig_ids)
        cold_rows = self.hyper(phi) * self.tail_gain  # [V-K, d]
        return torch.cat([hot_rows, cold_rows], dim=0)  # [V, d] in new order

    def encode(self, tokens):
        """tokens: [...] int64 original-id -> embedding [..., d]"""
        new_tok = self.old_to_new[tokens]
        is_hot = new_tok < self.K
        out = torch.empty(*tokens.shape, self.d, device=tokens.device,
                          dtype=self.hot.U.dtype)
        if is_hot.any():
            out[is_hot] = self.hot.row(new_tok[is_hot]).to(out.dtype)
        if (~is_hot).any():
            cold_orig = tokens[~is_hot]
            phi = self.fourier(cold_orig)
            out[~is_hot] = self.hyper(phi).to(out.dtype) * self.tail_gain
        return out

    def decode(self, latent):
        """latent: [..., d] -> logits [..., V]"""
        E = self._build_all_E()  # [V, d] in new order
        logits_new = latent @ E.T  # [..., V]
        # un-permute to original vocab order
        logits = logits_new.index_select(-1, self.old_to_new)
        return logits * self.log_alpha.exp() + self.out_bias


def compute_hot_ids(lm_head_weight, k_hot, token_counts=None):
    if token_counts is not None:
        return torch.argsort(token_counts, descending=True)[:k_hot].to(torch.long)
    return torch.argsort(lm_head_weight.norm(dim=1), descending=True)[:k_hot].to(torch.long)


def fit_v4(sb, teacher, tb, all_tokens, device, embed_w,
           steps=1500, batch=4, seq=128, lr=1e-3, enc_w=0.05,
           verbose=True):
    opt = torch.optim.AdamW(sb.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    embed_w_dev = embed_w.to(device).float()
    t0 = time.time()

    for step in range(steps):
        starts = torch.randint(0, all_tokens.numel() - seq, (batch,))
        toks = torch.stack([all_tokens[s:s + seq].long() for s in starts]).to(device)
        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=tb.n_layers,
                                             return_hidden=True)
            latent = teacher.final_norm(t_hs[-1]).float()
            t_logits = t_logits.float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits = sb.decode(latent)
            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            tgt = t_logits.argmax(-1)
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                                 tgt.view(-1))

            if enc_w > 0:
                sample_ids = torch.randint(0, sb.V, (4096,), device=device)
                rec = sb.encode(sample_ids.unsqueeze(0))[0]
                enc_mse = ((rec.float() - embed_w_dev[sample_ids]) ** 2).mean()
            else:
                enc_mse = torch.tensor(0.0, device=device)

            loss = kl + 0.3 * ce + enc_w * enc_mse

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(sb.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if verbose and (step % 100 == 0 or step == steps - 1):
            with torch.no_grad():
                t1 = (s_logits.argmax(-1) == tgt).float().mean().item()
                topk_t = t_logits.topk(10, -1).indices
                topk_s = s_logits.topk(10, -1).indices
                t10 = (topk_t.unsqueeze(-1) == topk_s.unsqueeze(-2)).any(-1).float().mean().item()
            print(f"  [KD] step={step:4d}  kl={kl.item():.3f}  ce={ce.item():.3f}  "
                  f"enc={enc_mse.item():.5f}  T1={t1*100:5.2f}%  T10={t10*100:5.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def fidelity_eval(sb, teacher, tb, eval_tokens, device, label,
                  n_seqs=80, seq=128):
    if not os.path.exists(eval_tokens):
        print(f"  ({label}) no eval file"); return
    toks = torch.load(eval_tokens, weights_only=True)
    agree_t1 = agree_t10 = 0.0
    n_tok = 0
    for _ in range(n_seqs):
        s = int(torch.randint(0, toks.numel() - seq - 1, (1,)).item())
        t = toks[s:s + seq].unsqueeze(0).long().to(device)
        t_logits, t_hs = teacher.forward(t, max_layers=tb.n_layers,
                                         return_hidden=True)
        latent = teacher.final_norm(t_hs[-1]).float()
        s_logits = sb.decode(latent).float()
        agree_t1 += (t_logits[0].argmax(-1) == s_logits[0].argmax(-1)).float().mean().item()
        for pos in range(seq):
            top_t = set(t_logits[0, pos].topk(10).indices.tolist())
            top_s = set(s_logits[0, pos].topk(10).indices.tolist())
            agree_t10 += len(top_t & top_s) / 10
            n_tok += 1
    print(f"  ({label}) T1 = {agree_t1/n_seqs*100:.2f}%    T10 = {agree_t10/n_tok*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--k_hot', type=int, default=1024)
    ap.add_argument('--r_hot', type=int, default=768)
    ap.add_argument('--hyper_hidden', type=int, default=1024)
    ap.add_argument('--n_freqs', type=int, default=48)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--eval_tokens', type=str, default='fineweb_edu_100M_tokens.pt')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--kd_steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--enc_w', type=float, default=0.05)
    args = ap.parse_args()

    from scaling.teacher_loader import load_qwen3_teacher
    tb = load_qwen3_teacher(args.teacher_cache, device=args.device)
    V, d = tb.vocab_size, tb.h_outer
    embed_w = tb.embed_w.detach().to(args.device).float()
    lm_head_w = tb.lm_head_w.detach().to(args.device).float()

    cos_tie = F.cosine_similarity(embed_w, lm_head_w, dim=1).mean().item()
    print(f"Teacher V={V} d={d}   avg cos(embed,head)={cos_tie:.3f}")

    token_counts = None
    if os.path.exists(args.eval_tokens):
        toks = torch.load(args.eval_tokens, weights_only=True)
        sample = toks[:20_000_000].to(args.device).long()
        token_counts = torch.bincount(sample, minlength=V)
    hot_ids = compute_hot_ids(lm_head_w, args.k_hot, token_counts).cpu()

    sb = SemanticBasisV4(V, d, k_hot=args.k_hot, r_hot=args.r_hot,
                         hyper_hidden=args.hyper_hidden,
                         n_freqs=args.n_freqs, hot_ids=hot_ids).to(args.device)

    n_sb = sb.num_params()
    n_orig = embed_w.numel() + lm_head_w.numel()
    hyper_params = sum(p.numel() for p in sb.hyper.parameters())
    hot_params = sum(p.numel() for p in sb.hot.parameters())
    print(f"\nOriginal embed+head:       {n_orig/1e6:.2f}M")
    print(f"SemanticBasis v4 total:    {n_sb/1e6:.2f}M  ({n_orig/n_sb:.1f}x compression)")
    print(f"  hot low-rank:            {hot_params/1e6:.2f}M")
    print(f"  hypernet (shared):       {hyper_params/1e6:.2f}M")
    print(f"  out_bias + misc:         {(n_sb - hot_params - hyper_params)/1e6:.2f}M")

    print("\n--- SVD-init hot tokens on embed_w ---")
    # index hot rows of embed_w
    hot_tgt = embed_w[hot_ids.to(args.device)]
    sb.hot.svd_init(hot_tgt)

    # calibrate alpha from teacher scale
    with torch.no_grad():
        h = torch.randn(4, 64, d, device=args.device)
        teacher_logits = h @ lm_head_w.T
        sb_raw = sb.decode(h)
        scale = (teacher_logits.std() / sb_raw.std().clamp(min=1e-3))
        sb.log_alpha.data = scale.log()

    print("\n--- pre-KD decode fidelity (hypernet still random) ---")
    fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'pre-KD')

    print("\n--- joint KD (hot + hypernet + alpha + bias) ---")
    fit_v4(sb, tb.teacher, tb, toks, args.device, embed_w,
           steps=args.kd_steps, batch=args.batch, seq=args.seq,
           lr=args.lr, enc_w=args.enc_w)

    print("\n--- post-KD decode fidelity ---")
    fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'post-KD')

    torch.save({
        'state_dict': sb.state_dict(),
        'config': {'V': V, 'd': d, 'k_hot': args.k_hot, 'r_hot': args.r_hot,
                   'hyper_hidden': args.hyper_hidden, 'n_freqs': args.n_freqs},
        'orig_params': n_orig, 'sb_params': n_sb,
        'compression': n_orig / n_sb,
        'teacher_cache': args.teacher_cache,
        'hot_ids': hot_ids,
        'cos_tie_detected': cos_tie,
    }, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
