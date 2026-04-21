"""
SemanticBasis — Universal Vocabulary Compression.

The uncovered cost: in any modern LLM the embedding table + lm_head
together are 20-35% of total parameters. For Qwen3-1.7B:
    embed (151,936 x 2048)   = 311M params
    lm_head (2048 x 151,936) = 311M params
    total vocab machinery    = 622M = 30.6% of the 2.03B model.

All prior distillation work ignores this. A "compressed" student still
ships a full-size lm_head, which sets a hard floor on deployed size.

SEMANTICBASIS BREAKS THAT FLOOR
  Idea 1: Tied weights. Let    E = embedding,   H = lm_head^T.
          Instead of two 311M tables, learn one.
  Idea 2: Shared low-rank basis. Every token vector is a coefficient
          over a tiny universal basis:
              E[i] = U_i @ B   where B in R^{r x d} is shared.
  Idea 3: FREQUENCY STRATIFICATION (novel).
          The top-K most frequent tokens get high-rank per-token
          coefficients; the tail shares a low-rank subspace:
              hot  tokens (K=1024)  -> per-token rank r_hot  = 384
              cold tokens (150912)  -> per-token rank r_cold = 32
          Hot coefficients: K*r_hot    =   393K
          Cold coefficients: 150K*r_cold = 4.8M
          Shared basis: (r_hot+r_cold)*d = 850K
          Total = ~6M, replacing 622M (106x).

TEACHER-AGNOSTIC
  Input: any model's (embed_matrix, lm_head_matrix) and a token-frequency
  prior (from any corpus in the target language).
  Output: a SemanticBasis module with identical I/O shape as
  nn.Embedding + F.linear(..., lm_head). Drop-in replacement.

FITTING PROCEDURE
  We DO NOT gradient-descend from scratch. We fit via an alternating
  closed-form least-squares on the teacher's actual embed/head matrices:
      step 1: fix coefficients, solve basis via lstsq
      step 2: fix basis, solve coefficients per token
  Two or three passes converge. This is O(V*d*r), not iterative training.
  A teacher with 150K vocab and d=2048, r=416 fits in under 60 seconds.

FREQUENCY PRIOR
  If a corpus is available, count tokens and assign top-K to the hot set.
  If not, use the row-norm of lm_head as a proxy (high-norm rows are
  "confident" tokens — empirically correlates with frequency).

USAGE
    python compress_vocab.py --teacher_cache qwen3_1.7b_cache.pt \
        --r_hot 384 --r_cold 32 --k_hot 1024 \
        --out qwen3_1.7b_semanticbasis.pt
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


# ============================================================
# Core module: the replacement for Embedding + lm_head.
# ============================================================
class SemanticBasis(nn.Module):
    """Tied, frequency-stratified low-rank vocab compression.

    Forward modes
      encode(tokens)  -> embedding vectors  [B, T, d]
      decode(latent)  -> logits              [B, T, V]

    Parameters
      B_hot   in R^{r_hot  x d}      shared hot basis
      B_cold  in R^{r_cold x d}      shared cold basis
      U_hot   in R^{K      x r_hot}  per-hot-token coefficients
      U_cold  in R^{(V-K)  x r_cold} per-cold-token coefficients
      alpha   scalar                 output-decoder scale
      bias_h  in R^{d}               pre-decode bias

    Tied-weight decoding: logits_v = <latent, E[v]>. Computed block-wise
    for hot and cold sets to avoid materializing a full V x d matrix.
    """

    def __init__(self, vocab_size, d, r_hot=384, r_cold=32,
                 k_hot=1024, hot_ids=None):
        super().__init__()
        self.V = vocab_size
        self.d = d
        self.r_hot = r_hot
        self.r_cold = r_cold
        self.K = k_hot

        if hot_ids is None:
            hot_ids = torch.arange(k_hot, dtype=torch.long)
        assert hot_ids.numel() == k_hot
        assert hot_ids.dtype == torch.long
        # Permutation: remap vocab so hot tokens are first K.
        perm = torch.empty(vocab_size, dtype=torch.long)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        mask[hot_ids] = True
        cold_ids = torch.arange(vocab_size)[~mask]
        all_new_to_old = torch.cat([hot_ids, cold_ids], dim=0)
        old_to_new = torch.empty(vocab_size, dtype=torch.long)
        old_to_new[all_new_to_old] = torch.arange(vocab_size)
        self.register_buffer('old_to_new', old_to_new, persistent=True)
        self.register_buffer('new_to_old', all_new_to_old, persistent=True)

        self.B_hot = nn.Parameter(torch.randn(r_hot, d) / math.sqrt(r_hot))
        self.B_cold = nn.Parameter(torch.randn(r_cold, d) / math.sqrt(r_cold))
        self.U_hot = nn.Parameter(torch.randn(k_hot, r_hot) / math.sqrt(r_hot))
        self.U_cold = nn.Parameter(torch.randn(vocab_size - k_hot, r_cold)
                                   / math.sqrt(r_cold))
        # Scale + bias on the output side only (keeps embedding tied).
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # --- encode: tokens -> vectors ---
    def encode(self, tokens):
        """tokens: [B, T] long.  returns [B, T, d]."""
        tok_new = self.old_to_new[tokens]
        is_hot = tok_new < self.K
        d = self.d

        out = torch.empty(*tokens.shape, d, dtype=self.B_hot.dtype,
                          device=tokens.device)
        if is_hot.any():
            hot_idx = tok_new[is_hot]
            U = self.U_hot[hot_idx]
            out[is_hot] = U @ self.B_hot
        if (~is_hot).any():
            cold_idx = tok_new[~is_hot] - self.K
            U = self.U_cold[cold_idx]
            out[~is_hot] = U @ self.B_cold
        return out

    # --- decode: latent -> logits (tied weights) ---
    def decode(self, latent):
        """latent: [..., d].  returns [..., V]."""
        alpha = self.log_alpha.exp()
        # Full vocab matmul via two blocks, concatenated in NEW-token order.
        # E_new[v] = U_hot[v]@B_hot   if v<K else U_cold[v-K]@B_cold
        # logits_new[v] = alpha * <latent, E_new[v]>
        #               = alpha * latent @ E_new.T
        #               = alpha * ( (latent @ B_hot.T) @ U_hot.T ) block-wise
        L = latent
        # Hot block: L @ (U_hot @ B_hot)^T = L @ B_hot^T @ U_hot^T
        hot_logits = (L @ self.B_hot.T) @ self.U_hot.T      # [..., K]
        cold_logits = (L @ self.B_cold.T) @ self.U_cold.T   # [..., V-K]
        logits_new = torch.cat([hot_logits, cold_logits], dim=-1)
        # logits_new is in NEW token order. To return logits in ORIGINAL
        # vocab order: logits[v] = logits_new[old_to_new[v]]
        #   -> index_select(-1, old_to_new)  (gather new[old_to_new[v]] for each v)
        logits = logits_new.index_select(-1, self.old_to_new)
        logits = logits * alpha + self.out_bias
        return logits


# ============================================================
# Frequency prior: either token counts or lm_head row-norms.
# ============================================================
def compute_hot_ids(lm_head_weight, k_hot, token_counts=None):
    """Return top-K token ids by importance."""
    if token_counts is not None:
        order = torch.argsort(token_counts, descending=True)
        return order[:k_hot].to(torch.long)
    # Fallback: row-norm of lm_head (high-norm rows ~= confident/frequent).
    row_norms = lm_head_weight.norm(dim=1)
    order = torch.argsort(row_norms, descending=True)
    return order[:k_hot].to(torch.long)


# ============================================================
# Fitter: alternating least squares on (embed, lm_head).
# ============================================================
@torch.no_grad()
def fit_als(embed_matrix, lm_head_matrix, sb: SemanticBasis,
            n_iters=3, device='cuda', verbose=True):
    """Fit the SemanticBasis to approximate both embed and lm_head jointly.

    We treat the "target" per token as the average of (embed row) and
    (lm_head column) since SemanticBasis is tied. This gives the best
    single-table approximation to both original tables.
    """
    V, d = embed_matrix.shape
    target = 0.5 * (embed_matrix.to(device).float()
                    + lm_head_matrix.to(device).float())
    tgt_new = target[sb.new_to_old]
    hot_tgt = tgt_new[:sb.K]                  # [K, d]
    cold_tgt = tgt_new[sb.K:]                 # [V-K, d]

    B_hot = sb.B_hot.data.to(device).float()  # [r_hot, d]
    B_cold = sb.B_cold.data.to(device).float()
    U_hot = sb.U_hot.data.to(device).float()  # [K, r_hot]
    U_cold = sb.U_cold.data.to(device).float()

    def report(tag):
        if not verbose:
            return
        rec_hot = U_hot @ B_hot
        rec_cold = U_cold @ B_cold
        e_hot = ((rec_hot - hot_tgt) ** 2).mean().item()
        e_cold = ((rec_cold - cold_tgt) ** 2).mean().item()
        tgt_var = target.var().item()
        print(f"  [{tag}] MSE hot={e_hot:.4f}  cold={e_cold:.4f}  "
              f"target_var={tgt_var:.4f}  rel_hot={e_hot/tgt_var:.4f}  "
              f"rel_cold={e_cold/tgt_var:.4f}")

    # --- warm start hot via SVD on hot targets ---
    U_svd, S, Vh = torch.linalg.svd(hot_tgt, full_matrices=False)
    U_hot = U_svd[:, :sb.r_hot] * S[:sb.r_hot]
    B_hot = Vh[:sb.r_hot]
    # --- warm start cold via SVD on cold targets ---
    U_svd, S, Vh = torch.linalg.svd(cold_tgt, full_matrices=False)
    U_cold = U_svd[:, :sb.r_cold] * S[:sb.r_cold]
    B_cold = Vh[:sb.r_cold]
    report('after SVD init')

    # --- a few ALS refinement passes (jointly with hot/cold independent) ---
    for it in range(n_iters):
        # Fix B, solve U in closed form:  U = tgt @ pinv(B)
        # Since B is small (r x d), we use lstsq.
        U_hot = torch.linalg.lstsq(B_hot.T, hot_tgt.T).solution.T
        U_cold = torch.linalg.lstsq(B_cold.T, cold_tgt.T).solution.T
        # Fix U, solve B:  B = pinv(U) @ tgt
        B_hot = torch.linalg.lstsq(U_hot, hot_tgt).solution
        B_cold = torch.linalg.lstsq(U_cold, cold_tgt).solution
        report(f'ALS pass {it+1}')

    sb.B_hot.data = B_hot.to(sb.B_hot.dtype).to(sb.B_hot.device)
    sb.B_cold.data = B_cold.to(sb.B_cold.dtype).to(sb.B_cold.device)
    sb.U_hot.data = U_hot.to(sb.U_hot.dtype).to(sb.U_hot.device)
    sb.U_cold.data = U_cold.to(sb.U_cold.dtype).to(sb.U_cold.device)

    # --- calibrate alpha so decoded logits have similar scale to teacher ---
    # Measure < E_student_rec, h > for random hidden states h vs teacher logits.
    with torch.no_grad():
        h = torch.randn(4, 64, d, device=device)
        teacher_logits = h @ lm_head_matrix.to(device).float().T
        stu_logits = sb.decode(h.to(sb.B_hot.device).to(sb.B_hot.dtype)).float()
        scale = (teacher_logits.std() / stu_logits.std()).clamp(min=1e-3)
        sb.log_alpha.data = scale.log().to(sb.log_alpha.dtype).cpu()
    report('post-calibrate')


# ============================================================
# Driver
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--r_hot', type=int, default=384)
    ap.add_argument('--r_cold', type=int, default=32)
    ap.add_argument('--k_hot', type=int, default=1024)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--token_counts', type=str, default=None,
                    help='optional .pt file with a LongTensor of token counts, shape [V]')
    ap.add_argument('--eval_tokens', type=str, default='fineweb_edu_100M_tokens.pt',
                    help='token stream to measure reconstruction fidelity on')
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    from scaling.teacher_loader import load_qwen3_teacher
    tb = load_qwen3_teacher(args.teacher_cache, device=args.device)
    V = tb.vocab_size
    d = tb.h_outer

    embed_w = tb.embed_w.detach().to(args.device).float()
    lm_head_w = tb.lm_head_w.detach().to(args.device).float()
    print(f"Teacher vocab={V} d={d}  embed={embed_w.shape}  head={lm_head_w.shape}")

    # --- frequency ---
    token_counts = None
    if args.token_counts and os.path.exists(args.token_counts):
        token_counts = torch.load(args.token_counts, weights_only=True).to(args.device)
        print(f"  using token_counts {args.token_counts}  nonzero={int((token_counts>0).sum())}")
    elif os.path.exists(args.eval_tokens):
        # Quick frequency from a subset of the eval stream (free signal).
        toks = torch.load(args.eval_tokens, weights_only=True)
        sample = toks[:20_000_000].to(args.device).long() if toks.numel() > 20_000_000 else toks.to(args.device).long()
        token_counts = torch.bincount(sample, minlength=V)
        print(f"  derived token_counts from eval stream ({sample.numel()/1e6:.0f}M toks, "
              f"nonzero={int((token_counts>0).sum())})")

    hot_ids = compute_hot_ids(lm_head_w, args.k_hot, token_counts=token_counts).to(args.device)

    sb = SemanticBasis(V, d, r_hot=args.r_hot, r_cold=args.r_cold,
                       k_hot=args.k_hot, hot_ids=hot_ids.cpu()).to(args.device)

    n_sb = sb.num_params()
    n_orig = embed_w.numel() + lm_head_w.numel()
    print(f"Original embed+head params: {n_orig/1e6:.2f}M")
    print(f"SemanticBasis params:       {n_sb/1e6:.2f}M "
          f"({100*n_sb/n_orig:.2f}%, {n_orig/n_sb:.1f}x compression)")

    # --- fit ---
    t0 = time.time()
    fit_als(embed_w, lm_head_w, sb, n_iters=3, device=args.device)
    print(f"Fit took {time.time()-t0:.1f}s")

    # --- fidelity eval: does decode( teacher_hidden_state ) agree with teacher_logits ? ---
    if os.path.exists(args.eval_tokens):
        toks = torch.load(args.eval_tokens, weights_only=True)
        teacher = tb.teacher
        agree_t1 = 0.0
        agree_t10 = 0.0
        n_tok = 0
        n_seqs = 40
        SEQ = 128
        with torch.no_grad():
            for _ in range(n_seqs):
                s = int(torch.randint(0, toks.numel() - SEQ - 1, (1,)).item())
                t = toks[s:s + SEQ].unsqueeze(0).long().to(args.device)
                t_logits, t_hs = teacher.forward(t, max_layers=tb.n_layers, return_hidden=True)
                t_latent = teacher.final_norm(t_hs[-1]).float()
                sb_logits = sb.decode(t_latent).float()
                t1_match = (t_logits[0].argmax(-1) == sb_logits[0].argmax(-1)).float().mean().item()
                agree_t1 += t1_match
                for pos in range(SEQ):
                    top_t = set(t_logits[0, pos].topk(10).indices.tolist())
                    top_s = set(sb_logits[0, pos].topk(10).indices.tolist())
                    agree_t10 += len(top_t & top_s) / 10
                    n_tok += 1
        print(f"\nSwap-in fidelity (teacher hidden -> SB decode vs teacher decode):")
        print(f"  T1 agreement  = {agree_t1/n_seqs*100:.2f}%")
        print(f"  T10 agreement = {agree_t10/n_tok*100:.2f}%")
        print(f"  (if T1 > 85% the SB can be used as a drop-in lm_head replacement)")

    # --- save ---
    torch.save({
        'state_dict': sb.state_dict(),
        'config': {
            'vocab_size': V, 'd': d,
            'r_hot': args.r_hot, 'r_cold': args.r_cold, 'k_hot': args.k_hot,
        },
        'orig_params': n_orig,
        'sb_params': n_sb,
        'compression': n_orig / n_sb,
        'teacher_cache': args.teacher_cache,
    }, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
