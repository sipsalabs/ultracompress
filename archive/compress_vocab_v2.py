"""
SemanticBasis v2 — logit-space distillation with separate basis per role.

v1 mistake: fit both embed and lm_head to the average target with Frobenius
MSE. Problems:
  - Qwen3 does NOT tie embed and lm_head; averaging produced a target
    neither approximates well.
  - Frobenius MSE is the wrong proxy for softmax-rank preservation.
  - At r_cold=32 the cold bucket captured only 10% of its variance;
    Eckart-Young is tight so no amount of ALS helps at that rank.

v2 changes:
  1. Two separate factorizations (embed_basis, head_basis), each
     frequency-stratified hot/cold. Still dramatic savings, but not
     unrealistically tight.
  2. Fit embed via SVD (the target IS the matrix, pure reconstruction).
  3. Fit lm_head via SHORT gradient descent in LOGIT SPACE: sample real
     hidden states from the teacher, backprop KL(teacher_logits,
     sb_logits) through the factorization. 500 steps, ~10 min on GPU.
     This is the actual objective we care about.
  4. Rank budget: r_hot=768, r_cold=192. Vocab params ~60M, 10x down
     from 622M. Still the best single-invention compression of the vocab
     machinery in any distillation paper I know of.
  5. Evaluation includes both reconstruction error AND swap-in T1/T10
     on real teacher hidden states.

USAGE
    python compress_vocab_v2.py --teacher_cache qwen3_1.7b_cache.pt \
        --r_hot 768 --r_cold 192 --k_hot 1024 \
        --kd_steps 500 --kd_batch 4 --kd_seq 128 \
        --out qwen3_1.7b_sb2_r768_192_k1024.pt
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


class StratifiedLowRank(nn.Module):
    """A single low-rank factorization of a V x d matrix, hot/cold stratified.

    Represents   W[v, :] = U_hot[perm(v)] @ B_hot      (v in hot set, K tokens)
                W[v, :] = U_cold[perm(v)-K] @ B_cold   (v in cold set)

    Parameters count:
       K*r_hot + (V-K)*r_cold + (r_hot+r_cold)*d
    """

    def __init__(self, V, d, r_hot, r_cold, k_hot, hot_ids):
        super().__init__()
        assert hot_ids.numel() == k_hot
        self.V, self.d = V, d
        self.r_hot, self.r_cold, self.K = r_hot, r_cold, k_hot

        mask = torch.zeros(V, dtype=torch.bool)
        mask[hot_ids] = True
        cold_ids = torch.arange(V)[~mask]
        all_new_to_old = torch.cat([hot_ids.long(), cold_ids.long()], dim=0)
        old_to_new = torch.empty(V, dtype=torch.long)
        old_to_new[all_new_to_old] = torch.arange(V)
        self.register_buffer('old_to_new', old_to_new, persistent=True)
        self.register_buffer('new_to_old', all_new_to_old, persistent=True)

        self.B_hot = nn.Parameter(torch.randn(r_hot, d) / math.sqrt(r_hot))
        self.B_cold = nn.Parameter(torch.randn(r_cold, d) / math.sqrt(r_cold))
        self.U_hot = nn.Parameter(torch.randn(k_hot, r_hot) / math.sqrt(r_hot))
        self.U_cold = nn.Parameter(torch.randn(V - k_hot, r_cold) / math.sqrt(r_cold))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def svd_init(self, target):
        """Warm-start the factors to the best-Frobenius low-rank approximation."""
        dev = self.B_hot.device
        target = target.to(dev).float()
        tgt_new = target[self.new_to_old]
        hot_tgt = tgt_new[:self.K]
        cold_tgt = tgt_new[self.K:]
        U, S, Vh = torch.linalg.svd(hot_tgt, full_matrices=False)
        self.U_hot.data = (U[:, :self.r_hot] * S[:self.r_hot]).to(self.U_hot.dtype)
        self.B_hot.data = Vh[:self.r_hot].to(self.B_hot.dtype)
        U, S, Vh = torch.linalg.svd(cold_tgt, full_matrices=False)
        self.U_cold.data = (U[:, :self.r_cold] * S[:self.r_cold]).to(self.U_cold.dtype)
        self.B_cold.data = Vh[:self.r_cold].to(self.B_cold.dtype)

    def reconstruct(self):
        """Return W as a [V, d] dense matrix (original token order)."""
        E_new_hot = self.U_hot @ self.B_hot
        E_new_cold = self.U_cold @ self.B_cold
        E_new = torch.cat([E_new_hot, E_new_cold], dim=0)
        return E_new[self.old_to_new]

    def encode(self, tokens):
        """tokens: [B, T].  returns [B, T, d]."""
        new_tok = self.old_to_new[tokens]
        is_hot = new_tok < self.K
        out = torch.empty(*tokens.shape, self.d,
                          dtype=self.B_hot.dtype, device=tokens.device)
        if is_hot.any():
            out[is_hot] = self.U_hot[new_tok[is_hot]] @ self.B_hot
        if (~is_hot).any():
            out[~is_hot] = self.U_cold[new_tok[~is_hot] - self.K] @ self.B_cold
        return out

    def decode(self, latent, scale=None, bias=None):
        """Compute latent @ W^T efficiently without materializing W.

        latent: [..., d] -> logits [..., V]
        """
        hot_logits = (latent @ self.B_hot.T) @ self.U_hot.T
        cold_logits = (latent @ self.B_cold.T) @ self.U_cold.T
        logits_new = torch.cat([hot_logits, cold_logits], dim=-1)
        logits = logits_new.index_select(-1, self.old_to_new)
        if scale is not None:
            logits = logits * scale
        if bias is not None:
            logits = logits + bias
        return logits


class SemanticBasisV2(nn.Module):
    def __init__(self, V, d, r_hot, r_cold, k_hot, hot_ids):
        super().__init__()
        self.V, self.d = V, d
        self.embed_fact = StratifiedLowRank(V, d, r_hot, r_cold, k_hot, hot_ids)
        self.head_fact = StratifiedLowRank(V, d, r_hot, r_cold, k_hot, hot_ids)
        # Logit calibration for the head decode.
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.out_bias = nn.Parameter(torch.zeros(V))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, tokens):
        return self.embed_fact.encode(tokens)

    def decode(self, latent):
        alpha = self.log_alpha.exp()
        return self.head_fact.decode(latent, scale=alpha, bias=self.out_bias)


def compute_hot_ids(lm_head_weight, k_hot, token_counts=None):
    if token_counts is not None:
        order = torch.argsort(token_counts, descending=True)
        return order[:k_hot].to(torch.long)
    row_norms = lm_head_weight.norm(dim=1)
    order = torch.argsort(row_norms, descending=True)
    return order[:k_hot].to(torch.long)


def fit_logit_kd(sb: SemanticBasisV2, teacher, tb, all_tokens, device,
                 kd_steps=500, kd_batch=4, kd_seq=128, lr=1e-3, verbose=True):
    """Train head_fact by distilling teacher logits on real hidden states.

    We freeze embed_fact (it's fit via SVD -- Eckart-Young optimal for that
    objective). Only head parameters + calibration are trained.
    """
    head_params = list(sb.head_fact.parameters()) + [sb.log_alpha, sb.out_bias]
    opt = torch.optim.AdamW(head_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=kd_steps,
                                                       eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time()

    for step in range(kd_steps):
        starts = torch.randint(0, all_tokens.numel() - kd_seq, (kd_batch,))
        toks = torch.stack([all_tokens[s:s + kd_seq].long() for s in starts]).to(device)

        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=tb.n_layers, return_hidden=True)
            latent = teacher.final_norm(t_hs[-1]).float()
            t_logits = t_logits.float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            s_logits = sb.decode(latent)
            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            t_prob = t_logp.exp()
            # Forward KL (teacher || student).
            kl = (t_prob * (t_logp - s_logp)).sum(-1).mean()
            # Also CE vs teacher argmax (sharp target, helps T1).
            tgt = t_logits.argmax(-1)
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                                 tgt.view(-1))
            loss = kl + 0.3 * ce

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if verbose and (step % 50 == 0 or step == kd_steps - 1):
            # Quick T1 / T10 agreement on this batch.
            with torch.no_grad():
                s_argmax = s_logits.argmax(-1)
                t_argmax = tgt
                t1 = (s_argmax == t_argmax).float().mean().item()
                topk_t = t_logits.topk(10, dim=-1).indices
                topk_s = s_logits.topk(10, dim=-1).indices
                # t10: fraction of teacher top-10 present in student top-10.
                t10 = (topk_t.unsqueeze(-1) == topk_s.unsqueeze(-2)).any(-1).float().mean().item()
            elapsed = time.time() - t0
            print(f"  [KD] step={step:4d}  kl={kl.item():.3f}  ce={ce.item():.3f}  "
                  f"T1={t1*100:5.2f}%  T10={t10*100:5.2f}%  ({elapsed:.0f}s)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--r_hot', type=int, default=768)
    ap.add_argument('--r_cold', type=int, default=192)
    ap.add_argument('--k_hot', type=int, default=1024)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--eval_tokens', type=str, default='fineweb_edu_100M_tokens.pt')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--kd_steps', type=int, default=500)
    ap.add_argument('--kd_batch', type=int, default=4)
    ap.add_argument('--kd_seq', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    from scaling.teacher_loader import load_qwen3_teacher
    tb = load_qwen3_teacher(args.teacher_cache, device=args.device)
    V, d = tb.vocab_size, tb.h_outer
    embed_w = tb.embed_w.detach().to(args.device).float()
    lm_head_w = tb.lm_head_w.detach().to(args.device).float()
    print(f"Teacher vocab={V} d={d}")
    print(f"  embed  {embed_w.shape}   norm-mean={embed_w.norm(dim=1).mean():.3f}")
    print(f"  lmhead {lm_head_w.shape}  norm-mean={lm_head_w.norm(dim=1).mean():.3f}")
    cos_tie = F.cosine_similarity(embed_w, lm_head_w, dim=1).mean().item()
    print(f"  avg cos(embed[v], head[v]) = {cos_tie:.3f}  "
          f"(1.0 would mean they're literally tied)")

    # --- derive token frequency from a sample of the stream ---
    token_counts = None
    if os.path.exists(args.eval_tokens):
        toks = torch.load(args.eval_tokens, weights_only=True)
        sample = toks[:20_000_000].to(args.device).long() \
            if toks.numel() > 20_000_000 else toks.to(args.device).long()
        token_counts = torch.bincount(sample, minlength=V)
        print(f"  token counts from {sample.numel()/1e6:.0f}M tokens, "
              f"nonzero={int((token_counts>0).sum())}")
    hot_ids = compute_hot_ids(lm_head_w, args.k_hot, token_counts=token_counts).cpu()

    sb = SemanticBasisV2(V, d, args.r_hot, args.r_cold, args.k_hot,
                         hot_ids).to(args.device)
    n_sb = sb.num_params()
    n_orig = embed_w.numel() + lm_head_w.numel()
    print(f"\nOriginal embed+head: {n_orig/1e6:.2f}M")
    print(f"SemanticBasis v2:    {n_sb/1e6:.2f}M  ({n_orig/n_sb:.1f}x compression)")

    print("\n--- step 1: SVD-init embed (Eckart-Young optimal) ---")
    sb.embed_fact.svd_init(embed_w)
    with torch.no_grad():
        e_rec = sb.embed_fact.reconstruct()
        mse = ((e_rec - embed_w) ** 2).mean().item()
        var = embed_w.var().item()
        print(f"  embed reconstruction: MSE={mse:.5f}  rel_err={mse/var:.4f}")

    print("\n--- step 2: SVD-init head (warm-start for logit KD) ---")
    sb.head_fact.svd_init(lm_head_w)
    with torch.no_grad():
        h_rec = sb.head_fact.reconstruct()
        mse = ((h_rec - lm_head_w) ** 2).mean().item()
        var = lm_head_w.var().item()
        print(f"  head reconstruction:  MSE={mse:.5f}  rel_err={mse/var:.4f}")

    # --- calibrate alpha so scales match ---
    with torch.no_grad():
        h = torch.randn(4, 64, d, device=args.device)
        teacher_logits = h @ lm_head_w.T
        sb_raw_logits = sb.head_fact.decode(h)
        scale = (teacher_logits.std() / sb_raw_logits.std()).clamp(min=1e-3)
        sb.log_alpha.data = scale.log()
        sb.out_bias.data.zero_()
        print(f"  calibrated alpha = {scale.item():.4f}")

    print("\n--- step 3: pre-KD swap-in fidelity (SVD-only head) ---")
    fidelity_eval(sb, tb.teacher, tb, args, args.eval_tokens, args.device, label='pre-KD')

    print("\n--- step 4: logit-KD fine-tune of head factorization ---")
    toks = torch.load(args.eval_tokens, weights_only=True)
    fit_logit_kd(sb, tb.teacher, tb, toks, args.device,
                 kd_steps=args.kd_steps, kd_batch=args.kd_batch,
                 kd_seq=args.kd_seq, lr=args.lr)

    print("\n--- step 5: post-KD swap-in fidelity ---")
    fidelity_eval(sb, tb.teacher, tb, args, args.eval_tokens, args.device, label='post-KD')

    torch.save({
        'state_dict': sb.state_dict(),
        'config': {
            'V': V, 'd': d,
            'r_hot': args.r_hot, 'r_cold': args.r_cold, 'k_hot': args.k_hot,
        },
        'orig_params': n_orig,
        'sb_params': n_sb,
        'compression': n_orig / n_sb,
        'teacher_cache': args.teacher_cache,
        'hot_ids': hot_ids,
    }, args.out)
    print(f"\nSaved: {args.out}")


@torch.no_grad()
def fidelity_eval(sb, teacher, tb, args, eval_tokens, device, label):
    if not os.path.exists(eval_tokens):
        print(f"  ({label}) no eval file")
        return
    toks = torch.load(eval_tokens, weights_only=True)
    SEQ, n_seqs = 128, 60
    agree_t1 = agree_t10 = 0.0
    n_tok = 0
    for _ in range(n_seqs):
        s = int(torch.randint(0, toks.numel() - SEQ - 1, (1,)).item())
        t = toks[s:s + SEQ].unsqueeze(0).long().to(device)
        t_logits, t_hs = teacher.forward(t, max_layers=tb.n_layers, return_hidden=True)
        latent = teacher.final_norm(t_hs[-1]).float()
        s_logits = sb.decode(latent).float()
        t1_match = (t_logits[0].argmax(-1) == s_logits[0].argmax(-1)).float().mean().item()
        agree_t1 += t1_match
        for pos in range(SEQ):
            top_t = set(t_logits[0, pos].topk(10).indices.tolist())
            top_s = set(s_logits[0, pos].topk(10).indices.tolist())
            agree_t10 += len(top_t & top_s) / 10
            n_tok += 1
    print(f"  ({label}) T1 = {agree_t1/n_seqs*100:.2f}%    T10 = {agree_t10/n_tok*100:.2f}%")


if __name__ == '__main__':
    main()
