"""
SemanticBasis v3 — Tied factorization for tied-embedding models.

OBSERVATION (made during v2 fitting on Qwen3-1.7B):
    avg cos(embed[v], lm_head[v]) = 1.000

  Qwen3 uses tied weights: embed and lm_head are the SAME matrix
  (different names, same storage). Llama-3, Gemma, Mistral, Phi-3,
  GPT-NeoX all do this too. Only a few large GPT/Falcon variants
  untie them.

  v2 stored two independent factorizations (one per role) and burned
  ~32M params on a redundant copy. v3 stores ONE factorization and
  uses it both as encoder (token -> vector) and decoder (vector ->
  logits via inner product).

  Detection is automatic: if cos > 0.95 we use the tied path. Else we
  fall back to v2's untied dual factorization.

DOUBLE WIN
  Tied models     : ~2x further savings on top of v2 (~30M instead of 60M).
  Untied models   : same as v2 (60M).
  Unified API     : single output file works for both, picked by config.

NEW IDEA on top of tying — DECODE-AWARE FITTING (novel)
  When we know the same factor will be used both for encoding (E[v]
  appears as a row in many positions of an input embedding) and
  decoding (E[v] is dot-producted against many query latents at the
  output), the OPTIMAL low-rank approximation depends on BOTH
  distributions.

  Naive SVD on E alone optimizes only encode.
  Logit-KD alone (v2 head fitting) optimizes only decode.
  v3 jointly optimizes:
       L = lambda_enc * MSE( E_recon[t_in],  E[t_in] )      [over real input tokens]
         + lambda_dec * KL ( softmax(h E^T), softmax(h E_recon^T) )  [over real hidden states]

  This is the right loss for a tied-embedding model.
  Closed form for fixed-rank case is unknown (open problem). We solve
  with short SGD warm-started from SVD.

USAGE
  python compress_vocab_v3.py --teacher_cache qwen3_1.7b_cache.pt \
      --r_hot 768 --r_cold 192 --k_hot 1024 \
      --kd_steps 800 --out qwen3_1.7b_sb3.pt
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
    def __init__(self, V, d, r_hot, r_cold, k_hot, hot_ids):
        super().__init__()
        self.V, self.d = V, d
        self.r_hot, self.r_cold, self.K = r_hot, r_cold, k_hot
        mask = torch.zeros(V, dtype=torch.bool); mask[hot_ids] = True
        cold_ids = torch.arange(V)[~mask]
        new_to_old = torch.cat([hot_ids.long(), cold_ids.long()], dim=0)
        old_to_new = torch.empty(V, dtype=torch.long)
        old_to_new[new_to_old] = torch.arange(V)
        self.register_buffer('old_to_new', old_to_new, persistent=True)
        self.register_buffer('new_to_old', new_to_old, persistent=True)
        self.B_hot = nn.Parameter(torch.randn(r_hot, d) / math.sqrt(r_hot))
        self.B_cold = nn.Parameter(torch.randn(r_cold, d) / math.sqrt(r_cold))
        self.U_hot = nn.Parameter(torch.randn(k_hot, r_hot) / math.sqrt(r_hot))
        self.U_cold = nn.Parameter(torch.randn(V - k_hot, r_cold) / math.sqrt(r_cold))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def svd_init(self, target):
        dev = self.B_hot.device
        target = target.to(dev).float()
        tgt_new = target[self.new_to_old]
        hot_tgt, cold_tgt = tgt_new[:self.K], tgt_new[self.K:]
        U, S, Vh = torch.linalg.svd(hot_tgt, full_matrices=False)
        self.U_hot.data = (U[:, :self.r_hot] * S[:self.r_hot]).to(self.U_hot.dtype)
        self.B_hot.data = Vh[:self.r_hot].to(self.B_hot.dtype)
        U, S, Vh = torch.linalg.svd(cold_tgt, full_matrices=False)
        self.U_cold.data = (U[:, :self.r_cold] * S[:self.r_cold]).to(self.U_cold.dtype)
        self.B_cold.data = Vh[:self.r_cold].to(self.B_cold.dtype)

    def reconstruct(self):
        E_new = torch.cat([self.U_hot @ self.B_hot, self.U_cold @ self.B_cold], dim=0)
        return E_new[self.old_to_new]

    def encode(self, tokens):
        new_tok = self.old_to_new[tokens]
        is_hot = new_tok < self.K
        out = torch.empty(*tokens.shape, self.d, dtype=self.B_hot.dtype, device=tokens.device)
        if is_hot.any():
            out[is_hot] = (self.U_hot[new_tok[is_hot]] @ self.B_hot).to(out.dtype)
        if (~is_hot).any():
            out[~is_hot] = (self.U_cold[new_tok[~is_hot] - self.K] @ self.B_cold).to(out.dtype)
        return out

    def decode(self, latent, scale=None, bias=None):
        hot_logits = (latent @ self.B_hot.T) @ self.U_hot.T
        cold_logits = (latent @ self.B_cold.T) @ self.U_cold.T
        logits_new = torch.cat([hot_logits, cold_logits], dim=-1)
        logits = logits_new.index_select(-1, self.old_to_new)
        if scale is not None: logits = logits * scale
        if bias is not None: logits = logits + bias
        return logits


class SemanticBasisV3(nn.Module):
    """One stratified factorization, used for both encode and decode."""

    def __init__(self, V, d, r_hot, r_cold, k_hot, hot_ids, tied=True):
        super().__init__()
        self.V, self.d = V, d
        self.tied = tied
        self.tied_fact = StratifiedLowRank(V, d, r_hot, r_cold, k_hot, hot_ids)
        if not tied:
            self.head_fact = StratifiedLowRank(V, d, r_hot, r_cold, k_hot, hot_ids)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.out_bias = nn.Parameter(torch.zeros(V))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, tokens):
        return self.tied_fact.encode(tokens)

    def decode(self, latent):
        alpha = self.log_alpha.exp()
        fact = self.tied_fact if self.tied else self.head_fact
        return fact.decode(latent, scale=alpha, bias=self.out_bias)


def compute_hot_ids(lm_head_weight, k_hot, token_counts=None):
    if token_counts is not None:
        return torch.argsort(token_counts, descending=True)[:k_hot].to(torch.long)
    return torch.argsort(lm_head_weight.norm(dim=1), descending=True)[:k_hot].to(torch.long)


def fit_decode_aware(sb: SemanticBasisV3, teacher, tb, all_tokens, device,
                     embed_w, kd_steps=800, batch=4, seq=128, lr=1e-3,
                     enc_w=0.1, verbose=True):
    """Decode-aware joint fitting.

    Loss = enc_w * MSE(reconstruct(), embed_w)        # encode fidelity
         + KL(teacher_logits, sb.decode(teacher_h))   # decode fidelity
         + 0.3 * CE(sb.decode(teacher_h), teacher.argmax)   # T1 sharpening
    """
    params = list(sb.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=kd_steps, eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    embed_w_dev = embed_w.to(device).float()
    t0 = time.time()

    for step in range(kd_steps):
        starts = torch.randint(0, all_tokens.numel() - seq, (batch,))
        toks = torch.stack([all_tokens[s:s + seq].long() for s in starts]).to(device)
        with torch.no_grad():
            t_logits, t_hs = teacher.forward(toks, max_layers=tb.n_layers, return_hidden=True)
            latent = teacher.final_norm(t_hs[-1]).float()
            t_logits = t_logits.float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # decode loss
            s_logits = sb.decode(latent)
            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            t_prob = t_logp.exp()
            kl = (t_prob * (t_logp - s_logp)).sum(-1).mean()
            tgt = t_logits.argmax(-1)
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), tgt.view(-1))

            # encode loss (only on a sampled subset to keep cheap)
            if enc_w > 0:
                sample_ids = torch.randint(0, sb.V, (4096,), device=device)
                rec_sample = sb.tied_fact.encode(sample_ids.unsqueeze(0))[0]
                tgt_sample = embed_w_dev[sample_ids]
                enc_mse = ((rec_sample.float() - tgt_sample) ** 2).mean()
            else:
                enc_mse = torch.tensor(0.0, device=device)

            loss = kl + 0.3 * ce + enc_w * enc_mse

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if verbose and (step % 50 == 0 or step == kd_steps - 1):
            with torch.no_grad():
                t1 = (s_logits.argmax(-1) == tgt).float().mean().item()
                topk_t = t_logits.topk(10, -1).indices
                topk_s = s_logits.topk(10, -1).indices
                t10 = (topk_t.unsqueeze(-1) == topk_s.unsqueeze(-2)).any(-1).float().mean().item()
            print(f"  [KD] step={step:4d}  kl={kl.item():.3f}  ce={ce.item():.3f}  "
                  f"enc={enc_mse.item():.5f}  T1={t1*100:5.2f}%  T10={t10*100:5.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def fidelity_eval(sb, teacher, tb, eval_tokens, device, label, n_seqs=60, seq=128):
    if not os.path.exists(eval_tokens):
        print(f"  ({label}) no eval file"); return
    toks = torch.load(eval_tokens, weights_only=True)
    agree_t1 = agree_t10 = 0.0
    n_tok = 0
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
            agree_t10 += len(top_t & top_s) / 10
            n_tok += 1
    print(f"  ({label}) T1 = {agree_t1/n_seqs*100:.2f}%    T10 = {agree_t10/n_tok*100:.2f}%")


@torch.no_grad()
def encode_fidelity(sb, embed_w, device, n_sample=4096):
    sample_ids = torch.randint(0, sb.V, (n_sample,), device=device)
    rec = sb.tied_fact.encode(sample_ids.unsqueeze(0))[0].float()
    tgt = embed_w.to(device).float()[sample_ids]
    mse = ((rec - tgt) ** 2).mean().item()
    var = embed_w.var().item()
    cos = F.cosine_similarity(rec, tgt, dim=-1).mean().item()
    return mse, mse / var, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--r_hot', type=int, default=768)
    ap.add_argument('--r_cold', type=int, default=192)
    ap.add_argument('--k_hot', type=int, default=1024)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--eval_tokens', type=str, default='fineweb_edu_100M_tokens.pt')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--kd_steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--enc_w', type=float, default=0.1,
                    help='weight on encode-fidelity term (0 to disable)')
    ap.add_argument('--force_untied', action='store_true',
                    help='disable auto-tied detection (use v2-style dual factorization)')
    args = ap.parse_args()

    from scaling.teacher_loader import load_qwen3_teacher
    tb = load_qwen3_teacher(args.teacher_cache, device=args.device)
    V, d = tb.vocab_size, tb.h_outer
    embed_w = tb.embed_w.detach().to(args.device).float()
    lm_head_w = tb.lm_head_w.detach().to(args.device).float()

    cos_tie = F.cosine_similarity(embed_w, lm_head_w, dim=1).mean().item()
    auto_tied = (cos_tie > 0.95) and not args.force_untied
    print(f"Teacher V={V} d={d}")
    print(f"  avg cos(embed,head)={cos_tie:.3f}  -> tied={auto_tied}")

    token_counts = None
    if os.path.exists(args.eval_tokens):
        toks = torch.load(args.eval_tokens, weights_only=True)
        sample = toks[:20_000_000].to(args.device).long()
        token_counts = torch.bincount(sample, minlength=V)
    hot_ids = compute_hot_ids(lm_head_w, args.k_hot, token_counts).cpu()

    sb = SemanticBasisV3(V, d, args.r_hot, args.r_cold, args.k_hot,
                         hot_ids, tied=auto_tied).to(args.device)

    n_sb = sb.num_params()
    n_orig = embed_w.numel() + lm_head_w.numel()
    print(f"\nOriginal embed+head: {n_orig/1e6:.2f}M")
    print(f"SemanticBasis v3:    {n_sb/1e6:.2f}M  ({n_orig/n_sb:.1f}x compression)  tied={auto_tied}")

    print("\n--- step 1: SVD-init tied factor on embed ---")
    sb.tied_fact.svd_init(embed_w)
    if not auto_tied:
        print("--- step 1b: SVD-init head factor on lm_head (untied path) ---")
        sb.head_fact.svd_init(lm_head_w)

    # calibrate alpha
    with torch.no_grad():
        h = torch.randn(4, 64, d, device=args.device)
        teacher_logits = h @ lm_head_w.T
        sb_raw = sb.decode(h)
        scale = (teacher_logits.std() / sb_raw.std()).clamp(min=1e-3)
        sb.log_alpha.data = scale.log()

    print("\n--- pre-KD encode/decode fidelity ---")
    mse, rel, cos = encode_fidelity(sb, embed_w, args.device)
    print(f"  encode: MSE={mse:.5f}  rel_err={rel:.4f}  cos={cos:.4f}")
    fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'pre-KD')

    print("\n--- decode-aware joint KD fine-tune ---")
    fit_decode_aware(sb, tb.teacher, tb, toks, args.device, embed_w,
                     kd_steps=args.kd_steps, batch=args.batch, seq=args.seq,
                     lr=args.lr, enc_w=args.enc_w)

    print("\n--- post-KD encode/decode fidelity ---")
    mse, rel, cos = encode_fidelity(sb, embed_w, args.device)
    print(f"  encode: MSE={mse:.5f}  rel_err={rel:.4f}  cos={cos:.4f}")
    fidelity_eval(sb, tb.teacher, tb, args.eval_tokens, args.device, 'post-KD')

    torch.save({
        'state_dict': sb.state_dict(),
        'config': {'V': V, 'd': d, 'r_hot': args.r_hot, 'r_cold': args.r_cold,
                   'k_hot': args.k_hot, 'tied': auto_tied},
        'orig_params': n_orig, 'sb_params': n_sb,
        'compression': n_orig / n_sb,
        'teacher_cache': args.teacher_cache,
        'hot_ids': hot_ids,
        'cos_tie_detected': cos_tie,
    }, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
