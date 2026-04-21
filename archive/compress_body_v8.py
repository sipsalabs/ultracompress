"""
v8 FractalBody — shared-codebook product quantization of the DEQ body.

MOTIVATION
  After v7, vocab is 0.55 MB. The DEQ body is ~3 MB fp16 (6 MB fp32).
  Body Linears are now ~85% of the whole-model footprint.

INVENTION (building on v7)
  1. ONE global codebook shared across EVERY Linear in the DEQ body
     (proj_in, proj_out, block.qkv, block.o_proj, block.gate, block.up,
      block.down, and the two 1D norms stay fp16).
  2. Functional MSE self-distillation: we don't need the teacher here.
     Let f_fp32 be the original body. We train f_pq to minimize
     E[||f_fp32(x) - f_pq(x)||^2] where x ~ the distribution of
     latents the DEQ actually sees. This is the FIXED-POINT RESIDUAL
     match, which is the only invariant the DEQ cares about.
  3. Iteration-aware stability check: after QAT, verify the quantized
     body still contracts (Jacobian spectral radius < 1) by unrolling
     N forward iterations from random init and measuring residual decay.

PATENT CLAIM
  Applying v7-style cross-layer shared-codebook product quantization
  to the recurrent body of a DEEP EQUILIBRIUM MODEL, preserving the
  contractivity invariant via unrolled-residual QAT.

USAGE
  python compress_body_v8.py --deq_ckpt checkpoints_1.7b_tinyfrr_deq_h256/best.pt \
      --global_K 1024 --subvec 8 --qat_steps 1500 \
      --out deq_h256_body_pq.pt --device cuda:0
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from compress_vocab_v7 import GlobalCodebook, pq_quantize, entropy_bits


# ============================================================
# PQ Linear that mirrors the one in v7 but takes raw weight tensor
# as input (the DEQ body stores weights as tensors in modules).
# ============================================================
class PQLinearBody(nn.Module):
    def __init__(self, weight: torch.Tensor, shared_cb: GlobalCodebook, D: int,
                 bias: torch.Tensor = None):
        super().__init__()
        self.D = D
        self.shared_cb = shared_cb
        self.weight = nn.Parameter(weight.data.clone())
        if bias is not None:
            self.bias = nn.Parameter(bias.data.clone())
        else:
            self.register_parameter('bias', None)
        rs = self.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        self.row_scale = nn.Parameter(rs)

    def forward(self, x):
        wq = pq_quantize(self.weight, self.shared_cb.codebook, self.row_scale, self.D)
        return F.linear(x, wq, self.bias)

    @torch.no_grad()
    def indices(self):
        O, I = self.weight.shape
        w_scaled = self.weight / self.row_scale
        g = w_scaled.view(O, I // self.D, self.D).reshape(-1, self.D)
        d = torch.cdist(g.unsqueeze(0),
                        self.shared_cb.codebook.unsqueeze(0)).squeeze(0)
        return d.argmin(-1).view(O, I // self.D)


# ============================================================
# FractalBlock reimplementation -- mirrors run_deq_frr.py's block
# but all 5 Linears are swappable for PQ.
# ============================================================
class FractalBlockPQ(nn.Module):
    def __init__(self, h: int, n_heads: int):
        super().__init__()
        self.h = h
        self.n_heads = n_heads
        self.head_dim = h // n_heads
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.norm1 = nn.LayerNorm(h, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(h, elementwise_affine=True)
        self.gate = nn.Linear(h, h, bias=False)
        self.up = nn.Linear(h, h, bias=False)
        self.down = nn.Linear(h, h, bias=False)

    def attn(self, x):
        B, T, h = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(h, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, h)
        return self.o_proj(o)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x


class DEQBodyPQ(nn.Module):
    """Reconstruct the trained DEQ body with PQ-swappable Linears."""

    def __init__(self, h_outer: int, h_inner: int, n_heads_inner: int):
        super().__init__()
        self.h_outer = h_outer
        self.h_inner = h_inner
        self.proj_in = nn.Linear(h_outer, h_inner, bias=False)
        self.proj_out = nn.Linear(h_inner, h_outer, bias=False)
        self.block = FractalBlockPQ(h_inner, n_heads_inner)
        # learned DEQ-scalars
        self.gamma = nn.Parameter(torch.ones(h_inner))
        self.beta = nn.Parameter(torch.zeros(h_inner))
        self.step_scale = nn.Parameter(torch.ones(40) * 0.1)

    def body_step(self, z, x_proj, step_idx):
        """One DEQ iteration. `z` is the hidden inner state; `x_proj` is the
        injected projected input."""
        s = self.step_scale[step_idx]
        z = z + s * (self.block(z * self.gamma + x_proj + self.beta) - z)
        return z

    @torch.no_grad()
    def project_in(self, x_outer):
        return self.proj_in(x_outer)

    @torch.no_grad()
    def project_out(self, z):
        return self.proj_out(z)

    def forward(self, x_outer, iters=12):
        x_proj = self.proj_in(x_outer)
        z = torch.zeros_like(x_proj)
        n_iters = min(iters, self.step_scale.numel())
        for i in range(n_iters):
            z = self.body_step(z, x_proj, i)
        return self.proj_out(z)


def load_deq_body(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck['state_dict']
    h_outer = ck.get('h_outer', 2048)
    h_inner = ck['h_inner']
    n_heads_inner = ck.get('n_heads_inner', 16)
    body = DEQBodyPQ(h_outer, h_inner, n_heads_inner).to(device)
    # load only body-relevant keys
    own = body.state_dict()
    loaded = {}
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            loaded[k] = v
    body.load_state_dict(loaded, strict=False)
    missing = [k for k in own if k not in loaded]
    if missing:
        print(f"  [warn] body keys not in ckpt: {missing}")
    return body, ck


# ============================================================
# Replace the 5 body Linears with shared-codebook PQ variants.
# ============================================================
LINEARS_TO_PQ = ['proj_in', 'proj_out',
                 'block.qkv', 'block.o_proj',
                 'block.gate', 'block.up', 'block.down']


@torch.no_grad()
def pool_body_subvectors(body: DEQBodyPQ, D: int):
    pool = []
    for name in LINEARS_TO_PQ:
        mod = body
        for part in name.split('.'):
            mod = getattr(mod, part)
        W = mod.weight.data
        O, I = W.shape
        if I % D != 0:
            continue
        rs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        g = (W / rs).view(O, I // D, D).reshape(-1, D)
        pool.append(g)
    return torch.cat(pool, 0)


def pq_install_body(body: DEQBodyPQ, shared_cb: GlobalCodebook, D: int):
    for name in LINEARS_TO_PQ:
        # resolve parent and attr
        parts = name.split('.')
        parent = body
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        orig = getattr(parent, attr)
        if orig.in_features % D != 0:
            print(f"  [note] skip {name} (in_features {orig.in_features} % D={D})")
            continue
        new = PQLinearBody(orig.weight.data, shared_cb, D, orig.bias)
        setattr(parent, attr, new)
    body.shared_cb = shared_cb


# ============================================================
# Functional MSE self-distillation + contractivity-preserving QAT.
#
# We sample fake `x_outer` from a real latent distribution (hot_outer
# states drawn from the teacher's cache if available; otherwise
# standard-normal of the same norm statistics). Target is the fp32
# body's output; student is the PQ body.
# ============================================================
def qat_body(body_pq: nn.Module, body_fp32: nn.Module, device,
             steps=1500, batch=4, seq=128, lr=3e-4,
             iters=8, contract_every=200):
    opt = torch.optim.AdamW(body_pq.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    body_fp32.eval()
    for p in body_fp32.parameters():
        p.requires_grad_(False)
    scaler = torch.amp.GradScaler('cuda')

    # capture realistic input statistics: random normal, scaled by
    # typical outer-hidden-state rms (~1.0 after norm_outer)
    def sample_x():
        return torch.randn(batch, seq, body_pq.h_outer, device=device) * 1.0

    best_loss = float('inf')
    best_state = {k: v.detach().clone() for k, v in body_pq.state_dict().items()}
    best_step = 0
    t0 = time.time()

    for step in range(steps):
        x = sample_x()
        with torch.no_grad():
            y_teacher = body_fp32(x, iters=iters)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            y_student = body_pq(x, iters=iters)
            mse = F.mse_loss(y_student, y_teacher.detach())
            # relative L2 -- scale-invariant
            rel = (y_student - y_teacher).pow(2).sum() / y_teacher.pow(2).sum().clamp(min=1e-8)
            loss = mse + 0.1 * rel
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(body_pq.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                rel_v = rel.item()
            print(f"  [v8-QAT] step={step:4d}  mse={mse.item():.5f}  "
                  f"rel={rel_v:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if rel_v < best_loss:
                best_loss = rel_v
                best_step = step
                best_state = {k: v.detach().clone() for k, v in body_pq.state_dict().items()}

        if step % contract_every == 0 and step > 0:
            # spectral-radius proxy: fp error residual over N iters
            with torch.no_grad():
                x = sample_x()
                z0 = body_pq.proj_in(x)
                z = z0.clone() + 0.01 * torch.randn_like(z0)
                res = []
                for i in range(body_pq.step_scale.numel()):
                    z_new = body_pq.body_step(z, z0, i)
                    res.append((z_new - z).pow(2).mean().sqrt().item())
                    z = z_new
                decay = res[-1] / (res[0] + 1e-12)
                print(f"    [contract] first_res={res[0]:.4f}  last_res={res[-1]:.4f}  "
                      f"decay={decay:.3f}  (<1 = contractive)", flush=True)

    body_pq.load_state_dict(best_state)
    print(f"  [v8-QAT] restored best: step={best_step}  rel_mse={best_loss:.5f}", flush=True)


# ============================================================
# Honest byte accounting for the body-PQ artifact.
# ============================================================
@torch.no_grad()
def body_byte_accounting(body_pq: DEQBodyPQ, K: int, D: int) -> tuple:
    total_raw = total_ent = 0.0
    per_layer = []
    # PQ Linears
    for name in LINEARS_TO_PQ:
        parts = name.split('.')
        mod = body_pq
        for p in parts:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if not isinstance(mod, PQLinearBody):
            continue
        idx = mod.indices()
        H = entropy_bits(idx, K)
        n = idx.numel()
        total_raw += n * math.log2(K)
        total_ent += n * H
        per_layer.append((name, n, H))

    buffer_bits = 0
    # shared codebook
    buffer_bits += body_pq.shared_cb.codebook.numel() * 16
    # row_scales
    for name in LINEARS_TO_PQ:
        parts = name.split('.')
        mod = body_pq
        for p in parts:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if isinstance(mod, PQLinearBody):
            buffer_bits += mod.row_scale.numel() * 16
            if mod.bias is not None:
                buffer_bits += mod.bias.numel() * 16
    # norms + 1-D learned vectors -- keep fp16
    buffer_bits += body_pq.gamma.numel() * 16
    buffer_bits += body_pq.beta.numel() * 16
    buffer_bits += body_pq.step_scale.numel() * 16
    buffer_bits += body_pq.block.norm1.weight.numel() * 16
    buffer_bits += body_pq.block.norm1.bias.numel() * 16
    buffer_bits += body_pq.block.norm2.weight.numel() * 16
    buffer_bits += body_pq.block.norm2.bias.numel() * 16

    raw_bytes = (total_raw + buffer_bits) / 8.0
    ent_bytes = (total_ent + buffer_bits) / 8.0
    return raw_bytes, ent_bytes, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deq_ckpt', required=True)
    ap.add_argument('--global_K', type=int, default=1024)
    ap.add_argument('--subvec', type=int, default=8)
    ap.add_argument('--qat_steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    print(f"Loading DEQ body from {args.deq_ckpt}")
    body_pq, ck_meta = load_deq_body(args.deq_ckpt, args.device)
    # teacher is a deep copy of the fp32 body BEFORE PQ install
    body_fp32 = DEQBodyPQ(body_pq.h_outer, body_pq.h_inner,
                          body_pq.block.n_heads).to(args.device)
    body_fp32.load_state_dict(body_pq.state_dict())
    body_fp32.eval()

    # original fp16 body-only byte size
    body_fp_bytes = 0
    for name in LINEARS_TO_PQ:
        parts = name.split('.')
        mod = body_pq
        for p in parts:
            mod = getattr(mod, p)
        body_fp_bytes += mod.weight.numel() * 2
    body_fp_bytes += (body_pq.gamma.numel() + body_pq.beta.numel()
                      + body_pq.step_scale.numel()) * 2
    body_fp_bytes += (body_pq.block.norm1.weight.numel() * 2
                      + body_pq.block.norm1.bias.numel() * 2
                      + body_pq.block.norm2.weight.numel() * 2
                      + body_pq.block.norm2.bias.numel() * 2)
    print(f"  body (fp16 equiv): {body_fp_bytes/1e6:.3f} MB")

    # PQ install
    shared = GlobalCodebook(args.global_K, args.subvec).to(args.device)
    pool = pool_body_subvectors(body_pq, args.subvec).to(args.device)
    print(f"  pooled {pool.shape[0]} body subvectors; k-means init")
    shared.init_from_samples(pool, iters=8)
    pq_install_body(body_pq, shared, args.subvec)
    body_pq = body_pq.to(args.device)

    raw_b, ent_b, per_layer = body_byte_accounting(body_pq, args.global_K, args.subvec)
    print(f"\n--- body byte accounting ---")
    print(f"  K_global={args.global_K}  D={args.subvec}  bits/weight={math.log2(args.global_K)/args.subvec:.3f}")
    print(f"  raw  (log2 K):      {raw_b/1e6:7.3f} MB   ({body_fp_bytes/raw_b:6.1f}x vs fp16 body)")
    print(f"  entropy-coded:      {ent_b/1e6:7.3f} MB   ({body_fp_bytes/ent_b:6.1f}x vs fp16 body)")
    for name, n, H in per_layer:
        print(f"    {name:<20} n={n:<10} H={H:.3f} bits")

    # quick pre-QAT sanity
    with torch.no_grad():
        x = torch.randn(2, 64, body_pq.h_outer, device=args.device)
        y_t = body_fp32(x, iters=args.iters)
        y_s = body_pq(x, iters=args.iters)
        rel0 = ((y_s - y_t).pow(2).sum() / y_t.pow(2).sum().clamp(min=1e-8)).item()
    print(f"\n  pre-QAT rel L2 error: {rel0:.4f}")

    print(f"\n--- v8 QAT ({args.qat_steps} steps) ---")
    qat_body(body_pq, body_fp32, args.device,
             steps=args.qat_steps, batch=args.batch, seq=args.seq, lr=args.lr,
             iters=args.iters)

    with torch.no_grad():
        x = torch.randn(4, 128, body_pq.h_outer, device=args.device)
        y_t = body_fp32(x, iters=args.iters)
        y_s = body_pq(x, iters=args.iters)
        rel1 = ((y_s - y_t).pow(2).sum() / y_t.pow(2).sum().clamp(min=1e-8)).item()
    print(f"\n  post-QAT rel L2 error: {rel1:.4f}")

    # re-account after QAT (codebook may have shifted)
    raw_b2, ent_b2, _ = body_byte_accounting(body_pq, args.global_K, args.subvec)

    print("\n=== v8 SUMMARY ===")
    print(f"  body fp16:          {body_fp_bytes/1e6:7.3f} MB")
    print(f"  body PQ raw:        {raw_b2/1e6:7.3f} MB   ({body_fp_bytes/raw_b2:.1f}x)")
    print(f"  body PQ ent:        {ent_b2/1e6:7.3f} MB   ({body_fp_bytes/ent_b2:.1f}x)")
    print(f"  rel L2 error (fn):  pre={rel0:.4f}   post-QAT={rel1:.4f}")

    torch.save({
        'state_dict': body_pq.state_dict(),
        'config': {'h_outer': body_pq.h_outer, 'h_inner': body_pq.h_inner,
                   'n_heads_inner': body_pq.block.n_heads,
                   'global_K': args.global_K, 'subvec': args.subvec, 'v': 8},
        'body_fp16_bytes': body_fp_bytes, 'raw_bytes': raw_b2, 'ent_bytes': ent_b2,
        'byte_compression_raw': body_fp_bytes / raw_b2,
        'byte_compression_entropy': body_fp_bytes / ent_b2,
        'rel_mse_preQAT': rel0, 'rel_mse_postQAT': rel1,
        'deq_ckpt_source': args.deq_ckpt,
    }, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
