"""
cache_activations.py — Cache per-Linear INPUT activation statistics from
the fp16 teacher on calibration tokens. Used by v17 for activation-aware
codebook refinement (Claim 14).

Protocol:
  - Run the teacher on N calibration windows of WikiText-103.
  - For each body Linear, hook the PRE-linear input X (shape [B, L, I]).
  - Record per-input-dim variance  sigma2[i] = E[X[:,:,i]^2], i=0..I-1.
  - Save {linear_name: sigma2 tensor} to disk.

These sigma2 vectors let v17 weight each input dim (column of W) by its
activation variance, so EM minimizes ||X (W - W_q)^T||_F^2 (the actual
output error) rather than ||W - W_q||_F^2 (reconstruction error).

Usage:
  python cache_activations.py --n_cal 32 --out v17_activations.pt
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.nn as nn

from compress_v14 import ROLE_PATTERNS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n_cal", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="v17_activations.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[act] loading teacher {args.teacher}")
    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    # Running sum of X^2 per dim, for each target Linear
    sum_sq: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "layers." in name and \
           any(p in name for p in ROLE_PATTERNS):
            full = name + ".weight"
            I = mod.in_features
            sum_sq[full] = torch.zeros(I, dtype=torch.float32, device=device)
            count[full] = 0
            def hk(full_key):
                def _h(m, inp, out):
                    x = inp[0]   # [..., I]
                    x = x.reshape(-1, x.shape[-1]).float()
                    sum_sq[full_key].add_(x.pow(2).sum(0))
                    count[full_key] += x.shape[0]
                return _h
            hooks.append(mod.register_forward_hook(hk(full)))
    print(f"[act] hooks attached to {len(sum_sq)} body Linears")

    g = torch.Generator().manual_seed(args.seed)
    total = all_tokens.numel()
    starts = torch.randint(0, total - args.seq_len - 1,
                           (args.n_cal,), generator=g)
    t0 = time.time()
    for i, s in enumerate(starts.tolist()):
        win = all_tokens[s:s + args.seq_len].to(device=device, dtype=torch.long).unsqueeze(0)
        _ = model(win)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{args.n_cal}]  {time.time()-t0:.0f}s")

    for h in hooks:
        h.remove()

    sigma2: dict[str, torch.Tensor] = {}
    for k in sum_sq:
        sigma2[k] = (sum_sq[k] / max(count[k], 1)).cpu()
    torch.save(sigma2, args.out)
    print(f"\nSaved {args.out}  ({len(sigma2)} tensors)")

    # summary stats
    print("\nsample sigma2 spectra (first 5 dims, max, ratio max/mean):")
    for k in list(sigma2.keys())[:6]:
        s = sigma2[k]
        print(f"  {k:<55s}  mean {s.mean():.4f}  max {s.max():.4f}  "
              f"ratio {(s.max()/s.mean()).item():.2f}")


if __name__ == "__main__":
    main()
