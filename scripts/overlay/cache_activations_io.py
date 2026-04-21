"""
cache_activations_io.py — Cache BOTH input AND output activation variances
for each body Linear. Input variances reproduce v17's s_col (Claim 14);
output variances enable Claim 15's row saliency u[o] = E[y_o^2]^(beta/2).

Saves:
  {
    "sigma2_in":  {linear_name: tensor[I]},   # E[x_i^2]
    "sigma2_out": {linear_name: tensor[O]},   # E[y_o^2]
  }
"""
from __future__ import annotations
import argparse, time
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
    ap.add_argument("--out", default="v18_activations_io.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[act] loading teacher")
    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    sum_sq_in: dict[str, torch.Tensor] = {}
    sum_sq_out: dict[str, torch.Tensor] = {}
    cnt_in: dict[str, int] = {}
    cnt_out: dict[str, int] = {}

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "layers." in name and \
           any(p in name for p in ROLE_PATTERNS):
            full = name + ".weight"
            sum_sq_in[full] = torch.zeros(mod.in_features,
                                          dtype=torch.float32, device=device)
            sum_sq_out[full] = torch.zeros(mod.out_features,
                                           dtype=torch.float32, device=device)
            cnt_in[full] = 0
            cnt_out[full] = 0
            def hk(full_key):
                def _h(m, inp, out):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    y = out.reshape(-1, out.shape[-1]).float()
                    sum_sq_in[full_key].add_(x.pow(2).sum(0))
                    sum_sq_out[full_key].add_(y.pow(2).sum(0))
                    cnt_in[full_key] += x.shape[0]
                    cnt_out[full_key] += y.shape[0]
                return _h
            hooks.append(mod.register_forward_hook(hk(full)))
    print(f"[act] hooks on {len(sum_sq_in)} Linears (input+output)")

    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n_cal,), generator=g)
    t0 = time.time()
    for i, s in enumerate(starts.tolist()):
        win = all_tokens[s:s+args.seq_len].to(device=device, dtype=torch.long).unsqueeze(0)
        _ = model(win)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{args.n_cal}]  {time.time()-t0:.0f}s")
    for h in hooks: h.remove()

    sigma2_in = {k: (sum_sq_in[k] / max(cnt_in[k], 1)).cpu() for k in sum_sq_in}
    sigma2_out = {k: (sum_sq_out[k] / max(cnt_out[k], 1)).cpu() for k in sum_sq_out}
    torch.save({"sigma2_in": sigma2_in, "sigma2_out": sigma2_out}, args.out)
    print(f"\nSaved {args.out}")

    print("\noutput-side sigma2 spectra (first 6 Linears):")
    for k in list(sigma2_out.keys())[:6]:
        s = sigma2_out[k]
        print(f"  {k:<55s}  mean {s.mean():.4f}  max {s.max():.4f}  "
              f"ratio {(s.max()/s.mean()).item():.2f}")


if __name__ == "__main__":
    main()
