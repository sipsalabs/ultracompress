"""
cache_teacher_8b.py — download Qwen3-8B and cache the state_dict to a local
.pt file so subsequent scripts can reload without re-downloading. Also
verifies the model loads cleanly at fp16 on cuda:0.
"""
from __future__ import annotations
import argparse, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="qwen3_8b_cache.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    print(f"[cache] downloading {args.model_id} (fp16) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, trust_remote_code=True)
    print(f"[cache] loaded in {time.time()-t0:.0f}s; moving state_dict to cpu")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    n_params = sum(v.numel() for v in sd.values())
    print(f"[cache] {len(sd)} tensors, {n_params/1e9:.3f} B params")

    torch.save(sd, args.out)
    print(f"[cache] saved {args.out}")


if __name__ == "__main__":
    main()
