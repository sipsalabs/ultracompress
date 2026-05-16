"""Run a Sipsa-compressed model locally — no API needed.

For anyone evaluating Sipsa. Hardware: any machine with
either CUDA, MPS (M-series Mac), or even CPU (slow but works for tiny models).

Walks the customer through:
  1. pip install ultracompress
  2. hf download SipsaLabs/qwen3-0.6b-uc-v3-bpw5
  3. wrap base HF skeleton with CorrectionLinear
  4. load per-layer reconstructed state dicts
  5. run real generate() — no Sipsa servers involved

This is the substrate proof: customers can audit the bit-identical
reconstruction themselves and run inference on their own hardware.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ---- step 0: pip install ultracompress (one time)
# pip install ultracompress transformers torch huggingface_hub

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from ultracompress.api_v3 import DEFAULT_TARGETS, _wrap_with_correction
from ultracompress.pack_v3 import (
    parse_uc_layer_v3,
    reconstruct_layer_state_dict_v3,
)


# Pick whichever model you want — all 6 of these are real Sipsa-compressed
# packs published on huggingface.co/SipsaLabs.
#
# Qwen3 family (sub-1% PPL ratio):
#   ("Qwen/Qwen3-0.6B",          "SipsaLabs/qwen3-0.6b-uc-v3-bpw5",          1.0069)
#   ("Qwen/Qwen3-1.7B-Base",     "SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5",     1.0040)  # all-time record
#   ("Qwen/Qwen3-1.7B",          "SipsaLabs/qwen3-1.7b-uc-v3-bpw5",          ~1.005)
#   ("Qwen/Qwen3-8B",            "SipsaLabs/qwen3-8b-uc-v3-bpw5",            1.0044)
#   ("Qwen/Qwen3-14B",           "SipsaLabs/qwen3-14b-uc-v3-bpw5",           1.0040)
#
# Other architectures (PPL ratio >1%, transparently published — these are
# arch-floor honest results, not failures):
#   ("NousResearch/Meta-Llama-3.1-8B",  "SipsaLabs/llama-3.1-8b-uc-v3-bpw5",  1.0125)
#   ("mistralai/Mistral-7B-v0.3",       "SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5", 1.00548)

BASE_HF_ID = "Qwen/Qwen3-0.6B"
SIPSA_HF_REPO = "SipsaLabs/qwen3-0.6b-uc-v3-bpw5"
PROMPT = "In one paragraph, explain why model compression matters for AI deployment."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    print("[warn] no GPU detected — falling back to CPU. Use a small model "
          "(0.6B or 1.7B) and expect slow generation.")
    return "cpu"


def main() -> int:
    device = pick_device()
    print(f"=== Sipsa local-inference demo — device={device}")
    print(f"=== model = {BASE_HF_ID} (Sipsa-compressed at {SIPSA_HF_REPO})")

    t0 = time.time()
    print("\n[1/5] downloading the Sipsa-compressed pack from HuggingFace...")
    pack_dir = Path(snapshot_download(SIPSA_HF_REPO))
    print(f"      pack at {pack_dir}")

    print(f"\n[2/5] loading the base bf16 skeleton ({BASE_HF_ID})...")
    base = AutoModelForCausalLM.from_pretrained(BASE_HF_ID, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE_HF_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"      base loaded — {sum(p.numel() for p in base.parameters())/1e9:.2f}B params")

    print("\n[3/5] detecting correction rank from layer_000.uc...")
    parsed_l0 = parse_uc_layer_v3(pack_dir / "layer_000.uc")
    ranks = sorted({int(p["rank"]) for k, p in parsed_l0.items()
                    if isinstance(p, dict) and "rank" in p})
    rank = ranks[0] if ranks else 32
    print(f"      rank={rank}")

    print(f"\n[4/5] wrapping target Linears + loading per-layer state dicts...")
    n_replaced = _wrap_with_correction(base, rank=rank, targets=DEFAULT_TARGETS)
    n_layers = base.config.num_hidden_layers
    for i in range(n_layers):
        sd = reconstruct_layer_state_dict_v3(pack_dir / f"layer_{i:03d}.uc")
        base.model.layers[i].load_state_dict(sd, strict=False)
    print(f"      replaced {n_replaced} Linears across {n_layers} layers")
    base = base.to(device)
    base.train(False)

    print(f"\n[5/5] running real inference (Sipsa-compressed substrate)...")
    print(f"      prompt: {PROMPT!r}")
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    t_gen = time.time()
    with torch.no_grad():
        out = base.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    gen_dt = time.time() - t_gen
    n_gen = int(out.shape[1] - inputs["input_ids"].shape[1])
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print()
    print("=" * 60)
    print(f"prompt_tokens:     {inputs['input_ids'].shape[1]}")
    print(f"completion_tokens: {n_gen}")
    print(f"speed:             {n_gen/gen_dt:.1f} tok/s on {device}")
    print(f"total_elapsed:     {time.time()-t0:.1f}s")
    print(f"\ntext:\n{text}")
    print("=" * 60)
    print("\nThat output came from a Sipsa-compressed model — same SHA-256")
    print("verifiable bit-identical reconstruction we serve via the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
