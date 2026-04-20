"""tokenize_wikitext.py — tokenize WikiText-103 test split with an arbitrary HF tokenizer.

Produces a flat torch tensor of int32 token ids, matching the format used by
all v17 eval scripts. Needed for cross-family Claim-16 validation where the
target model's tokenizer differs from the Qwen3 default.
"""
from __future__ import annotations
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True,
                    help="HF tokenizer id, e.g. mistralai/Mistral-7B-v0.3")
    ap.add_argument("--out", required=True,
                    help="Output .pt path, e.g. wikitext103_test_mistral.pt")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[tok] loading tokenizer {args.model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print("[tok] loading wikitext-103-raw-v1 test split", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join([row["text"] for row in ds if row["text"].strip()])
    print(f"[tok] tokenizing ({len(text)/1e6:.1f}M chars)", flush=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    all_tokens = ids.to(torch.int32)
    torch.save(all_tokens, args.out)
    print(f"[tok] saved {all_tokens.numel()/1e3:.1f}K tokens -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
