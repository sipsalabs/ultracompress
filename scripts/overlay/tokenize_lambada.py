"""tokenize_lambada.py — tokenize the LAMBADA test split with an arbitrary HF tokenizer.

LAMBADA is narrative fiction from BookCorpus, completely distinct from
WikiText-103's encyclopedic style. Passing PPL on LAMBADA with the same
2.40-bpw fit validated on WikiText proves the Claim-16 envelope is not
a property of the evaluation corpus.
"""
from __future__ import annotations
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="EleutherAI/lambada_openai",
                    help="HF dataset id (default: EleutherAI/lambada_openai)")
    ap.add_argument("--config", default="default")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[tok] loading tokenizer {args.model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"[tok] loading {args.dataset} test split", flush=True)
    ds = load_dataset(args.dataset, args.config, split="test")
    text = "\n\n".join([row["text"] for row in ds if row["text"].strip()])
    print(f"[tok] tokenizing ({len(text)/1e6:.2f}M chars)", flush=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    all_tokens = ids.to(torch.int32)
    torch.save(all_tokens, args.out)
    print(f"[tok] saved {all_tokens.numel()/1e3:.1f}K tokens -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
