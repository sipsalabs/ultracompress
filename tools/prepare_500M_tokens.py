"""
Download and tokenize more FineWeb-Edu data for training.
Produces fineweb_edu_500M_tokens.pt (500M tokens, ~1.9GB).

This gives 5x more data diversity for distillation training.
Uses Qwen3 tokenizer to match teacher vocabulary.
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

TARGET_TOKENS = 500_000_000
OUTPUT_FILE = 'fineweb_edu_500M_tokens.pt'

if os.path.exists(OUTPUT_FILE):
    t = torch.load(OUTPUT_FILE, weights_only=True)
    print(f"Already exists: {OUTPUT_FILE}  shape={t.shape}  numel={t.numel()/1e6:.0f}M")
    sys.exit(0)

print(f"Target: {TARGET_TOKENS/1e6:.0f}M tokens → {OUTPUT_FILE}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# Stream FineWeb-Edu (no download of full dataset)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)

all_ids = []
total = 0
t0 = time.time()

for i, example in enumerate(ds):
    text = example.get('text', '')
    if len(text) < 50:
        continue
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_ids.extend(ids)
    total = len(all_ids)

    if i % 10000 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = total / elapsed
        eta = (TARGET_TOKENS - total) / rate if rate > 0 else 0
        print(f"  docs={i:,}  tokens={total/1e6:.1f}M / {TARGET_TOKENS/1e6:.0f}M  "
              f"rate={rate/1e6:.2f}M tok/s  ETA={eta/60:.1f}min", flush=True)

    if total >= TARGET_TOKENS:
        break

# Truncate to exact target
all_ids = all_ids[:TARGET_TOKENS]
tensor = torch.tensor(all_ids, dtype=torch.int32)
torch.save(tensor, OUTPUT_FILE)
elapsed = time.time() - t0
print(f"\nDone: {tensor.numel()/1e6:.0f}M tokens saved to {OUTPUT_FILE}")
print(f"  file size: {os.path.getsize(OUTPUT_FILE)/1e9:.2f} GB")
print(f"  time: {elapsed/60:.1f} min")
