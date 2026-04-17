"""
FRR FROM-SCRATCH V2: Train FRR as a language model from scratch.
NOT distillation — proper next-token prediction on real text.

This tests: can FRR be a competitive LLM architecture on its own?
If yes, it's not just compression — it's a new efficient architecture.

Uses FineWeb-Edu for real text training.
Architecture: shared block + per-scale modulation, no teacher.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.moonshot import FractalModel

device = 'cuda'
STEPS = 30000
SEQ_LEN = 128
BATCH_SIZE = 4

print("=" * 70)
print("FRR FROM-SCRATCH: Training as a real language model")
print("=" * 70)

# Load tokenizer + dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

USE_REAL_TEXT = False
ds_iter = None
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds_iter = iter(ds)
    USE_REAL_TEXT = True
    print("FineWeb-Edu loaded!")
except Exception as e:
    print(f"FineWeb failed: {e}. Using random tokens.")


def get_batch():
    global ds_iter
    if not USE_REAL_TEXT:
        return torch.randint(100, 50000, (BATCH_SIZE, SEQ_LEN + 1), device=device)
    tokens_list = []
    for _ in range(BATCH_SIZE):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                toks = tokenizer.encode(text, max_length=SEQ_LEN + 1,
                                       truncation=True, return_tensors='pt')[0]
                if len(toks) >= SEQ_LEN + 1:
                    tokens_list.append(toks[:SEQ_LEN + 1])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)


# Build FRR — no teacher embeddings, learn everything from scratch
print("Building FRR from scratch...")
model = FractalModel(
    hidden_dim=1024, n_heads=16, n_scales=4, iters_per_scale=7,
    vocab_size=151936, ff_mult=1,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable: {trainable:,}")
print(f"For comparison: Qwen3-0.6B has 751M params")
print(f"FRR from scratch: {total_params/1e6:.0f}M params ({751/total_params*1e6:.0f}x smaller)")

# All params trainable (including embed/head)
for p in model.parameters():
    p.requires_grad = True

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

print(f"\nTraining {STEPS} steps on {'FineWeb-Edu' if USE_REAL_TEXT else 'random tokens'}...")
t0 = time.time()
for step in range(STEPS):
    batch = get_batch()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()

    if step % 5000 == 0 or step == STEPS - 1:
        perplexity = math.exp(min(loss.item(), 20))
        elapsed = time.time() - t0

        # Quick generation test
        prompt = tokenizer.encode("The future of AI is", return_tensors='pt').to(device)
        with torch.no_grad():
            gen = prompt.clone()
            for _ in range(30):
                logits_gen = model(gen)
                next_tok = logits_gen[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen = torch.cat([gen, next_tok], dim=1)
            gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)

        print(f"  Step {step}: loss={loss.item():.4f} ppl={perplexity:.1f} ({elapsed:.0f}s)")
        print(f"    Gen: {gen_text[:150]}")

print(f"\nFINAL loss: {loss.item():.4f}")
print(f"Done! Total time: {time.time()-t0:.0f}s")
