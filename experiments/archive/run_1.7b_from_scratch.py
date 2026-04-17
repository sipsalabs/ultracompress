"""
1.7B FRR FROM SCRATCH — No teacher. No ceiling. Just learn.

0.6B from scratch at 30K steps got HellaSwag 25.5% (78.5% of teacher).
1.7B should be significantly better — more capacity, same shared block.
100K steps on real FineWeb-Edu text.

No teacher = no ceiling. The model learns language directly.
If it matches the teacher's HellaSwag (32.5%), that proves FRR
is a viable TRAINING architecture, not just compression.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 100000

print("=" * 60)
print("1.7B FRR FROM SCRATCH — No teacher. No ceiling.")
print("=" * 60)

# Load ONLY embeddings from 1.7B (no teacher model needed)
print("Loading 1.7B embeddings only (no teacher)...")
wd = torch.load('qwen3_1.7b_cache.pt', weights_only=True)
embed_w = wd['model.embed_tokens.weight'].float().to(device)
hidden = embed_w.shape[1]  # 2048
norm_w = wd.get('model.norm.weight', torch.ones(hidden)).float().to(device)
lm_head_w = wd.get('lm_head.weight', embed_w).to(device)
vocab_size = embed_w.shape[0]
del wd  # Free the big dict — we only need embeddings
torch.cuda.empty_cache()
print(f"  Hidden: {hidden}, Vocab: {vocab_size}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_batch(batch_size=4, seq_len=128):
    global ds_iter
    tokens_list = []
    for _ in range(batch_size):
        while True:
            try:
                sample = next(ds_iter)
                text = sample.get('text', '')
                if len(text) < 200: continue
                toks = tokenizer.encode(text, max_length=seq_len + 1, truncation=True, return_tensors='pt')[0]
                if len(toks) >= seq_len + 1:
                    tokens_list.append(toks[:seq_len + 1])
                    break
            except StopIteration:
                ds_iter = iter(ds)
    return torch.stack(tokens_list).to(device)

# Build FRR — unfreeze block for from-scratch training
model = FractalModel(hidden, 16, 4, 7, vocab_size, 1,
                     embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
for p in model.parameters():
    p.requires_grad = True
# Keep embeddings frozen (they're pretrained vocabulary, not the model)
model.embed.weight.requires_grad = False
if hasattr(model, 'lm_head') and model.lm_head is not None:
    model.lm_head.weight.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FRR from scratch: {trainable:,} trainable params")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()

print(f"Training {STEPS} steps on FineWeb-Edu (NO teacher)...")
for step in range(STEPS):
    batch = get_batch(4, 128)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    if step % 10000 == 0:
        ppl = math.exp(min(loss.item(), 20))
        elapsed = time.time() - t0
        print(f"  Step {step}/{STEPS}: loss={loss.item():.4f} ppl={ppl:.1f} ({elapsed:.0f}s)")

ppl = math.exp(min(loss.item(), 20))
print(f"\nFINAL: PPL={ppl:.1f}")
print(f"Trainable: {trainable:,}")
print("Done!")
