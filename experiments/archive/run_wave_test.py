"""
WAVE ENGINE TEST — Can waves learn language?

The most radical test: no attention, no FFN, no matrix multiply.
Just wave propagation and interference.

If this learns ANYTHING: computation through waves is viable.
If PPL drops below 100: waves can model language.
If PPL drops below 10: paradigm shift confirmed.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.wave_engine import WaveEngine
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 20000

print("=" * 60)
print("WAVE ENGINE: Can interference learn language?")
print("=" * 60)

wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
embed_w = wd['model.embed_tokens.weight'].float().to(device)
norm_w = wd.get('model.norm.weight', torch.ones(1024)).float().to(device)
lm_head_w = wd.get('lm_head.weight', embed_w).to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_batch(batch_size=8, seq_len=128):
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

# Build wave engine
model = WaveEngine(1024, n_freqs=128, n_steps=28, vocab_size=151936,
                   embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

dna = model.dna_size()
print(f"Wave DNA: {dna}")
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable: {total_trainable:,}")

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()

for step in range(STEPS):
    batch = get_batch(8, 128)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    if step % 3000 == 0:
        ppl = math.exp(min(loss.item(), 20))
        elapsed = time.time() - t0
        print(f"  Step {step}: loss={loss.item():.4f} ppl={ppl:.1f} ({elapsed:.0f}s)")

ppl = math.exp(min(loss.item(), 20))
print(f"\n{'='*60}")
print(f"WAVE ENGINE RESULTS")
print(f"{'='*60}")
print(f"  Final PPL: {ppl:.1f}")
print(f"  DNA size: {dna}")
print(f"  Total trainable: {total_trainable:,}")
if ppl < 10:
    print(f"  >>> PARADIGM SHIFT: Waves can model language!")
elif ppl < 100:
    print(f"  >>> PROMISING: Waves show language structure!")
else:
    print(f"  Waves didn't learn language structure.")
print("Done!")
