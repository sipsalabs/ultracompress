"""
RESONANCE ENGINE TEST — Can selective amplification learn language?

No attention. No FFN. Just resonators that amplify matching patterns.
Like the cochlea decomposes sound through resonance.

If PPL < 100: resonance models language structure.
If PPL < 10: paradigm shift — intelligence through resonance.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.resonance_net import ResonanceEngine
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 20000

print("=" * 60)
print("RESONANCE ENGINE: Can selective amplification learn language?")
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

model = ResonanceEngine(1024, n_resonators=128, n_cross_resonators=64,
                        n_cycles=28, vocab_size=151936,
                        embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Resonance engine: {trainable:,} trainable params")

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
print(f"RESONANCE ENGINE RESULTS")
print(f"{'='*60}")
print(f"  Trainable: {trainable:,}")
print(f"  Final PPL: {ppl:.1f}")
if ppl < 10:
    print(f"  >>> PARADIGM SHIFT: Resonance models language!")
elif ppl < 100:
    print(f"  >>> PROMISING: Resonance captures language structure!")
elif ppl < 500:
    print(f"  >>> LEARNING: Better than wave engine (PPL 444)")
else:
    print(f"  Resonance didn't converge well.")
print("Done!")
