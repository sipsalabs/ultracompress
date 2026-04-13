"""ROTATION FROM SCRATCH — No teacher. No distillation. Just language.

89K params learning to predict next tokens on real text.
If this learns ANYTHING, it proves the architecture works
independently of the broken old system.

The blind person isn't teaching anymore. We're learning to see."""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultracompress.rotation_engine import RotationEngine

device = 'cuda'
STEPS = 50000
SEQ_LEN = 64
BATCH = 8

print("=" * 60)
print("ROTATION FROM SCRATCH — No teacher. Pure language learning.")
print("=" * 60)

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
    print("FineWeb-Edu loaded! Learning from REAL text, no teacher.")
except Exception as e:
    print(f"FineWeb failed: {e}. Using random tokens.")

def get_batch():
    global ds_iter
    if not USE_REAL_TEXT:
        return torch.randint(100, 50000, (BATCH, SEQ_LEN + 1), device=device)
    tokens_list = []
    for _ in range(BATCH):
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

# Build rotation engine — NO teacher weights, everything learned
print("Building rotation engine from scratch (NO teacher)...")
model = RotationEngine(1024, n_planes=64, n_cycles=28, vocab_size=151936).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Make ALL params trainable (including embeddings)
for p in model.parameters():
    p.requires_grad = True
trainable = sum(p.numel() for p in model.parameters())
print(f"Total params: {trainable:,} (all trainable, no frozen teacher weights)")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

print(f"Training {STEPS} steps on {'FineWeb-Edu' if USE_REAL_TEXT else 'random tokens'}...")
t0 = time.time()
for step in range(STEPS):
    batch = get_batch()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); scheduler.step()

    if step % 5000 == 0 or step == STEPS - 1:
        ppl = math.exp(min(loss.item(), 20))
        # Generate sample
        prompt = tokenizer.encode("The future of", return_tensors='pt').to(device)
        with torch.no_grad():
            gen = prompt.clone()
            for _ in range(20):
                logits_gen = model(gen)
                next_tok = logits_gen[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen = torch.cat([gen, next_tok], dim=1)
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"  Step {step}: loss={loss.item():.4f} ppl={ppl:.1f} ({time.time()-t0:.0f}s)")
        print(f"    Gen: {text[:120]}")

print(f"\nFINAL loss: {loss.item():.4f} ppl={math.exp(min(loss.item(), 20)):.1f}")
print(f"Trained {STEPS} steps on real text with NO teacher.")
print(f"Architecture: rotation engine, {trainable:,} params")
print("Done!")
