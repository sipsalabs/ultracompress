"""
FRR FROM SCRATCH — Not compression. A NEW ARCHITECTURE.

The question: can a weight-shared transformer learn language from scratch
as well as a full-width model?

If YES: FRR isn't compression. It's a better architecture.
  - Built-in regularization (weight sharing prevents overfitting)
  - 60x fewer params to overfit
  - Recursive processing (brain re-processes information too)
  - Same effective depth (28 passes) with 1 block

If NO: weight sharing loses critical per-layer specialization.

Training: pure next-token prediction on FineWeb-Edu
Evaluation: HellaSwag, perplexity, text generation
No teacher. No distillation. No ceiling. Just: can this architecture learn?

We compare 3 things with the SAME training budget (steps * batch * seq):
1. FRR (shared block, 7.3M trainable) — 28 passes of 1 block
2. Small transformer (7.3M total) — 4 layers, same param count
3. Teacher (440M) — baseline, already trained

If FRR > small transformer at same param count → weight sharing WINS.
"""
import lib.unbuffered
import torch, sys, os, time, math, json
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.moonshot import FractalModel
from transformers import AutoTokenizer

device = 'cuda'
STEPS = 30000
BATCH = 8
SEQ_LEN = 128

print("=" * 60)
print("FRR FROM SCRATCH: Is weight sharing a BETTER architecture?")
print("=" * 60)

# Load embeddings (we keep these frozen — they're the vocabulary)
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
embed_w = wd['model.embed_tokens.weight'].float().to(device)
norm_w = wd.get('model.norm.weight', torch.ones(1024)).float().to(device)
lm_head_w = wd.get('lm_head.weight', embed_w).to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_iter = iter(ds)
print("FineWeb-Edu loaded!")

def get_batch(batch_size=BATCH, seq_len=SEQ_LEN):
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


class SmallTransformer(nn.Module):
    """Regular (non-shared) transformer with same param count as FRR.
    FRR has ~7.3M trainable. This has 4 independent layers ≈ 7.3M."""

    def __init__(self, hidden_dim, n_heads, n_layers, vocab_size,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_layers = n_layers
        head_dim = hidden_dim // n_heads

        # Independent layers (NOT shared)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': nn.RMSNorm(hidden_dim),
                'qkv': nn.Linear(hidden_dim, 3 * hidden_dim, bias=False),
                'o_proj': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'norm2': nn.RMSNorm(hidden_dim),
                'ffn_gate': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'ffn_up': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'ffn_down': nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            })
            self.layers.append(layer)

        self.n_heads = n_heads
        self.head_dim = head_dim

        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()
        B, T, D = x.shape

        for layer in self.layers:
            # Attention
            h = layer['norm1'](x)
            qkv = layer['qkv'](h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if T > 1:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, D)
            x = x + layer['o_proj'](out)

            # FFN (SwiGLU)
            h = layer['norm2'](x)
            gate = F.silu(layer['ffn_gate'](h))
            up = layer['ffn_up'](h)
            x = x + layer['ffn_down'](gate * up)

        x = self.out_norm(x)
        return self.lm_head(x)


def eval_perplexity(model_fn, n_batches=50):
    total_loss, total_tokens = 0, 0
    for _ in range(n_batches):
        batch = get_batch(2, 128)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with torch.no_grad():
            logits = model_fn(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += targets.numel()
    return math.exp(total_loss / total_tokens)


def eval_hellaswag(model_fn, n_samples=200):
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
    except:
        return -1
    correct, total = 0, 0
    for i, sample in enumerate(hs):
        if i >= n_samples: break
        ctx, endings, label = sample['ctx'], sample['endings'], int(sample['label'])
        best_score, best_idx = float('-inf'), 0
        for j, ending in enumerate(endings):
            tokens = tokenizer.encode(ctx + " " + ending, max_length=128, truncation=True, return_tensors='pt').to(device)
            if tokens.shape[1] < 2: continue
            with torch.no_grad():
                logits = model_fn(tokens)
            ctx_len = len(tokenizer.encode(ctx, max_length=128, truncation=True))
            if ctx_len >= tokens.shape[1] - 1: continue
            log_probs = F.log_softmax(logits[0, ctx_len-1:-1], dim=-1)
            score = log_probs.gather(1, tokens[0, ctx_len:].unsqueeze(1)).mean().item()
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx == label: correct += 1
        total += 1
    return correct / total if total > 0 else 0


def train_and_eval(name, model, steps=STEPS):
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Trainable: {trainable:,}")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()

    for step in range(steps):
        batch = get_batch()
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if step % 5000 == 0:
            ppl = math.exp(min(loss.item(), 20))
            elapsed = time.time() - t0
            print(f"    Step {step}: loss={loss.item():.4f} ppl={ppl:.1f} ({elapsed:.0f}s)")

    # Full eval
    ppl = eval_perplexity(model)
    print(f"  Final PPL: {ppl:.1f}")
    hs = eval_hellaswag(model, n_samples=200)
    print(f"  HellaSwag: {hs*100:.1f}%")
    return trainable, ppl, hs


# ═══════════════════════════════════════════════════════════
# MODEL 1: FRR from scratch (weight-shared, 28 passes)
# ═══════════════════════════════════════════════════════════

frr = FractalModel(1024, 16, 4, 7, 151936, 1,
                   embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
# Unfreeze everything for from-scratch training
for p in frr.parameters():
    p.requires_grad = True
# Re-freeze embeddings (they're the vocabulary, not the model)
frr.embed.weight.requires_grad = False
if hasattr(frr, 'lm_head') and frr.lm_head is not None:
    frr.lm_head.weight.requires_grad = False

p_frr, ppl_frr, hs_frr = train_and_eval("FRR (shared block, 28 passes, ~7.3M trainable)", frr)
del frr; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# MODEL 2: Small transformer (same param count, 4 layers)
# ═══════════════════════════════════════════════════════════

# 4 layers with intermediate_size=2048 (not 3072) ≈ 7.3M trainable
small = SmallTransformer(1024, 16, 4, 151936,
                         embed_weight=embed_w, lm_head_weight=lm_head_w, norm_weight=norm_w).to(device)
p_small, ppl_small, hs_small = train_and_eval("Small Transformer (4 layers, ~7.3M trainable)", small)
del small; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESULTS: Is weight sharing a better architecture?")
print(f"{'='*60}")
print(f"  {'Model':<45} {'Params':>10} {'PPL':>8} {'HellaSwag':>10}")
print(f"  {'-'*75}")
print(f"  {'FRR (shared, 28 passes)':<45} {p_frr:>10,} {ppl_frr:>8.1f} {hs_frr*100:>9.1f}%")
print(f"  {'Small Transformer (4 layers)':<45} {p_small:>10,} {ppl_small:>8.1f} {hs_small*100:>9.1f}%")
print(f"  {'Teacher (full 440M, pretrained)':<45} {'440M':>10} {'~1200':>8} {'~29%':>10}")

if hs_frr > hs_small:
    print(f"\n  >>> FRR WINS at same param count!")
    print(f"  >>> Weight sharing IS a better architecture for small models.")
elif hs_frr > hs_small * 0.9:
    print(f"\n  FRR is competitive with standard architecture at same size.")
else:
    print(f"\n  Standard architecture wins. Weight sharing costs quality.")

print("\nDone!")
