"""
Phase 1: Train FRR (Fractal Residual Recursion) from scratch as a language model.

NOT distillation. This trains the FractalModel architecture directly on
next-token prediction to prove it can learn language patterns.

Architecture: FractalModel with GatedRecurrence + LoRA adapters
Data: Synthetic repeating patterns (ABC, counting, copy sequences)
Goal: Prove FRR can learn — if loss drops and generation works, the
      architecture is viable for 100T training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultracompress.moonshot import FractalModel, GatedRecurrence


# ================================================================
# FRR model with GatedRecurrence integrated
# ================================================================

class FRRFromScratch(nn.Module):
    """FractalModel enhanced with GatedRecurrence for stable deep recursion.

    Wraps the core FractalBlock with gated updates instead of the simple
    interpolation in the base FractalModel. This is critical for training
    from scratch where gradients must flow cleanly through 16+ virtual layers.
    """

    def __init__(self, hidden_dim=512, n_heads=8, n_scales=4, iters_per_scale=4,
                 vocab_size=32000, ff_mult=2, lora_rank=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # Import the shared block directly
        from ultracompress.moonshot import FractalBlock
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # Per-scale modulation
        self.scale_gamma = nn.Parameter(torch.ones(n_scales, hidden_dim))
        self.scale_beta = nn.Parameter(torch.zeros(n_scales, hidden_dim))

        # Gated recurrence for each virtual layer (the key ingredient)
        self.gates = nn.ModuleList([
            GatedRecurrence(hidden_dim, init_bias=-2.0)
            for _ in range(self.total_layers)
        ])

        # Per-layer LoRA adapters for specialization
        from ultracompress.moonshot import LoRAAdapter
        self.adapters = nn.ModuleList([
            LoRAAdapter(hidden_dim, rank=lora_rank)
            for _ in range(self.total_layers)
        ])

        # Embedding + LM head (learned from scratch)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

        # Tie embedding weights to LM head
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Sensible init for training from scratch."""
        nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        x = self.embed(tokens)
        layer_idx = 0

        for scale in range(self.n_scales):
            gamma = self.scale_gamma[scale]
            beta = self.scale_beta[scale]

            for it in range(self.iters_per_scale):
                h_old = x
                h_new = self.block(x, gamma, beta)
                # Gated recurrence instead of simple interpolation
                x = self.gates[layer_idx](h_new, h_old)
                # LoRA adapter for per-layer specialization
                x = self.adapters[layer_idx](x)
                layer_idx += 1

        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        block_p = sum(p.numel() for p in self.block.parameters())
        gate_p = sum(p.numel() for p in self.gates.parameters())
        adapter_p = sum(p.numel() for p in self.adapters.parameters())
        embed_p = self.embed.weight.numel()
        return {
            'total': total,
            'block (shared)': block_p,
            'gates': gate_p,
            'adapters': adapter_p,
            'embed+head (tied)': embed_p,
            'scale_mod': self.scale_gamma.numel() + self.scale_beta.numel(),
        }


# ================================================================
# Synthetic data generators
# ================================================================

class SyntheticDataset:
    """Generates synthetic pattern data for training.

    Three pattern types mixed together:
    1. ABC repetition: repeating token sequences
    2. Counting: 1 2 3 4 5 1 2 3 4 5 ...
    3. Copy: "the cat sat the cat sat ..."
    """

    def __init__(self, vocab_size=32000, seq_len=64, n_patterns=50):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.patterns = []

        # Reserve tokens 0-99 for special patterns, rest is noise padding
        # Pattern 1: ABC-style repetition (various cycle lengths)
        for cycle_len in range(2, 12):
            # Use tokens 10..10+cycle_len as the repeating unit
            base = list(range(10, 10 + cycle_len))
            pattern = (base * ((seq_len + 1) // cycle_len + 1))[:seq_len + 1]
            self.patterns.append(('repeat', torch.tensor(pattern, dtype=torch.long)))

        # Pattern 2: Counting sequences (various ranges)
        for count_max in range(3, 15):
            base = list(range(50, 50 + count_max))
            pattern = (base * ((seq_len + 1) // count_max + 1))[:seq_len + 1]
            self.patterns.append(('count', torch.tensor(pattern, dtype=torch.long)))

        # Pattern 3: Copy sequences (phrase repetition)
        phrase_lens = [2, 3, 4, 5, 6, 8, 10]
        for plen in phrase_lens:
            # Use tokens 80..80+plen as a "phrase"
            phrase = list(range(80, 80 + plen))
            pattern = (phrase * ((seq_len + 1) // plen + 1))[:seq_len + 1]
            self.patterns.append(('copy', torch.tensor(pattern, dtype=torch.long)))

        # Pattern 4: Alternating patterns (ABABAB, ABCABC with different offsets)
        for offset in range(5):
            base_tok = 100 + offset * 10
            for cycle in [2, 3, 4]:
                unit = list(range(base_tok, base_tok + cycle))
                pattern = (unit * ((seq_len + 1) // cycle + 1))[:seq_len + 1]
                self.patterns.append(('alt', torch.tensor(pattern, dtype=torch.long)))

        # Pattern 5: Arithmetic-like (increment by 1, 2, 3...)
        for step in range(1, 5):
            vals = [(200 + (i * step) % 50) for i in range(seq_len + 1)]
            self.patterns.append(('arith', torch.tensor(vals, dtype=torch.long)))

        print(f"  Created {len(self.patterns)} synthetic patterns")

    def get_batch(self, batch_size, device='cpu'):
        """Sample a batch of (input, target) pairs."""
        indices = torch.randint(0, len(self.patterns), (batch_size,))
        inputs = []
        targets = []
        for idx in indices:
            _, seq = self.patterns[idx]
            inputs.append(seq[:-1])    # tokens 0..seq_len-1
            targets.append(seq[1:])    # tokens 1..seq_len
        return (torch.stack(inputs).to(device),
                torch.stack(targets).to(device))


# ================================================================
# Training loop
# ================================================================

def train(
    hidden_dim=512,
    n_heads=8,
    n_scales=4,
    iters_per_scale=4,
    vocab_size=32000,
    lora_rank=8,
    batch_size=16,
    seq_len=64,
    n_steps=10000,
    lr=3e-4,
    warmup_steps=500,
    log_every=100,
    eval_every=1000,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 1: FRR FROM-SCRATCH LANGUAGE MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Build model
    print("Building FRR model...")
    model = FRRFromScratch(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_scales=n_scales,
        iters_per_scale=iters_per_scale,
        vocab_size=vocab_size,
        lora_rank=lora_rank,
    ).to(device)

    param_counts = model.count_params()
    print(f"  Architecture: {n_scales} scales x {iters_per_scale} iters = "
          f"{n_scales * iters_per_scale} effective layers")
    print(f"  Parameter breakdown:")
    for name, count in param_counts.items():
        print(f"    {name:25s}: {count:>10,} ({count * 4 / 1e6:.2f} MB)")
    print()

    # Build dataset
    print("Building synthetic dataset...")
    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len)
    print()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))

    def cosine_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # Training
    print(f"Training for {n_steps} steps (batch={batch_size}, seq_len={seq_len})...")
    print("-" * 70)

    model.train()
    losses = []
    best_loss = float('inf')
    t0 = time.time()

    for step in range(1, n_steps + 1):
        inputs, targets = dataset.get_batch(batch_size, device)

        logits = model(inputs)  # (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-log_every:]) / len(losses[-log_every:])
            current_lr = scheduler.get_last_lr()[0]
            steps_per_sec = step / elapsed
            print(f"  step {step:>6d}/{n_steps} | loss {avg_loss:.4f} | "
                  f"best {best_loss:.4f} | lr {current_lr:.2e} | "
                  f"{steps_per_sec:.1f} steps/s")

        if step % eval_every == 0:
            eval_generation(model, dataset, device, step)

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print()

    # Final loss curve summary (10 checkpoints)
    print("Loss curve (sampled every 10%):")
    for i in range(10):
        idx = int((i + 1) * len(losses) / 10) - 1
        print(f"  {int((i+1)*10):>3d}% (step {idx+1:>5d}): {losses[idx]:.4f}")
    print()

    return model, dataset, losses


# ================================================================
# Evaluation: generate completions
# ================================================================

@torch.no_grad()
def eval_generation(model, dataset, device, step, n_examples=4):
    """Generate completions from prompts to show the model learned patterns."""
    model.eval()
    print(f"\n  --- Generation samples (step {step}) ---")

    for i in range(min(n_examples, len(dataset.patterns))):
        ptype, seq = dataset.patterns[i]
        # Use first 8 tokens as prompt, generate next 16
        prompt_len = 8
        gen_len = 16
        prompt = seq[:prompt_len].unsqueeze(0).to(device)
        generated = generate(model, prompt, gen_len)

        prompt_toks = prompt[0].tolist()
        gen_toks = generated[0, prompt_len:].tolist()
        expected_toks = seq[prompt_len:prompt_len + gen_len].tolist()

        # Count how many generated tokens match expected
        correct = sum(1 for g, e in zip(gen_toks, expected_toks) if g == e)
        acc = correct / len(expected_toks) * 100

        print(f"    [{ptype:>6s}] prompt: {prompt_toks}")
        print(f"           expected: {expected_toks}")
        print(f"           got:      {gen_toks}  ({acc:.0f}% match)")

    print()
    model.train()


@torch.no_grad()
def generate(model, prompt_tokens, n_tokens, temperature=0.0):
    """Autoregressive generation from a prompt."""
    tokens = prompt_tokens.clone()  # (1, T)
    for _ in range(n_tokens):
        logits = model(tokens)  # (1, T, V)
        next_logits = logits[:, -1, :]  # (1, V)
        if temperature == 0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens


# ================================================================
# Final comprehensive evaluation
# ================================================================

@torch.no_grad()
def final_evaluation(model, dataset, device):
    """Run comprehensive generation evaluation after training."""
    model.eval()
    print("=" * 70)
    print("FINAL EVALUATION: Can FRR learn language patterns?")
    print("=" * 70)

    total_correct = 0
    total_tokens = 0

    # Test all pattern types
    pattern_type_results = {}
    for i, (ptype, seq) in enumerate(dataset.patterns):
        prompt_len = 8
        gen_len = min(24, len(seq) - prompt_len)
        prompt = seq[:prompt_len].unsqueeze(0).to(device)
        generated = generate(model, prompt, gen_len)

        gen_toks = generated[0, prompt_len:].tolist()
        expected_toks = seq[prompt_len:prompt_len + gen_len].tolist()

        correct = sum(1 for g, e in zip(gen_toks, expected_toks) if g == e)

        if ptype not in pattern_type_results:
            pattern_type_results[ptype] = {'correct': 0, 'total': 0, 'count': 0}
        pattern_type_results[ptype]['correct'] += correct
        pattern_type_results[ptype]['total'] += len(expected_toks)
        pattern_type_results[ptype]['count'] += 1
        total_correct += correct
        total_tokens += len(expected_toks)

    print("\nResults by pattern type:")
    print(f"  {'Type':>10s}  {'Patterns':>8s}  {'Accuracy':>10s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}")
    for ptype, r in sorted(pattern_type_results.items()):
        acc = r['correct'] / r['total'] * 100
        print(f"  {ptype:>10s}  {r['count']:>8d}  {acc:>9.1f}%")

    overall_acc = total_correct / total_tokens * 100
    print(f"\n  Overall generation accuracy: {overall_acc:.1f}% "
          f"({total_correct}/{total_tokens} tokens)")

    # Verdict
    print()
    if overall_acc > 90:
        print("  VERDICT: FRR learns patterns excellently. Architecture is viable!")
    elif overall_acc > 60:
        print("  VERDICT: FRR learns patterns well. Architecture shows promise.")
    elif overall_acc > 30:
        print("  VERDICT: FRR learns some patterns. Needs tuning but fundamentals work.")
    else:
        print("  VERDICT: FRR struggles. May need architecture changes.")
    print()

    # Show a few detailed examples
    print("Detailed generation examples:")
    print("-" * 70)
    examples = [0, len(dataset.patterns) // 3,
                2 * len(dataset.patterns) // 3, len(dataset.patterns) - 1]
    for i in examples:
        if i >= len(dataset.patterns):
            continue
        ptype, seq = dataset.patterns[i]
        prompt_len = 8
        gen_len = 32
        prompt = seq[:prompt_len].unsqueeze(0).to(device)
        generated = generate(model, prompt, gen_len)

        gen_toks = generated[0, prompt_len:prompt_len + gen_len].tolist()
        expected_toks = seq[prompt_len:prompt_len + gen_len].tolist()
        correct = sum(1 for g, e in zip(gen_toks, expected_toks) if g == e)
        acc = correct / len(expected_toks) * 100

        print(f"  Pattern type: {ptype}")
        print(f"    Prompt:   {seq[:prompt_len].tolist()}")
        print(f"    Expected: {expected_toks}")
        print(f"    Got:      {gen_toks}")
        print(f"    Accuracy: {acc:.0f}%")
        print()


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    model, dataset, losses = train(
        hidden_dim=512,
        n_heads=8,
        n_scales=4,
        iters_per_scale=4,   # 16 effective layers
        vocab_size=32000,
        lora_rank=8,
        batch_size=16,
        seq_len=64,
        n_steps=10000,
        lr=3e-4,
        warmup_steps=500,
        log_every=100,
        eval_every=2000,
    )

    device = next(model.parameters()).device
    final_evaluation(model, dataset, device)

    # Save checkpoint
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'frr_scratch_phase1.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'config': {
            'hidden_dim': 512,
            'n_heads': 8,
            'n_scales': 4,
            'iters_per_scale': 4,
            'vocab_size': 32000,
            'lora_rank': 8,
        }
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
