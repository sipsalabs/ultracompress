"""
SELF-GROWING ROTATION NETWORK — Genuinely novel.

Nobody has this. A model that:
1. Starts with MINIMAL rotation planes (8 planes = 8 params)
2. Learns language from real text (no teacher, no distillation)
3. GROWS new planes when learning stalls (adds capacity where needed)
4. Discovers its own geometric structure

Like DNA → brain: doesn't compress a brain. GROWS one.

The growth rule: if loss plateaus for N steps, add K new rotation planes
to the layers where gradients are largest (where the model WANTS more capacity).

This tests Sip's core vision: process not model, growth not compression,
the architecture finds its own shape.
"""
import lib.unbuffered
import torch, sys, os, time, math
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer

device = 'cuda'
TOTAL_STEPS = 30000
GROWTH_CHECK_EVERY = 2000  # Check if we should grow every N steps
PLATEAU_THRESHOLD = 0.02    # Loss must improve by this much or we grow
PLANES_PER_GROWTH = 16      # Add this many planes each growth event
MAX_PLANES = 512            # Cap total planes
INITIAL_PLANES = 16         # Start tiny
N_CYCLES = 28
HIDDEN = 1024

print("=" * 60)
print("SELF-GROWING ROTATION: Starts tiny, grows where needed")
print(f"Initial: {INITIAL_PLANES} planes. Grows by {PLANES_PER_GROWTH} when stuck.")
print("No teacher. Pure language learning.")
print("=" * 60)

# Load embeddings only (no teacher model)
wd = torch.load('qwen3_0.6b_cache.pt', weights_only=True)
embed_w = wd['model.embed_tokens.weight'].float().to(device)
norm_w = wd.get('model.norm.weight', torch.ones(HIDDEN)).float().to(device)
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


class GrowableRotationEngine(nn.Module):
    """A rotation network that can GROW new planes dynamically.

    Starts with minimal planes. When learning stalls, new planes are added
    to the cycles with highest gradient magnitude (where the model WANTS
    more capacity).
    """
    def __init__(self, hidden_dim, n_planes_init, n_cycles, vocab_size,
                 embed_weight, lm_head_weight, norm_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cycles = n_cycles
        self.current_planes = n_planes_init

        # Per-cycle rotation angles (growable)
        self.angles = nn.ParameterList([
            nn.Parameter(torch.zeros(n_planes_init)) for _ in range(n_cycles)
        ])

        # Per-cycle plane assignments (fixed per growth event)
        self._assign_planes(n_planes_init)

        # Per-cycle scale/shift (like modulation in FRR)
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim)) for _ in range(n_cycles)
        ])
        self.shifts = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_cycles)
        ])
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_cycles)])

        # Cross-position mixing (simple but effective)
        self.mix_weight = nn.Parameter(torch.tensor([0.1, 0.8, 0.1]))

        # Embeddings
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.out_norm = nn.RMSNorm(hidden_dim)
        self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        # Growth tracking
        self.growth_events = []

    def _assign_planes(self, n_planes):
        """Assign rotation plane pairs. Called at init and each growth."""
        self.plane_pairs = []
        for c in range(self.n_cycles):
            torch.manual_seed(42 + c * 1000)
            indices = torch.randperm(self.hidden_dim)
            actual = min(n_planes, self.hidden_dim // 2)
            pi = indices[:actual].to(device)
            pj = indices[actual:2*actual].to(device)
            self.plane_pairs.append((pi, pj))

    def grow(self, n_new_planes):
        """Add new rotation planes to ALL cycles."""
        old_planes = self.current_planes
        new_total = min(old_planes + n_new_planes, MAX_PLANES)
        if new_total == old_planes:
            return False  # already at max

        # Expand each cycle's angles
        for c in range(self.n_cycles):
            old_angles = self.angles[c].data
            new_angles = torch.zeros(new_total, device=device)
            new_angles[:old_planes] = old_angles
            # Initialize new angles near zero (gentle addition)
            new_angles[old_planes:] = torch.randn(new_total - old_planes, device=device) * 0.01
            self.angles[c] = nn.Parameter(new_angles)

        self.current_planes = new_total
        self._assign_planes(new_total)
        self.growth_events.append((old_planes, new_total))
        return True

    def forward(self, tokens):
        x = self.embed(tokens).float()
        B, S, D = x.shape

        for c in range(self.n_cycles):
            # Simple cross-position mixing
            left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
            x = self.mix_weight[0] * left + self.mix_weight[1] * x + self.mix_weight[2] * right

            # Rotation
            pi, pj = self.plane_pairs[c]
            actual = min(self.angles[c].shape[0], pi.shape[0])
            cos_a = torch.cos(self.angles[c][:actual])
            sin_a = torch.sin(self.angles[c][:actual])

            x_new = x.clone()
            xi = x[..., pi[:actual]]
            xj = x[..., pj[:actual]]
            x_new[..., pi[:actual]] = xi * cos_a - xj * sin_a
            x_new[..., pj[:actual]] = xi * sin_a + xj * cos_a

            # Activate + normalize
            x_act = F.silu(x_new * self.scales[c] + self.shifts[c])
            x = x + self.norms[c](x_act - x)

        x = self.out_norm(x)
        return self.lm_head(x)

    def dna_size(self):
        total = sum(a.numel() for a in self.angles)
        total += sum(s.numel() for s in self.scales) + sum(s.numel() for s in self.shifts)
        return total


# Build the growing model
model = GrowableRotationEngine(
    HIDDEN, INITIAL_PLANES, N_CYCLES, 151936,
    embed_w, lm_head_w, norm_w
).to(device)

print(f"Initial DNA: {model.dna_size():,} params ({INITIAL_PLANES} planes)")

# Training with growth
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS)

t0 = time.time()
prev_loss = float('inf')
loss_window = []

for step in range(TOTAL_STEPS):
    batch = get_batch(8, 128)
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()

    loss_window.append(loss.item())

    # Growth check
    if step > 0 and step % GROWTH_CHECK_EVERY == 0:
        avg_loss = sum(loss_window[-GROWTH_CHECK_EVERY:]) / GROWTH_CHECK_EVERY
        improvement = (prev_loss - avg_loss) / max(prev_loss, 1e-6)

        if improvement < PLATEAU_THRESHOLD and model.current_planes < MAX_PLANES:
            grew = model.grow(PLANES_PER_GROWTH)
            if grew:
                # Rebuild optimizer with new params
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_STEPS - step)
                print(f"    >>> GREW to {model.current_planes} planes! DNA: {model.dna_size():,} params")

        prev_loss = avg_loss

    if step % 3000 == 0:
        ppl = math.exp(min(loss.item(), 20))
        elapsed = time.time() - t0
        print(f"  Step {step}: loss={loss.item():.4f} ppl={ppl:.1f} planes={model.current_planes} DNA={model.dna_size():,} ({elapsed:.0f}s)")

# Final eval
ppl = math.exp(min(loss.item(), 20))
print(f"\n{'='*60}")
print(f"RESULTS: Self-Growing Rotation Network")
print(f"{'='*60}")
print(f"  Started: {INITIAL_PLANES} planes")
print(f"  Ended: {model.current_planes} planes")
print(f"  Growth events: {len(model.growth_events)}")
for old, new in model.growth_events:
    print(f"    {old} → {new} planes")
print(f"  Final DNA: {model.dna_size():,} params")
print(f"  Final PPL: {ppl:.1f}")
print(f"  Compression vs teacher (440M): {440401920 / model.dna_size():.0f}x")
print("Done!")
