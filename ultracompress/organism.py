"""
COMPUTATIONAL ORGANISM — Process-based intelligence.

NOT a neural network. NOT weights being multiplied.
A tiny program that COMPUTES by running.

Inspired by:
- FRR: one function + recursion = full model
- Athena's organisms: 4 neurons beat 24, evolved not trained
- Octopus: walnut brain, intelligence in the PROCESS
- DNA: 750MB encodes a human brain through GROWTH RULES

The organism is:
- A tiny state vector (the "cell")
- A set of update rules (the "DNA")
- Applied recursively (the "life cycle")
- Intelligence EMERGES from execution

For existing models: extract the organism that reproduces behavior
For new models: evolve the organism from scratch

This is step 3 of the vision:
  Step 1: FRR (weights redundant) ✓
  Step 2: Seed (core + corrections) — sequential failed, joint pending
  Step 3: THIS — no weights at all, just a process
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UpdateRule(nn.Module):
    """A single update rule — the DNA of the organism.

    Instead of matrix multiplication (W @ x), this applies a RULE:
    a tiny parameterized function that transforms the state.

    The rule is applied to EVERY position in the state simultaneously
    (like cellular automata) — local computation, global emergence.
    """
    def __init__(self, state_dim, rule_dim=32):
        super().__init__()
        # The rule is tiny — just rule_dim parameters that define
        # how each state element interacts with its neighbors
        self.rule_dim = rule_dim

        # Interaction kernel: how neighbors influence each other
        # This is NOT a weight matrix — it's a RULE that generates behavior
        self.kernel = nn.Parameter(torch.randn(rule_dim) * 0.01)

        # State mixing: how the global state feeds back
        self.mix = nn.Parameter(torch.zeros(rule_dim))

        # Output projection: tiny, maps rule output back to state
        self.project = nn.Linear(rule_dim, state_dim, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, state):
        """state: (B, S, D) -> updated state: (B, S, D)"""
        B, S, D = state.shape

        # Local interaction: each position looks at nearby positions
        # Using circular convolution (every position interacts with rule_dim neighbors)
        # Pad and unfold to get local neighborhoods
        k = self.rule_dim
        padded = F.pad(state, (0, 0, k//2, k//2), mode='constant', value=0)

        # Extract neighborhoods: (B, S, D, k)
        neighborhoods = padded.unfold(1, k, 1)  # (B, S, D, k)

        # Apply kernel (element-wise rule application)
        # kernel: (k,) broadcasts across (B, S, D, k)
        activated = neighborhoods * self.kernel  # (B, S, D, k)

        # Mix with global state mean
        global_signal = state.mean(dim=1, keepdim=True)  # (B, 1, D)
        mix_signal = (global_signal.unsqueeze(-1) * self.mix).sum(dim=-1)  # (B, 1, D)

        # Combine local + global
        local = activated.sum(dim=-1)  # (B, S, D)
        combined = torch.tanh(local + mix_signal)

        # Simple residual: combined already has shape (B, S, D), scale it down
        return state + combined * 0.1  # Small residual update


class Organism(nn.Module):
    """A computational organism that generates intelligence through execution.

    The organism has:
    - A state (hidden representation, like a cell's internal state)
    - DNA (a set of tiny update rules, <1000 params each)
    - A life cycle (recursive application of rules)
    - Modulation (which rules to activate at each step)

    Total params: DNA rules (~1000 each) * n_rules + modulation
    For n_rules=8, rule_dim=32: ~8000 params of DNA
    Compare: FRR shared block = 7,000,000 params

    The intelligence ISN'T in the params. It's in the PROCESS of
    running the rules recursively. Same rules, different steps,
    different emergent behavior.
    """
    def __init__(self, hidden_dim, n_rules=8, rule_dim=32,
                 n_cycles=28, vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_rules = n_rules
        self.n_cycles = n_cycles

        # DNA: the update rules (tiny!)
        self.rules = nn.ModuleList([
            UpdateRule(hidden_dim, rule_dim) for _ in range(n_rules)
        ])

        # Rule selector: which rule(s) to apply at each cycle step
        # Static selection (learned schedule, not input-dependent)
        self.schedule = nn.Parameter(torch.randn(n_cycles, n_rules) * 0.1)

        # Cycle scaling (like FRR's iter_scale)
        self.cycle_scale = nn.Parameter(torch.ones(n_cycles))

        # State norm (stabilizes recursive application)
        self.norm = nn.RMSNorm(hidden_dim)

        # Embedding and head
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.out_norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.out_norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        # Initialize state from embeddings
        state = self.embed(tokens).float()

        # Life cycle: apply rules recursively
        for cycle in range(self.n_cycles):
            # Select rules for this cycle (soft selection)
            weights = F.softmax(self.schedule[cycle], dim=0)  # (n_rules,)

            # Apply weighted combination of rules
            update = torch.zeros_like(state)
            for i, rule in enumerate(self.rules):
                if weights[i] > 0.01:  # Skip negligible rules
                    update = update + weights[i] * (rule(state) - state)

            # Apply with cycle scaling
            state = state + update * self.cycle_scale[cycle]

        state = self.out_norm(state)
        return self.lm_head(state)

    def dna_size(self):
        """How big is the organism's DNA?"""
        rule_params = sum(p.numel() for p in self.rules.parameters())
        schedule_params = self.schedule.numel() + self.cycle_scale.numel()
        return {
            'rules': rule_params,
            'schedule': schedule_params,
            'total_dna': rule_params + schedule_params,
        }
