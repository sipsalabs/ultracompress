"""
CRYSTAL NETWORK — Architecture emerges from energy minimization.

Nobody designs a crystal. Atoms find lowest-energy configurations
through physics. The structure EMERGES from the interactions.

What if neural computation worked the same way?

Instead of: design architecture → train weights
This does: define energy function → let computation crystallize

How it works:
1. Tokens are "atoms" in a high-dimensional energy landscape
2. Each atom has a position (hidden state) and a charge (embedding)
3. Atoms interact through an energy function (learned, tiny)
4. The system iterates toward equilibrium (like FRR's recursion)
5. The equilibrium state IS the output

The "model" is the energy function — how atoms interact.
Intelligence emerges from the equilibrium of interactions.

Like how protein folding works: the sequence defines the energy,
the structure (fold) emerges from minimizing that energy.
AlphaFold proved this works. We're doing it for language.

This is the most radical departure from transformers yet:
- No layers, no attention, no FFN
- Just an energy landscape and equilibrium
- The computation IS the process of reaching equilibrium
- Naturally parallel, naturally iterative, naturally compressed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnergyFunction(nn.Module):
    """The learned energy function — how tokens interact.

    Defines pairwise interaction energy between token states.
    Lower energy = more compatible states.
    The system moves toward lowest energy = correct output.

    Inspired by Hopfield networks but continuous and learned.
    """
    def __init__(self, hidden_dim, n_energy_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_energy_heads
        self.head_dim = hidden_dim // n_energy_heads

        # Pairwise interaction: how do two token states affect each other?
        self.interaction_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.interaction_val = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Self-energy: what's the "natural" state for each token?
        self.self_energy = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Energy mixing
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def compute_force(self, x):
        """Compute the 'force' on each token — direction of energy decrease.

        Force = -gradient of energy = direction to move toward equilibrium.
        We approximate this with learned interactions (like attention but
        framed as physics, not routing).
        """
        B, S, D = x.shape

        # Pairwise forces (how other tokens push/pull this one)
        keys = self.interaction_key(x).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        vals = self.interaction_val(x).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Energy-based interaction (softmax = Boltzmann distribution)
        energy = (keys @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal: future tokens can't exert force on past tokens
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            energy = energy.masked_fill(mask, float('-inf'))
        weights = F.softmax(-energy, dim=-1)  # NEGATIVE energy: low energy = high probability

        pairwise_force = (weights @ vals).transpose(1, 2).reshape(B, S, D)

        # Self-energy force (tendency toward natural state)
        self_force = self.self_energy(x)

        # Combined force
        total_force = self.out_proj(pairwise_force + self_force)
        return total_force


class CrystalNet(nn.Module):
    """Language model via energy minimization.

    Tokens enter as high-energy states.
    Through iterative relaxation, they find equilibrium.
    The equilibrium state is the output.

    The SAME energy function is applied at every iteration (like FRR).
    The number of iterations = computational depth.
    More iterations = more refined equilibrium = better output.
    """
    def __init__(self, hidden_dim, n_energy_heads=8, n_iterations=28,
                 vocab_size=151936,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        self.n_iterations = n_iterations

        # THE ENERGY FUNCTION (shared across all iterations)
        self.energy = EnergyFunction(hidden_dim, n_energy_heads)

        # Per-iteration "temperature" (how much to move each step)
        # High temp early = big moves. Low temp late = fine-tuning.
        self.step_size = nn.Parameter(torch.linspace(0.5, 0.1, n_iterations))

        # Norms for stability
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_iterations)])

        # Per-iteration modulation (like FRR)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, hidden_dim)) for _ in range(n_iterations)
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(n_iterations)
        ])

        # Embeddings
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

        for i in range(self.n_iterations):
            h = self.norms[i](x) * self.gammas[i] + self.betas[i]

            # Compute force (direction of energy decrease)
            force = self.energy.compute_force(h)

            # Move toward equilibrium
            x = x + force * self.step_size[i]

        x = self.out_norm(x)
        return self.lm_head(x)
