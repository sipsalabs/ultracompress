"""
DUAL-OBJECTIVE FRR — Sip's split idea.

Two approaches to maximize BOTH T10 and T1:

1. Two-Phase Recursion:
   - Passes 1-20: "Broad phase" — standard KL loss, builds general representation
   - Passes 21-28: "Sharp phase" — top-1 CE loss, sharpens to exact token
   Different loss weighting per recursion depth.

2. Dual-Mode Block:
   - One block with two sets of modulation: broad_gamma/beta + sharp_gamma/beta
   - Early passes use broad modulation (soft distribution)
   - Late passes use sharp modulation (peaked distribution)
   - Smooth transition via learned gate

The insight: T10 and T1 optimize different things. T10 wants smooth
distributions (right neighborhood). T1 wants peaked distributions
(exact answer). Splitting the job across recursion depth lets one
block do both without compromise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualObjectiveFRR(nn.Module):
    """FRR with dual-phase recursion for T10 + T1 optimization.

    Passes 1-N_broad: broad modulation (soft, T10-focused)
    Passes N_broad+1 to total: sharp modulation (peaked, T1-focused)
    """
    def __init__(self, hidden_dim, n_heads, n_scales=4, iters_per_scale=7,
                 vocab_size=151936, ff_mult=1, sharp_fraction=0.25,
                 embed_weight=None, lm_head_weight=None, norm_weight=None):
        super().__init__()
        from .moonshot import FractalBlock

        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.iters_per_scale = iters_per_scale
        self.total_layers = n_scales * iters_per_scale

        # How many passes are "sharp" (T1-focused)
        self.n_sharp = max(1, int(self.total_layers * sharp_fraction))
        self.n_broad = self.total_layers - self.n_sharp

        # ONE shared block (same weights for both phases)
        self.block = FractalBlock(hidden_dim, n_heads, ff_mult)

        # BROAD modulation (T10-focused, soft distributions)
        self.broad_gamma = nn.Parameter(torch.ones(self.n_broad, hidden_dim))
        self.broad_beta = nn.Parameter(torch.zeros(self.n_broad, hidden_dim))

        # SHARP modulation (T1-focused, peaked distributions)
        self.sharp_gamma = nn.Parameter(torch.ones(self.n_sharp, hidden_dim))
        self.sharp_beta = nn.Parameter(torch.zeros(self.n_sharp, hidden_dim))

        # Per-layer iteration scaling
        self.iter_scale = nn.Parameter(torch.ones(self.total_layers))

        # Sharpening layer: additional transform that peaks the distribution
        self.sharpen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        # Initialize near-identity
        nn.init.zeros_(self.sharpen[-1].weight)

        # Embedding and head
        if embed_weight is not None:
            self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm = nn.RMSNorm(hidden_dim)
        if norm_weight is not None:
            self.norm.weight = nn.Parameter(norm_weight, requires_grad=False)

    def forward(self, tokens):
        x = self.embed(tokens).float()

        # BROAD PHASE: passes 0 to n_broad-1
        for i in range(self.n_broad):
            gamma = self.broad_gamma[i]
            beta = self.broad_beta[i]
            iter_s = self.iter_scale[i]
            x = x + (self.block(x, gamma, beta) - x) * iter_s

        # SHARP PHASE: passes n_broad to total-1
        for i in range(self.n_sharp):
            gamma = self.sharp_gamma[i]
            beta = self.sharp_beta[i]
            iter_s = self.iter_scale[self.n_broad + i]
            x_block = self.block(x, gamma, beta)
            # Apply sharpening transform in sharp phase
            x_sharp = x_block + self.sharpen(x_block)
            x = x + (x_sharp - x) * iter_s

        x = self.norm(x)
        return self.lm_head(x)


def dual_phase_loss(student_logits, teacher_logits, step, total_steps,
                    temperature=2.0):
    """Loss that shifts from T10-focused to T1-focused over training.

    Early training: pure KL (match distribution, maximize T10)
    Late training: KL + CE on argmax (sharpen to T1)
    """
    # KL divergence (always)
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature * temperature)

    # CE on teacher's argmax (ramp up over training)
    teacher_top1 = teacher_logits.argmax(dim=-1)
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_top1.reshape(-1),
        reduction='mean'
    )

    # Ramp CE weight from 0 to 0.5 over training
    ce_weight = min(0.5, step / total_steps)
    kl_weight = 1.0 - ce_weight

    return kl_weight * kl + ce_weight * ce
