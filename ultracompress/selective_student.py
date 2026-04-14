"""
SELECTIVE STUDENT — The model decides when to trust the teacher.

The teacher is wrong sometimes. We proved it — diverging from the
teacher improves PPL. But the teacher is RIGHT sometimes too.

What if the student could DETECT when the teacher is reliable
and only learn from those moments?

How it works:
1. Student sees input, makes prediction
2. Teacher makes prediction
3. A tiny "trust gate" compares student confidence vs teacher confidence
4. High trust: learn from teacher (KL distillation)
5. Low trust: learn from data (next-token prediction, no teacher)

The trust gate learns WHEN the teacher is helpful.
Over time, the student trusts the teacher less and less as it
develops its own understanding. Like growing up.

This combines:
- Distillation (for bootstrapping — learn fast from teacher)
- From-scratch (for independence — break past teacher ceiling)
- Automatically — the model decides when to switch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustGate(nn.Module):
    """Decides whether to trust the teacher at each position.

    Compares student and teacher confidence levels.
    If teacher is very confident and student isn't → trust teacher.
    If both are confident but disagree → trust self (teacher might be wrong).
    If neither is confident → learn from data directly.
    """
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        # Takes: student entropy, teacher entropy, agreement level
        self.gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Bias toward trusting teacher initially (warm start)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(self, student_logits, teacher_logits):
        """Returns trust score per position: 1.0 = trust teacher, 0.0 = trust self."""
        # Student confidence (negative entropy)
        s_probs = F.softmax(student_logits, dim=-1)
        s_entropy = -(s_probs * (s_probs + 1e-10).log()).sum(dim=-1)  # (B, S)

        # Teacher confidence
        t_probs = F.softmax(teacher_logits, dim=-1)
        t_entropy = -(t_probs * (t_probs + 1e-10).log()).sum(dim=-1)  # (B, S)

        # Agreement (do they predict the same top token?)
        s_top = student_logits.argmax(dim=-1)
        t_top = teacher_logits.argmax(dim=-1)
        agreement = (s_top == t_top).float()  # (B, S)

        # Stack features
        features = torch.stack([s_entropy, t_entropy, agreement], dim=-1)  # (B, S, 3)

        # Trust score
        trust = torch.sigmoid(self.gate(features)).squeeze(-1)  # (B, S)
        return trust


def selective_loss(student_logits, teacher_logits, targets, trust_gate, step, total_steps):
    """Combined loss that balances teacher distillation and self-learning.

    Early training: mostly trust teacher (fast bootstrap)
    Late training: mostly trust self (break past ceiling)
    The trust gate learns when each is appropriate.
    """
    B, S, V = student_logits.shape

    # Trust score per position
    trust = trust_gate(student_logits.detach(), teacher_logits)  # (B, S)

    # Teacher loss (KL distillation)
    T = max(2.0, 5.0 * (1 - step / total_steps))
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='none'
    ).sum(dim=-1) * T * T  # (B, S)

    # Self loss (next-token prediction, no teacher)
    ntp_loss = F.cross_entropy(
        student_logits[:, :-1].reshape(-1, V),
        targets[:, 1:].reshape(-1),
        reduction='none'
    ).reshape(B, S - 1)  # (B, S-1)

    # Pad ntp_loss to match kl_loss shape
    ntp_loss = F.pad(ntp_loss, (0, 1), value=0)  # (B, S)

    # Weighted combination: trust * teacher + (1-trust) * self
    combined = trust * kl_loss + (1 - trust) * ntp_loss

    # Also train the trust gate itself (minimize total loss)
    return combined.mean()
