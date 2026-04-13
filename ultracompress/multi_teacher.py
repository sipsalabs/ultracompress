"""
MULTI-TEACHER FUSION — Learn from multiple models simultaneously.

Single teacher = blind person teaching archery.
Multiple teachers = JURY of blind people. Where they AGREE,
the signal is strong. Where they DISAGREE, nobody knows.

The student learns:
1. WHERE teachers agree → match that (high confidence)
2. WHERE teachers disagree → learn independently (low confidence)

This naturally handles "teacher is wrong" — disagreement
signals uncertainty, and the student learns to be uncertain
there too (or find its own answer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_teacher_loss(student_logits, teacher_logits_list, temperature=2.0):
    """Distillation loss from multiple teachers.

    Where teachers agree: strong KL signal.
    Where teachers disagree: weighted by agreement.

    teacher_logits_list: list of (B, S, V) tensors from different teachers.
    """
    n_teachers = len(teacher_logits_list)

    # Get teacher distributions
    teacher_probs = [F.softmax(tl / temperature, dim=-1) for tl in teacher_logits_list]

    # Mean teacher distribution
    mean_probs = sum(teacher_probs) / n_teachers  # (B, S, V)

    # Agreement: how much do teachers agree? (low entropy of mean = high agreement)
    # Variance across teachers at each position
    teacher_stack = torch.stack(teacher_probs, dim=0)  # (T, B, S, V)
    variance = teacher_stack.var(dim=0).sum(dim=-1)  # (B, S) — per-position variance

    # Agreement weight: high where variance is low
    max_var = variance.max().clamp(min=1e-8)
    agreement = 1.0 - (variance / max_var)  # (B, S) in [0, 1]
    agreement = agreement.unsqueeze(-1)  # (B, S, 1) for broadcasting

    # Student loss: KL to mean teacher, weighted by agreement
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Per-position KL
    kl_per_pos = F.kl_div(student_log_probs, mean_probs, reduction='none').sum(dim=-1)  # (B, S)

    # Weight by agreement
    weighted_kl = (kl_per_pos * agreement.squeeze(-1)).mean()

    # Also: where teachers DISAGREE, add entropy regularization
    # (encourage student to be uncertain where teachers are uncertain)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)  # (B, S)
    disagreement = 1.0 - agreement.squeeze(-1)
    entropy_bonus = (student_entropy * disagreement).mean()

    return weighted_kl * (temperature ** 2) - 0.1 * entropy_bonus
