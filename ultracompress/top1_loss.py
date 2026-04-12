"""
Top-1 focused loss functions for FRR distillation.

Standard KL divergence matches the full distribution but doesn't
specifically optimize for getting the #1 token right. These losses
add a penalty when the student's top-1 prediction differs from
the teacher's top-1.

Expected: improves T1 from ~48% toward 55-60% with minimal T10 impact.
"""
import torch
import torch.nn.functional as F


def top1_match_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5):
    """Combined KL + top-1 matching loss.

    Args:
        student_logits: (B, S, V) student model output
        teacher_logits: (B, S, V) teacher model output
        temperature: KL temperature
        alpha: weight of top-1 loss (0=pure KL, 1=pure top-1)

    Returns:
        Combined loss scalar
    """
    # Standard KL divergence
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature * temperature)

    # Top-1 cross-entropy: treat teacher's argmax as the "correct" label
    teacher_top1 = teacher_logits.argmax(dim=-1)  # (B, S)
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_top1.reshape(-1),
        reduction='mean'
    )

    return (1 - alpha) * kl + alpha * ce


def margin_ranking_loss(student_logits, teacher_logits, margin=1.0):
    """Ensure student ranks the teacher's #1 token above all others.

    Instead of matching the full distribution, this loss only cares
    about the RANKING — is the teacher's top token also the student's
    top token? Uses margin-based ranking loss.
    """
    B, S, V = student_logits.shape
    teacher_top1 = teacher_logits.argmax(dim=-1)  # (B, S)

    # Get student's score for teacher's top token
    top1_scores = student_logits.gather(2, teacher_top1.unsqueeze(-1)).squeeze(-1)  # (B, S)

    # Get student's max score (for any token)
    max_scores = student_logits.max(dim=-1).values  # (B, S)

    # Loss: push top1_scores above max_scores by margin
    # When top1 IS the max, loss = 0
    loss = F.relu(margin - (top1_scores - max_scores + margin))  # hinge loss

    return loss.mean()


def combined_distillation_loss(student_logits, teacher_logits,
                                temperature=2.0, kl_weight=0.5,
                                ce_weight=0.3, margin_weight=0.2):
    """Full distillation loss: KL + CE + margin ranking.

    Balances distribution matching (KL), top-1 accuracy (CE),
    and ranking correctness (margin).
    """
    # KL
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature * temperature)

    # CE on teacher's argmax
    teacher_top1 = teacher_logits.argmax(dim=-1)
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_top1.reshape(-1),
        reduction='mean'
    )

    # Margin ranking
    mr = margin_ranking_loss(student_logits, teacher_logits)

    return kl_weight * kl + ce_weight * ce + margin_weight * mr
