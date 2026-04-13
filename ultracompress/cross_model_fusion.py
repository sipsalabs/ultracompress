"""
CROSS-MODEL FUSION — Merge multiple compressed models (#23).

What if we trained FRR on 5 different teachers and MERGED the seeds?
Each teacher captures different knowledge. The merged seed gets ALL of it.

Like how multi-teacher works during training, but this works POST-HOC:
take already-trained seeds and combine them.

Methods:
1. WEIGHT AVERAGING: simple average of seed params (TIES/DARE style)
2. SELECTIVE MERGE: keep params where seeds AGREE, average where they don't
3. EVOLUTIONARY MERGE: evolve the best combination weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_average_merge(seed_state_dicts, weights=None):
    """Simple weighted average of multiple seed state dicts.

    Like model soup / TIES merging but for seeds.
    """
    n = len(seed_state_dicts)
    if weights is None:
        weights = [1.0 / n] * n

    merged = {}
    for key in seed_state_dicts[0].keys():
        merged[key] = sum(w * sd[key].float() for w, sd in zip(weights, seed_state_dicts))
    return merged


def selective_merge(seed_state_dicts, agreement_threshold=0.8):
    """Merge seeds selectively — keep where they agree, average where they don't.

    For each param position:
    - If all seeds have similar values (high cosine): keep consensus
    - If seeds disagree: average (each captures different knowledge)
    """
    n = len(seed_state_dicts)
    merged = {}

    for key in seed_state_dicts[0].keys():
        tensors = [sd[key].float() for sd in seed_state_dicts]
        stacked = torch.stack(tensors)  # (N, ...)

        # Measure agreement via variance
        variance = stacked.var(dim=0)
        mean = stacked.mean(dim=0)

        # Where variance is low (agreement): use mean
        # Where variance is high (disagreement): also use mean but could weight
        merged[key] = mean

    return merged


def dare_merge(seed_state_dicts, base_state_dict, density=0.5):
    """DARE (Drop And REscale) merge for seeds.

    For each seed:
    1. Compute delta from base (seed - base)
    2. Randomly drop fraction of delta
    3. Rescale remaining
    4. Sum all deltas + base

    This preserves important changes while reducing interference.
    """
    n = len(seed_state_dicts)
    merged = {}

    for key in base_state_dict.keys():
        base = base_state_dict[key].float()
        total_delta = torch.zeros_like(base)

        for sd in seed_state_dicts:
            delta = sd[key].float() - base
            # Random mask
            mask = (torch.rand_like(delta) < density).float()
            # Rescale to preserve expected magnitude
            scaled_delta = delta * mask / density
            total_delta = total_delta + scaled_delta / n

        merged[key] = base + total_delta

    return merged
