"""
FRR → Organism distillation.

Instead of distilling directly from the 440M teacher,
distill from the trained FRR (7.3M params) into the organism (66K params).

The FRR already captured the essential computation. The organism
just needs to learn the PROCESS that FRR uses, not the raw weights.

This is compression of compression:
  Teacher (440M) → FRR (7.3M) → Organism (66K)
  = 6,667x compression in two hops
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def distill_frr_to_organism(frr_model, organism, device='cuda',
                             steps=20000, lr=1e-3):
    """Distill a trained FRR model into an organism.

    The organism learns to replicate FRR's input-output behavior.
    Since FRR is already compressed, the organism's task is easier
    than learning from the full teacher.
    """
    print(f"Distilling FRR -> Organism...")
    frr_params = sum(p.numel() for p in frr_model.parameters())
    org_dna = organism.dna_size()['total_dna']
    print(f"  FRR: {frr_params:,} -> Organism DNA: {org_dna:,} ({frr_params//org_dna}x)")

    frr_model.eval()
    opt = torch.optim.AdamW(
        [p for p in organism.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for step in range(steps):
        torch.manual_seed(step * 7)
        tokens = torch.randint(100, 50000, (4, 32), device=device)

        with torch.no_grad():
            frr_logits = frr_model(tokens)

        org_logits = organism(tokens)

        T = max(2.0, 4.0 * (1 - step / steps))
        loss = F.kl_div(
            F.log_softmax(org_logits / T, dim=-1),
            F.softmax(frr_logits / T, dim=-1),
            reduction='batchmean') * (T * T)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 5000 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}")

    return organism
