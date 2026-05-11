# Customer Phase 0 — Generic Operational Playbook

**Use:** Sip's day-by-day operational guide for ANY signed Phase 0 customer engagement. Applies to both aerospace track (Varion / Hadrian / Saronic / Anduril) and IaaS track (Together / Fireworks / Lambda / CoreWeave / Groq).
**Variant-specific overrides:** see `docs/VARION_PHASE_0_OPERATIONAL_RUNBOOK.md` for Varion specifically and `docs/IAAS_PHASE_0_POC_SCOPE_2026_05.md` for the IaaS shorter-cycle variant.

---

## Pre-kickoff (T-7 to T-0)

### NDA + SOW execution
- NDA signed by both parties before any technical artifact crosses.
- SOW signed before Day 1.
- Invoice for 50% kickoff fee sent within 24 hours of SOW signature.
- Calendar invite for Day 1 kickoff sent same-day.

### Customer-side prerequisites confirmed
- Designated technical contact (one engineer, with bandwidth).
- Designated business contact (signs deliverables).
- Model class identified.
- Evaluation suite specification received OR mutually agreed default.
- Tolerance specification received OR mutually agreed default.
- Transfer mechanism agreed (Sipsa pulls from public HF / customer ships private checkpoint to Sipsa secure storage / Sipsa enters customer cloud as no-model-sharing variant).

### Sipsa-side prerequisites
- `~/customer/<customer_short_name>/` directory created on Sipsa hardware.
- Compute reserved (typically dual 5090s for the Phase 0 week; cloud GPU credits if customer's model class doesn't fit on consumer hardware).
- Calibration data prepared (default: WikiText-2 validation; custom if customer specifies).
- LAB-NOTEBOOK section opened with engagement timeline.

---

## Day 1 — Kickoff and intake

### Morning (T+0 to T+4 hours)

**Welcome call (30 min, joint customer + Sipsa):**
- Confirm prerequisites are in place.
- Walk through Phase 0 deliverable structure (the written report sections per SOW).
- Confirm tolerance specification and acceptance criteria.
- Set daily end-of-day status email cadence.
- Confirm escalation path if any party hits a blocker.

**Intake of customer artifacts:**
- Receive checkpoint via agreed transfer (HF pull / S3 presigned / SFTP).
- Run integrity checks: SHA256 vs customer-emailed hash, file size sanity, parameter count via `python -c "import torch; sd = torch.load('checkpoint.pt', map_location='cpu'); print(sum(v.numel() for v in sd.values())/1e9, 'B params')"`.
- If integrity check fails: pause, email customer, do not proceed.

### Afternoon (T+4 to T+8 hours)

**Architecture identification:**
- Inspect state dict keys, dtypes, layer count, hidden dimension.
- Identify operator types: standard transformer (attention + FFN), Fourier Neural Operator, U-Net, Neural ODE, custom.
- Document findings in `~/customer/<name>/notes/architecture.md`.
- For non-standard architectures: flag in end-of-day-1 email if architecture identification will require more time than Day 1.

**Reference baseline reproduction:**
- Load checkpoint at full precision (fp32 or fp16 native).
- Run customer's evaluation suite. Capture reference predictions tensor.
- Compute customer's primary metric (relative L2 / PPL / accuracy, etc.).
- Wall-clock per sample, peak VRAM.
- This is the truth set. Every compressed run gets diffed against it.

**End-of-day-1 email (template):**
> Subject: <Customer> Phase 0 — Day 1 status
>
> Body: Checkpoint received, integrity verified ([SHA256 prefix]). Architecture identified as [class]. Baseline reproduced within [tolerance]. Day 2 begins compression evaluation.

---

## Day 2-3 — Compression evaluation

### Day 2: Apply compression pipeline

**Morning:**
- Apply scalar BPW-target quantization. For transformer LLMs, default BPW 5; for non-transformer, start with BPW 6 to give a clean baseline before pushing lower.
- Attach rank-32 calibration-fitted correction overlay per linear projection.
- Train correction overlay against teacher hidden states for 200 distillation steps per layer.

**Afternoon:**
- Evaluate compressed model against customer's evaluation suite.
- Compare per-metric scores against Day 1 reference baseline.
- Document per-layer reconstruction quality.

**Watchpoints:**
- Loss NaN, grad norm spikes, oscillation. Drop lr to 3e-5 and retry if observed.
- Per-layer hidden-MSE saturation (production sweet spot is 200 steps; 500 steps regresses end-to-end PPL — documented in LAB-NOTEBOOK).

### Day 3: Bit-rate sweep

- Run pipeline at BPW 3, 4, 5, 6 across the same model.
- Build Pareto table: bpw vs metric vs file size vs eval latency.
- Identify production target: lowest bpw where degradation is within customer tolerance.
- Save: `~/customer/<name>/results/bpw_sweep.json` plus markdown summary.

**End-of-day-3 email (template):**
> Subject: <Customer> Phase 0 — Day 3 update
>
> Body: BPW 3-6 sweep complete. Current best operating point is BPW [X] at [metric] = [Y]. [One sentence on stability.] Day 4-5 focus on full eval-set validation and edge cases.

---

## Day 4-5 — Validation and edge cases

### Day 4: Full eval-set validation

- Run chosen compressed config across customer's FULL evaluation suite, not the subset used during sweeps.
- Stratify by any customer-defined regime labels (model class, batch size, sequence length, customer-specific use case).
- Compression failure often hides in one regime while overall metric looks fine.
- Capture per-sample distribution, not just the mean.

### Day 5: Architecture-specific audits

For aerospace / scientific ML customers:
- **Conservation law audit.** Mass / momentum / energy conservation residuals. If compression breaks any by more than ~1e-4 relative, document which layers responsible.
- **Long-rollout stability.** PDE solvers compound bias over time steps; verify acceptable drift over customer-specified rollout horizon.
- **Numerical edge cases.** Boundary conditions, shock fronts, fp32-pinned critical paths.

For IaaS / standard transformer customers:
- **Tail-token quality.** Compressed model behavior on rare tokens vs common tokens.
- **Throughput measurement.** tokens/sec/GPU at customer's batch size and sequence length.
- **KV cache footprint.** Peak GPU memory under customer's workload (not just compression peak).

---

## Day 6-10 — Report drafting and delivery

### Day 6: Final report draft

Write `~/customer/<name>/deliverables/feasibility_report.md`. Sections:
1. **Setup summary.** Model class, parameters, layers, hidden dim, attention layout. BPW target. Calibration set provenance.
2. **Reference baseline reproduction.** Customer's stated baseline reproduced on Sipsa hardware against customer's evaluation suite.
3. **Compression measurement table.** Per-metric scores, per-layer profile, per-bpw Pareto.
4. **Comparison vs current production.** If customer is on AWQ / GPTQ / HQQ / fp16 today, side-by-side at the same bit-rate target.
5. **Edge case audit.** Architecture-specific (conservation laws OR tail-token quality OR throughput, depending on customer).
6. **Recommendation.** One of: (a) compression mechanism transfers cleanly, propose Phase 1 production integration; (b) compression mechanism transfers but needs per-customer tuning, priced as Phase 1 with longer scope; (c) compression mechanism does not transfer, full refund and walk recommendation.

### Day 7: Internal review

- Strip any pre-filing IP detail.
- Check for internal codename leakage (correction overlay, scalar quantization, low-rank refinement, k-cluster recovery, DSR-Q, SP-band, PCR — none should appear).
- Verify all numbers reproduce.
- Verify customer-stated tolerance language matches the SOW exactly.

### Day 8: Customer review

- Send report draft to customer's technical contact for fact-check.
- 48-hour review window.
- Address any clarifying questions.

### Day 9: Final polish

- Incorporate customer fact-check feedback.
- Generate any final figures / charts.
- Sanitize for NDA-compliant external sharing.

### Day 10: Delivery

- Deliverable package: `feasibility_report.pdf` + compressed-checkpoint demo + reproducibility receipt + Phase 1 scope sketch (if positive recommendation).
- Send via customer's preferred secure channel.
- Send invoice for second 50% per SOW.
- Schedule 30-min readout call with customer's technical + business contacts.

---

## Communication cadence

### Daily (Day 1-10 active engagement)
- **End-of-day status email.** Factual, ~3 sentences. What was done, what's next, any blockers. Use the templates above.

### Weekly (Phase 0 lasting >5 days)
- **Friday weekly summary.** What works, what doesn't, Phase 1 path candidates.

### Milestone-based
- **NDA signed.** Welcome packet (one-pager + technical FAQ + reproducibility setup guide).
- **SOW signed.** 50% kickoff invoice + Day 1 calendar invite.
- **Day 1 complete.** End-of-day-1 email (intake confirmed, baseline reproduced, compression eval starting).
- **BPW sweep complete.** Mid-engagement email (current best operating point + tomorrow's plan).
- **Report draft ready.** Customer review request email.
- **Final delivery.** Deliverable package + 50% completion invoice + readout call request.

### Anti-patterns
- DO NOT send updates more than once per day during active engagement (looks frantic).
- DO NOT skip the end-of-day-1 email under any circumstances (sets the engagement tone).
- DO NOT use "we" plural pronoun (Sipsa is solo until cofounder hires).
- DO NOT send long updates without numbers (calibrated voice = numbers, not adjectives).

---

## Customer artifacts to produce

For every Phase 0:
1. **Architecture identification note** (`~/customer/<name>/notes/architecture.md`).
2. **Day 1 baseline reproduction record** (`~/customer/<name>/results/baseline.json`).
3. **BPW sweep results JSON** (`~/customer/<name>/results/bpw_sweep.json`).
4. **Compressed checkpoint** at recommended bit-rate (per-layer .pt files plus manifest.json).
5. **Reproducibility receipt** (pip version, calibration parameters, training step count, random seed, hardware fingerprint).
6. **Feasibility report** (the deliverable PDF).
7. **Phase 1 scope sketch** (only if positive recommendation; price + timeline + deliverables).
8. **LAB-NOTEBOOK entry** for Sipsa-internal record (engagement timeline, what worked, what didn't, anything new learned about the customer's architecture class).

---

## Pricing tier guidance for Phase 1 scope

After Phase 0 outcome:

| Outcome | Phase 1 tier | Price | Timeline |
|---|---|---|---|
| Existing pipeline applies cleanly (transformer LLM, mainstream architecture) | Tier 1 | $50K-$70K | 4-6 weeks |
| Existing pipeline applies, customer-specific tuning needed | Tier 2 | $65K-$80K | 6-10 weeks |
| Hybrid mechanism stack required (one new mechanism beyond existing pipeline) | Tier 3 | $80K-$100K | 8-12 weeks |
| Full custom mechanism development required | Tier 4 / addendum | $100K+ | 12-20 weeks |
| No transfer — refund and walk | n/a | refund 50% Phase 0 fee | n/a |

Annual deployment license post-Phase-1: $150K-$500K per customer per deployed model class per year.

---

## Common failure modes

- **Customer slice doesn't load.** Wrong PyTorch version, missing custom op, dtype mismatch. Mitigation: pin torch version + minimal requirements.txt + container image.
- **Customer's tolerance is fp64.** Compression cannot meaningfully apply. Honest decline.
- **Customer's evaluation suite isn't reproducible.** Sipsa cannot land Day 1 baseline within stated tolerance. Pause, email, escalate to business contact.
- **Calibration data unavailable / proprietary.** Use default calibration; flag as potential mismatch in report.
- **Customer's metric isn't shippable** (e.g., requires customer's internal proprietary scoring). Use proxy metric; flag the substitution in report.
- **Transfer fails on customer's specific architecture** (e.g., spectral conv with complex weights breaks scalar quantization). Honest decline — that's a deliverable too.
- **Customer pulls scope mid-engagement** (asks for Phase 1 quote without finishing Phase 0). Stay disciplined: finish Phase 0 deliverable first, then Phase 1 conversation.

---

## Customer types and tactical adjustments

### Aerospace / defense
- ITAR / EAR pre-cleared (customer responsibility per SOW Section 8).
- Code-delivery model: Sipsa never receives weights.
- No-model-sharing protocol per `docs/AEROSPACE_NO_MODEL_SHARING_PILOT_PROTOCOL.md`.
- Conservation law audit included (Day 5).
- Phase 1 license includes BOM royalty for any embedded / edge deployment.

### IaaS / inference providers
- Faster cycle: 5-day Phase 0 ($5K) vs 10-day aerospace Phase 0 ($10-15K).
- Throughput measurement included (Day 4).
- Phase 1 includes vLLM / TensorRT-LLM / llama.cpp kernel integration.
- License includes customer-specific model class deployment.

### Frontier-lab evaluation
- Highest-tier engagement: Phase 0 $50-200K, Phase 1 $500K-$2M.
- Multi-model batch evaluation focus.
- Source escrow available.
- Most likely path: license to evaluation infrastructure rather than serving infrastructure.

---

## End of Phase 0 — what next

If customer signs Phase 1 same week:
- Calendar Phase 1 kickoff Monday following Phase 0 close.
- Update LAB-NOTEBOOK + OPS_DASHBOARD with engagement state.
- Notify YC update list (next monthly) of paid Phase 1 customer.

If customer takes Phase 0 to think:
- 5-business-day reminder + 2-line "ready when you are" email.
- 30-day reminder with quarterly update offering.

If customer declines Phase 1:
- Cordial follow-up.
- Quarterly update list ask (if engagement was substantively positive on the technical side).
- LAB-NOTEBOOK lessons-learned entry: what did we learn about this customer / architecture class / market segment that informs the next engagement?

If Phase 0 ended in honest walk:
- Customer keeps diagnostic. Sipsa keeps reputation for honest calibration.
- Note: don't burn the bridge; aerospace / defense pilots especially have multi-year horizons.

---

*This playbook updates per engagement learnings. Every Phase 0 closes with a 1-paragraph LAB-NOTEBOOK retrospective: what worked, what didn't, what's now in the playbook for next time.*
