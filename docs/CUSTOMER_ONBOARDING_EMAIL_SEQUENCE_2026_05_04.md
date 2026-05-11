# Customer Onboarding Email Sequence — Post-Phase-0-Signature

**Use:** After a customer signs the Phase 0 SOW, fire these emails on the documented cadence. Reduces Sip's per-customer email time and ensures every engagement gets the same professional touch.
**Customization:** Replace [CUSTOMIZE] markers and adapt the salutation per customer. Don't paste-as-is.
**Tone:** Calibrated, factual, no fluff. Customer expects business-direct from Sipsa.

---

## EMAIL 1 — SOW execution acknowledgment (same-day after countersignature)

**Trigger:** Sipsa countersigns the Phase 0 SOW.

**Subject:** Sipsa Phase 0 — countersignature received, Day 0 prep next

```
Hi [Customer technical contact],

The Phase 0 SOW is countersigned and on file as of [date/time]. Engagement window: [Day 1 start
date] to [delivery target].

Today's Day 0 prep on the Sipsa side:
- Created `~/customer/[customer-short-name]/` workspace.
- Reserved compute (cuda:0 + cuda:1 dedicated for the engagement week).
- Opened LAB-NOTEBOOK section for engagement timeline.

What I need from you by Day 1 ([date]):
1. The model checkpoint (or model class identifier if pulling from public HF).
2. The evaluation suite specification (input prompts + evaluation function + tolerance specification).
3. Designated technical contact for daily status emails.

If anything in the above is unclear, let's get a 15-min call on the calendar before Day 1 kickoff.
Otherwise the kickoff call is scheduled per the SOW for [time/date].

Invoice for the 50% kickoff payment will land in your inbox separately within 24 hours.

Best,
Missipssa Ounnar
Founder, Sipsa Labs
founder@sipsalabs.com
```

---

## EMAIL 2 — Day 1 kickoff (morning of Day 1)

**Trigger:** Day 1 of the engagement window.

**Subject:** Sipsa Phase 0 Day 1 — intake and baseline reproduction starting today

```
Hi [Customer technical contact],

Day 1 of the engagement window kicks off this morning. Plan for today:

- Hour 1-2: receive your slice via the secure transfer channel; SHA256 integrity check against the
  hash you emailed separately.
- Hour 3-4: architecture identification (we'll send a 1-paragraph note on what we found before
  proceeding).
- Hour 5-6: reference baseline reproduction at full precision against your evaluation suite.
- Hour 7-8: scalar BPW=5 quantization probe (no correction overlay yet) to gauge architecture
  receptivity.

End of day you'll get the Day 1 status email per the SOW. If integrity check fails or baseline
doesn't reproduce within your stated tolerance, we'll pause and call you before proceeding —
we won't compress against a baseline we couldn't reproduce.

Let me know if the secure transfer channel needs a credential rotation on your side; otherwise
we'll pull from the location specified in the SOW.

Best,
Missipssa
```

---

## EMAIL 3 — Day 1 end-of-day status (evening of Day 1)

**Trigger:** End of Day 1.

**Subject:** [Customer name] Phase 0 — Day 1 status

```
Hi [Customer technical contact],

End of Day 1 status:

- Slice received and integrity-verified ([SHA256 prefix]). Architecture identified as
  [class — e.g., "transformer LLM with multi-head attention" or "FNO with spectral convolution"].
- Reference baseline reproduced within [tolerance budget]. Reference predictions saved.
- Scalar BPW=5 receptivity probe shows [X]% relative L2 vs baseline. Decision rule:
  - [<5% degradation: receptive, correction overlay will likely transfer cleanly. Proceeding
    with correction overlay overlay tomorrow.]
  - [5-20% degradation: receptive but needs correction overlay (expected case). Proceeding.]
  - [>20% degradation: architecture-specific issue. Day 2 will diagnose; may surface
    discussion before continuing.]

On track to begin correction-overlay distillation tomorrow morning. Day 2 status email
end-of-day tomorrow.

Best,
Missipssa
```

---

## EMAIL 4 — Mid-engagement check-in (Day 3 or Day 5 depending on engagement length)

**Trigger:** Mid-point of the engagement window.

**Subject:** [Customer name] Phase 0 — interim findings

```
Hi [Customer technical contact],

Interim status at the midpoint of the engagement:

What's working:
- [Specific result. E.g., "Compression at BPW 5 + low-rank correction transfers to your
  architecture with [X]% relative L2 on the primary metric."]
- [Specific result. E.g., "Per-layer scaling profile shows [N] layers compress cleanly,
  [M] layers need attention."]

What's not working (or has open questions):
- [Specific issue. E.g., "Conservation residual on layer [X] exceeds tolerance by [Y]%; we're
  testing per-layer fp32 pinning as mitigation today."]
- [Specific issue. E.g., "Tail-token quality on rare-token cases is [X]% below the average;
  documenting whether this is a regression or an architecture-specific artifact."]

Phase 1 path candidates:
- [Branch A — existing pipeline applies cleanly: Tier 1 Phase 1 scope at $50-70K]
- [Branch B — needs per-customer tuning: Tier 2 Phase 1 scope at $65-80K]
- [Branch C — hybrid mechanism stack required: Tier 3 Phase 1 scope at $80-100K]
- [Or: honest decline-with-diagnostic if mechanisms don't transfer]

Let me know if you want a 30-min call to walk through interim findings before the deliverable
is finalized.

Best,
Missipssa
```

---

## EMAIL 5 — Final report draft delivery (penultimate day)

**Trigger:** Final report draft is ready for customer review.

**Subject:** [Customer name] Phase 0 — feasibility report draft for your review

```
Hi [Customer technical contact],

The Phase 0 feasibility report draft is attached for your fact-check. Sections:

1. Setup summary: model class, parameters, layers, hidden dim, attention layout, BPW target,
   calibration set.
2. Reference baseline reproduction: your stated baseline reproduced on Sipsa hardware against
   your evaluation suite.
3. Compression measurement table: per-metric scores, per-layer profile, per-bpw Pareto.
4. Comparison vs current production (if applicable): side-by-side at the same bit-rate.
5. Edge case audit: [conservation laws / tail-token quality / throughput], depending on your
   use case.
6. Recommendation: [Tier 1-3 Phase 1 OR honest decline with diagnostic].

The compressed checkpoint is ready for shipping; we'll send it via the same secure transfer
channel after you're satisfied with the report.

48-hour review window; flag any factual errors or framing concerns. Substantive Phase 1 scope
discussions can happen on a 30-min call once you've reviewed.

Invoice for the second 50% payment will go out after final delivery (per the SOW).

Best,
Missipssa
```

---

## EMAIL 6 — Final delivery (last day)

**Trigger:** Final report is finalized + compressed checkpoint is ready to ship.

**Subject:** [Customer name] Phase 0 — final deliverable + Phase 1 next steps

```
Hi [Customer technical contact],

Phase 0 deliverable is shipped:

- Final feasibility report (PDF): attached / [transfer link].
- Compressed checkpoint at recommended config: shipped via [secure transfer channel].
- Reproducibility receipt: attached / [transfer link].
- Phase 1 scope sketch (if positive recommendation): attached / [transfer link].

Phase 1 conversation: per the report's recommendation, we're proposing [Tier X at $YK / honest
decline]. Calendar invite for a 30-min readout call with your technical + business team:
[3 calendar slots within next 7 days].

Second 50% invoice goes out today.

If Phase 0 ends here without Phase 1: thanks for the engagement; the diagnostic stays valid for
your team's internal use indefinitely; we're available for follow-up questions for 90 days
post-delivery.

Best,
Missipssa
```

---

## EMAIL 7 — Post-engagement follow-up (T+30 days)

**Trigger:** 30 days after Phase 0 final delivery.

**Subject:** [Customer name] Phase 0 — 30-day follow-up

```
Hi [Customer technical contact],

Following up 30 days after Phase 0 delivery to check in:

- Have you had a chance to deploy the compressed checkpoint to your environment?
- Any production behavior surprises or follow-up questions on the diagnostic?
- Phase 1 conversation: still on the table, or has the timing shifted?

If Phase 1 timing is unclear: no pressure, the diagnostic is yours regardless. I just want
to be available if questions surface.

If Phase 1 is moving forward: let's schedule the next call at your team's pace.

If Phase 1 is no-longer-relevant: also fine; happy to put you on a quarterly Sipsa update list
so we stay in light touch as our roadmap evolves.

Best,
Missipssa
```

---

## EMAIL 8 — Quarterly update (Q+90 days, if customer agreed)

**Trigger:** ~90 days after engagement, or quarterly thereafter.

**Subject:** Sipsa Labs Q[X] update — what shipped this quarter

```
Hi [Customer technical contact],

Quick quarterly update on Sipsa Labs progress, since you indicated interest:

What shipped this quarter:
- [Concrete shipping milestone. E.g., "v0.5 release with production CUDA kernels for vLLM
  integration."]
- [Concrete metric. E.g., "Two additional aerospace customers in Phase 1; one IaaS provider
  in production deployment."]
- [Patent / research. E.g., "Track A v3 supplement filed; logit-KL distillation NeurIPS
  paper accepted."]

What's coming next quarter:
- [Concrete next milestone.]

Anything specific to your environment that we'd want to share early-access? Reply if so.
Otherwise, see you next quarter.

Best,
Missipssa
```

---

## SCENARIO: Customer signed Phase 0 then went silent (no response to Email 4)

**Trigger:** Mid-engagement check-in (Email 4) sent, no response within 5 business days.

**Subject:** [Customer name] Phase 0 — proceeding with deliverable, please confirm if anything blocks

```
Hi [Customer technical contact],

Following up on the interim status email from [date]. We're proceeding with the final-week
deliverable per the SOW timeline. Final report draft target: [date].

If anything from your side has changed (priorities shifted, eval suite needs adjustment,
business situation different), please flag now so we can adapt the deliverable. Otherwise we'll
deliver per the original scope.

Best,
Missipssa
```

---

## SCENARIO: Customer wants to extend the engagement scope mid-Phase-0

**Trigger:** Customer asks "can you also compress model B" or "can you eval against eval suite C" mid-engagement.

**Subject:** [Customer name] Phase 0 — scope extension request

```
Hi [Customer technical contact],

Got your note on extending the engagement scope. Here's how to think about this:

Option A — finish current Phase 0 first, then a separate Phase 0 for the additional model/eval:
- Current engagement delivers per the SOW timeline.
- Additional engagement scoped as a separate $5K-$15K Phase 0 with its own SOW + 50% kickoff.
- Cleaner contractually; predictable timeline for both.

Option B — extend the current engagement now:
- Updated SOW addendum signed before any additional work begins.
- Additional fee: $[X] depending on incremental scope.
- Timeline shifts by [Y] days.
- Risk: scope creep; stay disciplined about deliverable boundaries.

I'd recommend Option A unless there's a strong reason to combine. Let me know your preference;
either path is fine.

Best,
Missipssa
```

---

## SCENARIO: Customer wants to escalate the engagement to a Phase 1 negotiation pre-Phase-0 completion

**Trigger:** Customer is impressed mid-engagement and wants to start Phase 1 conversations early.

**Subject:** [Customer name] Phase 0 — Phase 1 conversations

```
Hi [Customer technical contact],

Great signal. Two paths from here:

Path A — finish Phase 0 first, Phase 1 SOW execution post-final-report:
- Phase 0 final deliverable on track per the SOW timeline.
- Phase 1 SOW execution within 14 days post-Phase-0-final-report.
- Standard structure; cleanest contractually.

Path B — Phase 1 SOW negotiation in parallel with Phase 0 final-week:
- Phase 1 SOW draft circulating during Phase 0 final-week.
- Phase 1 execution can start within 3-7 days of Phase 0 deliverable.
- Faster but requires Phase 1 scope to be confidently locked before Phase 0 measures finalize.

I'd recommend Path A for the first engagement to keep the contractual structure clean. Path B is
a fit if your team needs Phase 1 to start on a tight timeline (e.g., quarterly planning cycle).

Either way, I'd like to set up a 60-min call with your technical and business decision-makers to
walk through the Phase 1 scope sketch and pricing tier discussion. Calendar slots: [3 within
next 7 days].

Best,
Missipssa
```

---

## SCENARIO: Customer wants to back out mid-engagement

**Trigger:** Customer requests engagement termination before final delivery.

**Subject:** [Customer name] Phase 0 — engagement modification request

```
Hi [Customer technical contact],

Got your note. Per the SOW, the Phase 0 engagement is fixed-fee with payment terms (50% kickoff
+ 50% on delivery). Termination structure depends on the reason:

- If the modification is "we want to pause and resume in Q[X]": no penalty; the engagement enters
  hold-state. Sipsa-side artifacts retained per the SOW; resumption requires a brief addendum
  acknowledging the pause window.
- If the modification is "we want to terminate without delivery": the kickoff payment is
  non-refundable (covers the work done to date). No second 50% invoiced.
- If the modification is "the engagement is failing technically and we want to invoke the refund
  clause": let's get on a 30-min call to walk through the SOW refund language. The 50% refund
  clause applies if compression doesn't meet your stated tolerance specification by more than
  50% of the tolerance budget.

Whatever the path, I want to make this clean and respect both parties' time. Let me know which
path fits your situation; we'll execute the corresponding paperwork same-day.

Best,
Missipssa
```

---

## OPS DISCIPLINE

For every customer engagement:
- File all emails (sent + received) in `~/customer/[name]/email/`.
- Update `OPS_DASHBOARD.md` with engagement state changes (NDA executed → SOW signed → Day 1 kickoff → mid-engagement → final delivery → Phase 1 negotiation OR closed).
- Save engagement-end retrospective to `~/customer/[name]/retrospective.md` (1 paragraph: what worked, what didn't, what's in the playbook for next time).
- Update LAB-NOTEBOOK with engagement-relevant technical learnings.
- Reply within 1 business day to ALL customer email; flag if longer wait is needed.

---

## ANTI-PATTERNS

- Don't send Email 1 through Email 6 in the same hour. The cadence matters; rushed emails feel disrespectful.
- Don't skip Email 3 (Day 1 status). Setting the engagement-comm tone on Day 1 is high-leverage.
- Don't customize Email 7 (30-day follow-up) extensively. Keep it short; the value is the touch, not the content.
- Don't reuse exact wording across customers. Customers compare notes; identical templates feel impersonal.
- Don't reply to Day 4 customer questions after midnight. Save for morning unless customer's timezone makes it actually-morning for them.
- Don't promise follow-up actions in email without delivering. "I'll send X tomorrow" requires sending X tomorrow.
- Don't include personal details, opinions on competitors, or speculation about customer's internal politics. Stay business-direct.

---

*This sequence updates per actual engagement learnings. Append a "what I learned" section after each Phase 0 closes.*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
