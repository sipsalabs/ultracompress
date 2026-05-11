# Phase 0 Customer Walkthrough — Compression-as-a-Service

**Use:** Sip's paste-ready operational runbook for the $5K, 1-week paid POC. Hand a prospect through this doc with name + model filled in. No custom writing required per engagement.
**Status:** PUBLIC DRAFT, v1 (2026-05-10).
**Pairs with:** `docs/INTERNAL_CAAS_CONTRACT_TEMPLATE_2026_05_10.md` (engagement letter), `docs/IAAS_PHASE_0_POC_SCOPE_2026_05.md` (SOW), `docs/CUSTOMER_ONBOARDING_v0.5.5_2026_05_09.md` (CLI install + repro).

---

## Honest disclaimer (read first)

- **Conversion.** Best-guess 60-70% Phase 0 → ongoing CaaS, based on similar self-serve devtool POC structures. No internal cohort yet — planning prior, not measured fact. Update at n>=5.
- **Fast NO is the goal.** Customer churn after Phase 0 is OK. Engineered for "decide in 5 days," not "we'll know it when we see it."
- **Bandwidth.** One Phase 0 per 2-week window. GPU eval runs can't overlap; Sip needs ~3 days recovery between engagements.
- **Recipe stays internal.** Per Selective Disclosure Charter — customer gets artifact + manifest + PPL number. NOT the codec, per-block recipe, or parameter selection logic, even at $5K.

---

## Pre-engagement (Day 0) — under 1 hour Sip-time

| Step | Action |
|---|---|
| 1 | Email confirmation + Stripe invoice link (50% upfront, $2,500). Sent from `founder@sipsalabs.com`. |
| 2 | Slack Connect channel created (default) — `sipsa-{customer-short-name}-phase0`. Discord fallback if customer prefers. |
| 3 | Pre-flight survey emailed (template below). All 8 questions in one message. |
| 4 | CaaS engagement letter sent for countersignature (`INTERNAL_CAAS_CONTRACT_TEMPLATE_2026_05_10.md`). |

### Pre-flight survey (paste-ready)

```email
Subject: Sipsa Phase 0 — 8 quick questions before we kick off

Hi {NAME},

Stripe invoice + engagement letter are in your inbox. Before Day 1 I need 8 short answers — reply inline:

1. Model architecture + checkpoint (HF id or "we'll ship private weights")?
2. Target deployment hardware — single GPU, multi-GPU, or cluster?
3. Eval suite you use today — custom harness, WikiText/c4, or "pick one for us"?
4. "Lossless enough" threshold — bit-identical, 1.005x PPL, 1.01x, 1.025x?
5. Data privacy — can weights leave your VPC, or do we need a no-share track?
6. Success criteria for the POC — a number, a binary pass/fail, or open-ended?
7. Timeline — hard 5-day window, or 10 days fine?
8. Decision-maker name + email for sign-off on the artifact?

Reply-all to this thread is fine. Slack channel for {MODEL} goes up once you confirm.

Sipsa Labs
```

---

## Pre-flight checklist (Sip's gate before saying yes)

- [ ] Model architecture is in `TARGET_SUBS` ((production trainer, patent-protected)). If not, propose a 5-line registry add OR scope as a separate Custom Architecture engagement.
- [ ] Customer ships **bf16** baseline weights — NOT pre-quantized INT4/INT8. Quantized inputs need conversion and break the lossless guarantee math.
- [ ] Stripe invoice paid before Day 1 starts. No NET-30 at the $5K tier.
- [ ] Success criteria is OBJECTIVE — a number ("under 1.01x PPL") or binary ("uc verify PASS"), not "we'll know it when we see it."
- [ ] Engagement letter countersigned (`INTERNAL_CAAS_CONTRACT_TEMPLATE_2026_05_10.md`).
- [ ] Sip has no other Phase 0 in flight or eval run on the same GPU within the 5-day window.

---

## Days 1-5 — Daily breakdown

| Day | Customer-facing | Sip-side | Risk + mitigation |
|---|---|---|---|
| **1 Kickoff + weights** | 30-min Slack Huddle/Zoom kickoff. Channel pinned with letter, SOW, Friday slot. | Receive HF private repo URL or S3 link. SHA-256 download. Architecture-ID vs `TARGET_SUBS`. Run fp16 baseline PPL. | Customer can't ship weights Day 1 → push start to next 2-week slot, don't burn GPU window. |
| **2 Compression run** | EOD note: "compression done, layer N done, eval running tomorrow." | Fire production compression pipeline. Watch peak VRAM + per-layer convergence. | Layer regresses → Day-2 retry with adapted hyperparameters, log either way. |
| **3 Eval + manifest** | Manifest + first PPL ratio ("{X.XXXX}x bf16 baseline on {eval}"). | Run PPL eval on customer's suite (or default WikiText/c4). Generate SHA-256 manifest. | PPL above pre-flight threshold → honest same-day call: re-tune or walk per letter §5.3. |
| **4 Customer validation** | Customer runs `uc verify` + `uc bench`. Sip Slack-available 4 hours. | Push pack to customer's secure transfer destination. Keep eval GPU on standby. | Hardware differs, `uc bench` numbers diverge → set expectation: only PPL is apples-to-apples. |
| **5 Wrap + decision** | 30-min Zoom readout. Customer decides convert vs walk. | Walk through manifest + PPL + per-layer profile. Wrap email with 3-tier menu. Trigger second 50% invoice. | Customer slow-rolls → wrap email sets 10-business-day decision deadline. |

### Email templates (paste-ready, ≤100 words each)

**Day 1 — Kickoff**

```email
Subject: Sipsa Phase 0 Day 1 — kickoff today, here's the plan

Hi {NAME},

Day 1 plan:
- Hour 1-2: pull {MODEL}, SHA-256 verify
- Hour 3-4: architecture ID + fp16 baseline on your eval
- EOD: status note with numbers

Slack: #sipsa-{NAME}-phase0 (pinned: letter, SOW, Friday Zoom). Slack > email for async.

Sipsa Labs
```

**Day 2 — EOD compression**

```email
Subject: Sipsa Day 2 — {MODEL} compression done, eval AM

Hi {NAME},

{MODEL} compression finished EOD. Per-layer FRR profile in Slack as JSON.

PPL eval starts tomorrow AM. First ratio + SHA-256 manifest by EOD Wednesday. No action your side.

Sipsa Labs
```

**Day 3 — Manifest + PPL**

```email
Subject: Sipsa Day 3 — {MODEL} PPL ratio {X.XXXX}x, manifest attached

Hi {NAME},

PPL eval done. {MODEL} measures {X.XXXX}x against bf16 baseline on {eval}. Pre-flight threshold: {Y}. {Pass / borderline / miss}.

SHA-256 manifest in Slack — end-to-end coverage.

Tomorrow you run `uc verify` + `uc bench`. Repro in `docs/CUSTOMER_ONBOARDING_v0.5.5_2026_05_09.md`. Sip Slack-available 1-5pm ET.

Sipsa Labs
```

**Day 4 — Validation prep**

```email
Subject: Sipsa Day 4 — {MODEL} pack ready for your `uc verify`

Hi {NAME},

Pack at {transfer-link}. Three commands:

    pip install ultracompress==0.5.5 huggingface_hub[cli]
    uc bench  ./{MODEL}-uc-v3-bpw5 --device cuda:0 --out delivered_bench.json
    uc verify ./{MODEL}-uc-v3-bpw5

Expected: `VERIFY: PASS`. TTFT/TPS/VRAM are hardware-dependent; only the PPL ratio is apples-to-apples.

Slack 1-5pm ET. Friday Zoom on calendar.

Sipsa Labs
```

**Day 5 — Wrap + 3-tier menu**

```email
Subject: Sipsa Phase 0 wrap — {MODEL} artifact + what's next

Hi {NAME},

Phase 0 wrap done. In your hands:
(a) {MODEL} UC pack (self-contained, SHA-256 verified)
(b) Manifest + PPL ratio ({X.XXXX}x)
(c) Per-layer compression profile

Three ways forward:
1. **Just the artifact** — you keep the pack, license per letter §3.1. Done.
2. **Ongoing CaaS** — re-compression of fine-tunes at 50%/re-run, 12-month window. Letter §2.3.
3. **On-prem deploy + integration** — Phase 1, $50-200K depending on stack. Separate SOW.

Decision window: 10 business days. Second 50% Stripe invoice today.

Sipsa Labs
```

---

## Post-engagement (Days 6-7)

### Customer reference / case study request

```email
Subject: Sipsa Phase 0 — reference permission?

Hi {NAME},

Quick ask: would {customer-company} let us add you to sipsalabs.com/customers with attribution? Two flavors:

(a) **Logo + 1-line quote** — you approve the quote, we publish.
(b) **Anonymized case study** — "leading {industry} infra team, {N}B model, {X.XXXX}x PPL, 5 days end-to-end." No company name.

Either is gold for inbound. Reply with (a) / (b) / no — all three answers are fine.

Sipsa Labs
```

### Internal capture (Sip's lab notebook, 5 questions)

After every Phase 0, Sip writes 5 answers into `docs/LAB-NOTEBOOK.md`:

1. Did the architecture transfer cleanly, or did it need tuning?
2. PPL ratio achieved vs customer's pre-flight threshold — within budget or over?
3. Customer-specific observation (eval quirk, deployment shape, weight format gotcha) worth feeding back into the codec?
4. Where did the 5-day clock slip, if anywhere? (Day-1 weight delay? Day-3 eval re-run? Day-5 calendar miss?)
5. Convert / walk / hung — and why?

---

## Tone constraints (apply to every customer email)

- Engineer-not-lawyer voice. Under 100 words per email. No "thrilled to partner with you," no "delighted to confirm."
- Communication preference: **Slack > Email > Zoom**. Zoom only for kickoff (Day 1) and wrap (Day 5).
- Charter compliance: results-only. No recipe, no per-block parameter logic, no codec internals — even to the paying customer.
- Founder-direct (Voice v6): Sip signs as "Sipsa Labs" or "the Sipsa Labs team," not personal name on outbound mail.

---

## INCOMPLETE — needs Sip input

- **Stripe invoice exact wording.** Doc assumes Sip's Stripe template exists; the literal email subject + line-item naming convention is not captured here.
- **Slack Connect channel naming convention.** Doc proposes `sipsa-{customer-short-name}-phase0` — confirm with whatever Sip already uses for live customers.
- **Secure transfer destination default.** Doc says "{transfer-link}" — Sip needs to lock whether default is HF private repo, S3 pre-signed URL, or SFTP, and bake into the Day 4 template.
- **Customer's eval suite handoff format.** Pre-flight Q3 asks "custom harness or standard" — operational wiring (do they ship a Python script? a YAML? a list of prompts?) is one Phase 0 away from being patternable.

---

*End of walkthrough. Pair with engagement letter (`INTERNAL_CAAS_CONTRACT_TEMPLATE_2026_05_10.md`) and SOW (`IAAS_PHASE_0_POC_SCOPE_2026_05.md`).*

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
