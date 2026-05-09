# Patent Filing Packet — 2026-05-09 / 2026-05-10

**Two filings to complete this weekend. Total budget: $390 USPTO micro-entity fees. Total time: ~45 minutes.**

---

## FILING #1 — Five-provisional batch ($325)

**Status:** Drafts already prepared, no action needed beyond uploading. Per your CLAUDE.md task #147 family.

**Process:**
1. Open USPTO Patent Center: https://patentcenter.uspto.gov/
2. Sign in (your existing account)
3. Click "New Filing" → "Provisional"
4. For each of the 5 provisionals:
   - Title: (read from each draft's first line)
   - Inventor: Missipssa Ounnar
   - Assignee: Sipsa Labs, Inc.
   - Specification: upload the corresponding `provisional_*.pdf`
   - Pay micro-entity fee ($65 each = $325 total)
5. Save the filing receipt PDFs in `~/ultracompress/patents/filings/2026-05-09/`

**Drafts to upload** (from `~/ultracompress/docs/`):
- `provisional_*.pdf` files prepared earlier this week (from your YC application packet)
- If you don't see them, check `docs/AFWERX_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_07.md` for the full list

---

## FILING #2 — Per-Linear Adaptive BPW Continuation-in-Part ($65)

**⚠️ DEFER THIS FILING — see warning below.**

**Specification ready at:** `docs/PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md`

### CRITICAL UPDATE 2026-05-09 (pm-agent overnight analysis)

**The CIP draft is anchored to the per-Linear adaptive bpw v1 mechanism (k_proj@6bpw), which was REFUTED on end-PPL apples-to-apples.** Mechanism worked at the substrate level (k_proj quant_rel_l2 -55%) but produced no PPL improvement (1.005097 v1 vs 1.004876 uniform 5bpw, within noise σ≈0.0003).

Filing a method-form CIP claim on a refuted mechanism is technically valid but strategically weak — the dependent claims that would matter (the *quality* improvement) cannot be substantiated with data. Patent attorneys advise against filing claims you cannot back with empirical results.

**Diagnostic from pm-agent overnight (in `docs/RESEARCH_v3_CURE_DIRECTION_2026_05_09.md`):**
Deep layers (23-27 of Qwen3-1.7B-Base) train_loss_final does NOT converge even with the v2 800-1000 step ramp. Deep layers are RANK-bound, not STEPS-bound. v1 attacked the wrong knob (bpw); v2 attacked the wrong knob less wrongly (training time).

**v3 cure recommendation:** rank-redistribution at constant total V18-C parameter budget. Linear ramp rank from 16 (layer 0) to 48 (layer 27), sum held at 28×32=896. Predicted PPL ratio 1.0030-1.0035 (3σ above noise). ~30 LOC behind `UC_RANK_REDISTRIBUTE=1` flag.

**REVISED RECOMMENDATION:**
- ✅ FILE: 5-provisional batch ($325) — these are independent of v1/v2/v3
- ⏸ DEFER: per-Linear adaptive bpw CIP ($65) — re-anchor the spec on v3 (rank-redistribution) once v3 lands and shows the predicted PPL improvement. Filing then = $65 spent on a defensible method-form claim.
- 📅 NEW CIP filing target: 2-3 weeks out, after v3 confirms (or fails — in which case we save the $65)

If you absolutely want to file something patent-related this weekend, file ONLY the 5-prov batch. The CIP can wait.

---

### Original CIP filing process (KEEP FOR REFERENCE — do NOT execute this weekend)

**Process:**

### Step 1 — Convert the markdown spec to PDF (5 min)

Two options:

**Option A: pandoc (if installed)**
```powershell
cd C:\Users\scamd\ultracompress\docs
pandoc PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.md `
    -o PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.pdf `
    --pdf-engine=wkhtmltopdf
```

**Option B: Browser save-as-PDF**
1. Open the .md file in VS Code with markdown preview
2. Right-click → "Open Preview"
3. Print → Save as PDF
4. Save to `docs/PATENT_CIP_DRAFT_PER_LINEAR_ADAPTIVE_BPW_2026_05_08.pdf`

### Step 2 — File on USPTO Patent Center (15 min)

1. Open USPTO Patent Center: https://patentcenter.uspto.gov/
2. Sign in
3. Click "New Filing"
4. Select "Application Type": **Continuation-in-Part** (or "Provisional" if CIP option not visible — both will reference the parent number)
5. **Parent Application Number: 64/049,511**
6. **Title:** "Per-Linear Adaptive Bits-Per-Weight Allocation for Transformer Weight Compression"
7. **Inventor:** Missipssa Ounnar
8. **Assignee:** Sipsa Labs, Inc.
9. **Specification:** upload the PDF from Step 1
10. **Fees:**
    - Application filing fee: $65 (micro-entity provisional)
    - No claim fee for provisional
11. Pay via existing payment method
12. **Save the filing receipt** to `~/ultracompress/patents/filings/2026-05-09/cip_per_linear_adaptive_bpw_filing_receipt.pdf`

### Step 3 — Verify on Workbench (24-72 hr)

The new application should appear in your USPTO Patent Center workbench within 24-72 hours. Note the assigned application number — likely `64/049,XXX` (sequential after 511 and 517).

---

## ALSO DUE BY 2026-06-25 — $130 fee on parent

USPTO requires the $130 micro-entity surcharge-free filing fee on parent application 64/049,511 by **2026-06-25** (no surcharge) or 2026-09-25 (abandonment risk). Pay early — set a calendar reminder.

---

## TOTAL COSTS THIS WEEKEND

| Item | Cost | Status |
|------|------|--------|
| 5-provisional batch | $325 | Ready to file |
| Per-Linear adaptive bpw CIP | $65 | NEW — spec drafted, file 5/9 or 5/10 |
| Parent fee (deadline 6/25) | $130 | Pay anytime before 6/25 |
| **THIS WEEKEND TOTAL** | **$390** | |
| **THIS QUARTER TOTAL** | **$520** | |

---

## STRATEGIC NOTE

The CIP claim is **method-form** — it covers the residual-driven per-projection bpw allocation procedure. Today's v1 PPL eval did NOT beat the 1.0040 floor (V18-C was already saturating the correction), but the **mechanism is novel** and replicates empirically (k_proj quant_rel_l2 -55% on every layer, 11/11 sampled Hermes-405B layers confirm). The claim language doesn't depend on a specific PPL outcome.

If you'd rather wait until v2 (rank-redistribution policy) confirms the PPL gain before filing, that's defensible. But **filing now locks the priority date** for the mechanism, which is the bigger competitive lever — especially because peer labs (Anthropic, Together, Mistral) are likely working on similar adaptive-bpw ideas at this moment given the AWQ / QTIP precedents.

**Recommendation: file the CIP this weekend. $65 is cheap insurance for a $1B+ outcome.**
