# Apps & Admin Sweep — Sip's Punch List (next 7 days)

**Generated:** 2026-05-08 evening MDT
**Window covered:** 2026-05-08 → 2026-05-15
**Format:** every item is a single checkbox with trigger / blocker / URL / paste-text / Sip-only flag.
**Time-on-Sip if zero blockers hit:** ~3.5 hours total spread across the week.

---

## Today / tomorrow (2026-05-08 → 2026-05-09): the things that ONLY Sip can do

- [ ] **HF avatar + banner drag-drop** — `Sip-only: YES` — trigger `now` — blocker `none` — URL https://huggingface.co/organizations/SipsaLabs/settings → upload `avatar_400.png` (org logo) and `hf_logo_512.png` if asked separately. Files: `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\avatar_400.png` + `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\hf_logo_512.png`. ~2 min.
- [ ] **X / Twitter avatar + banner drag-drop** — `Sip-only: YES` — trigger `now` — blocker `none` — URL https://x.com/settings/profile → upload `avatar_400.png` (profile picture) + `x_banner_1500x500.png` (header). Files: `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\avatar_400.png` + `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\x_banner_1500x500.png`. ~2 min.
- [ ] **LinkedIn (Sipsa Labs) avatar + banner drag-drop** — `Sip-only: YES` — trigger `now` — blocker `none` — URL https://www.linkedin.com/company/sipsalabs/admin/ → Edit page → upload `avatar_400.png` (logo) + `linkedin_banner_1128x191.png` (cover). Files: `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\avatar_400.png` + `C:\Users\scamd\AppData\Local\Temp\sipsa_brand\linkedin_banner_1128x191.png`. ~2 min.
- [ ] **Send 3 SSM cold-email drafts from `sipsalabs@gmail.com`** — `Sip-only: YES` — trigger `now` — blocker `verify recipients first` — URL https://mail.google.com (logged in as sipsalabs@gmail.com → Drafts). Drafts: Cartesia AI (Albert Gu), AI21 (Jamba team), Tri Dao's lab (Princeton). Verify each To: address resolves to a real person before clicking Send. Drafts source: `docs/LAUNCH_DRAFTS_2026_05_08.md` § 9. ~5 min.
- [ ] **Reddit r/LocalLLaMA Show & Tell post** — `Sip-only: YES` (Chrome MCP can't reach reddit.com) — trigger `now` — blocker `none` — URL https://www.reddit.com/r/LocalLLaMA/submit?type=TEXT — title (paste): `[Show] UltraCompress 0.5.1 — first mathematically lossless 5-bit transformer compression. Bit-identical reconstruction. 8 architectures live on HF (1.7B → 70B + 3 MoE). Mamba SSM compatible.` — body: copy verbatim from `docs/LAUNCH_DRAFTS_2026_05_08.md` § 7 (Markdown body block, ~35 lines). ~5 min.
- [ ] **HN follow-up reply window** — `Sip-only: YES` — trigger `if any HN commenter shows up` — blocker `none` — URL https://news.ycombinator.com/item?id=48065657 — currently 4 points / 0 comments. If the thread gets a question, reply within ~3 hours; the author has already posted the priming first comment. Refresh once tonight + once tomorrow morning, no need to camp.

## Filing window (2026-05-09, Saturday): USPTO supplements

- [ ] **USPTO 5-provisional batch filing (NOT just Track A supplement)** — `Sip-only: YES` — trigger `Sat 2026-05-09 09:00 MDT` — blocker `pre-flight checklist in MAY_9_FILING_RUNBOOK.md` — URL https://patentcenter.uspto.gov — fee `$325 total = 5 × $65 micro-entity` — files (convert markdown to PDF night before via pandoc):
  1. `docs/PATENT_DSR_Q_PROVISIONAL_DRAFT.md` (file FIRST — strongest novelty, standalone)
  2. `docs/PATENT_TRACK_A_SUPPLEMENT_v4.md` (cite parent `64/049,511`)
  3. `docs/PATENT_TRACK_B_V2_SUPPLEMENT_DRAFT.md` (cite parent `64/049,517`)
  4. `docs/PATENT_TRACK_C_PROVISIONAL_v2.md` (standalone)
  5. `docs/PATENT_UC_IR_LAYER2_PROVISIONAL.md` (standalone)
  Per-doc form-fill: Provisional Application for Patent / Inventor `Missipssa Ounnar` / Assignee `Sipsa Labs, Inc.` (Delaware) / Micro Entity / drawings NONE. Save 5 receipts to `~/filing_2026_05_09/receipts/`. Full runbook + risk register at `docs/MAY_9_FILING_RUNBOOK.md`. ~2 hours block.
  > **Correction to the original briefing:** the briefing said "Track A supplement only, $65". The runbook has expanded scope to a 5-provisional batch totaling $325. Confirm the additional 4 filings are still on for 2026-05-09 before proceeding (only Track A supp was originally promised in `IN_FLIGHT.md` → mismatch worth a 30-second sanity check).

## Daily watch (2026-05-09 → 2026-05-15): the EIN trigger and what cascades from it

- [ ] **Watch inbox for Atlas EIN email** — `Sip-only: YES` — trigger `inbox check 2× per day` — blocker `IRS issuance, day 1-7 of window (Atlas filed 2026-05-07)` — sender pattern: from `atlas@stripe.com` or `noreply@stripe.com` with subject containing "EIN" / "tax" / "Sipsa Labs". Forward immediately to `founder@sipsalabs.com` + `legal@sipsalabs.com` so it's archived in the corp record. **Critical:** legal name spelling is `Missipssa Ounnar` (M-I-S-S-I-P-S-S-A O-U-N-N-A-R, with R at end, NOT L). Cascading actions unlock the moment the EIN lands — see four items below.
- [ ] **(EIN-blocked) SAM.gov UEI registration** — `Sip-only: YES` — trigger `same day Atlas EIN arrives` — blocker `EIN required` — URL https://sam.gov/content/entity-registration — pre-fill: legal name `Sipsa Labs, Inc.`, EIN (from Atlas email), business address (Atlas registered-agent address). UEI issued same-day; full DLA validation (CAGE code) takes 3-4 weeks. Save confirmation email. ~20 min. **This unblocks NASA SBIR + AFWERX SBIR.**
- [ ] **(EIN-blocked) DSIP account** — `Sip-only: YES` — trigger `same day as SAM.gov` — blocker `EIN` — URL https://www.dodsbirsttr.mil/submissions/ → Register → Small Business. Pre-fill: `Sipsa Labs Inc.`, EIN, `founder@sipsalabs.com`. Active immediately. ~15 min.
- [ ] **(EIN-blocked) SBA Company Registry (SBC Control ID)** — `Sip-only: YES` — trigger `same day as SAM.gov` — blocker `EIN` — URL https://www.sbir.gov/registration — form-fill: `Sipsa Labs Inc.`, EIN, business address, ownership `Missipssa Ounnar 100%`. ~10 min.
- [ ] **(EIN-blocked) Mercury bank account application** — `Sip-only: YES` — trigger `same day Atlas EIN arrives` — blocker `EIN + Atlas Articles of Incorporation PDF + Atlas Bylaws PDF` — URL https://mercury.com/apply/business — fill from `docs/MERCURY_BANK_QUICK_FORM_2026_05_07.md`. Identity-bound bits: SSN, DOB, home address, driver's license image upload. Approval typically 1-3 business days. Save routing + account number when issued. ~30-45 min.

## (EIN-blocked) downstream filings, can prep but not submit

- [ ] **AFWERX SBIR Phase I submission** — `Sip-only: YES` — trigger `after SAM.gov UEI issued (same day) AND DSIP account live` — blocker `SAM.gov UEI + DSIP login` — URL https://www.dodsbirsttr.mil/submissions/ → AFWERX 25.x Open Topic SBIR Phase I — proposal source: `docs/AFWERX_SBIR_PHASE1_PROPOSAL_DRAFT_2026_05_07.md` (will be PDF-converted by operator before Sip submits). ~1 hr. Decision turnaround ~60 days.
- [ ] **NASA SBIR Phase I submission** — `Sip-only: YES` — trigger `after SAM.gov UEI` — blocker `SAM.gov UEI + NASA EHB account` — URL https://ehb6.gsfc.nasa.gov (NASA Electronic Handbook) — proposal source: `docs/NASA_SBIR_PHASE1_PROPOSAL_DRAFT.md` — Topic `ENABLE.2.S26B` (HPSC). Imminent deadline `2026-05-21 17:00 ET` is a longshot (only if SAM.gov fully clears in 14 days; realistic target is the next NASA appendix Q4 2026). ~1 hr.
- [ ] **USPTO patent assignment recordation** — `Sip-only: YES` — trigger `after Atlas EIN + signed assignment PDF from operator` — blocker `EIN + Sip's wet/DocuSign signature on assignment doc` — URL https://epas.uspto.gov → Recordation → Assignment. Inventor `Missipssa Ounnar` → Assignee `Sipsa Labs, Inc.`. Patents `64/049,511 + 64/049,517` (and the 5 to-be-filed on 2026-05-09 once application numbers come back). Cost free for first recordation. ~15 min.

## YC + status checks (no blockers, light touch)

- [ ] **YC S26 application status check** — `Sip-only: NO` (operator can refresh) — trigger `once mid-week` — blocker `none` — URL https://www.workatastartup.com/companies/sipsa-labs — currently "In Review" per memory file. No action unless an interview email lands. If interview gets scheduled, drop everything else and prep. ~30 sec to refresh.

## Verify-only sanity checks (5 min total — Sip can browse, no Sip-only login)

- [ ] **PyPI: confirm v0.5.1 is the only non-yanked release** — `Sip-only: NO` — trigger `now` — blocker `none` — URL https://pypi.org/project/ultracompress/#history — expect: v0.5.1 listed as current; v0.5.0 listed with yank reason "Broken import..."; no other live versions. `pyproject.toml` confirms `version = "0.5.1"`.
- [ ] **GitHub: confirm sipsalabs/ultracompress shows v0.5.1 tag + 3 latest commits today** — `Sip-only: NO` — trigger `now` — blocker `none` — URL https://github.com/sipsalabs/ultracompress — releases page https://github.com/sipsalabs/ultracompress/releases should show `v0.5.1`. Recent commits per `IN_FLIGHT.md`: `206d7f7`, `469c7c9`, `f96e550`.

---

## Quick-reference table — Sip's exact next moves by trigger

| When | What | Sip-only? | Time |
|---|---|---|---|
| Now (tonight 2026-05-08) | HF / X / LinkedIn drag-drops + 3 cold emails + Reddit post | YES | ~15 min |
| Sat 2026-05-09 09:00 | USPTO 5-provisional filing batch ($325) | YES | ~2 hr |
| Daily 2× | Inbox check for Atlas EIN | YES | ~30 sec |
| Day Atlas EIN lands | SAM.gov + DSIP + SBA + Mercury (4 forms) | YES | ~90 min |
| Day after Atlas EIN | AFWERX + NASA SBIR submissions | YES | ~2 hr |
| Day after EIN | USPTO assignment recordation | YES | ~15 min |
| Mid-week | YC dashboard status refresh | NO | ~30 sec |
| If HN comment lands | Reply on item 48065657 | YES | ~10 min/reply |

## Hard-do-not-forget rules

- Spelling on every legal/government form: `Missipssa Ounnar` (with R, not L) — verified against `~/.claude/...user_legal_name_and_email.md`.
- Public surfaces: still no personal info. Use `founder@sipsalabs.com` / `legal@sipsalabs.com` / `security@sipsalabs.com` — never `micipsa.ounner@gmail.com`.
- Pre-USPTO-filing-on-Saturday: do NOT push compression numbers / method specifics on Twitter / LinkedIn / HF / GitHub. Anti-checklist in `docs/MAY_9_FILING_RUNBOOK.md` § 1.7.
- USPTO parent fees `$130` ($65 × 2) due 2026-06-25 (no surcharge) or 2026-09-25 (abandonment) — pay from Mercury debit card once account is live (clean-books reason).
