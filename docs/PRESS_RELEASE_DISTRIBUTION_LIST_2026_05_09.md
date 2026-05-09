# PRESS RELEASE DISTRIBUTION LIST — Hermes-3-Llama-3.1-405B announcement (2026-05-09)

**Companion to:** `PRESS_RELEASE_HERMES_405B_2026_05_09.md`
**Embargo:** 2026-05-09 09:00 PT (16:00 UTC)
**Use:** Sip selects from this list and sends the embargoed release manually. Do NOT auto-distribute.
**Outreach principle:** target reporters who actually own AI-infra, frontier-model inference, MLOps, or open-source ML beats. Avoid generalists who will rewrite the headline as a hype piece. Prefer reporters who have written specifically about quantization, model compression, on-device LLMs, or open-source releases in the last 12 months.

---

## How to use this list

1. **Sequence by tier, not by alphabet.** Tier 1 (specialist beat reporters) get the embargoed release first, ~24 hours before lift. Tier 2 (broader AI/MLOps coverage) gets it ~12 hours before. Tier 3 (industry trade press) gets it concurrent with the public lift at 09:00 PT.
2. **One reporter per outlet.** Do not double-pitch the same outlet — it reads as spray.
3. **Subject line for cold pitches:** `Embargoed until 9 AM PT Friday: 405B model compressed to a single 32 GB consumer GPU`.
4. **Body framing:** lead with the news (405B → one consumer GPU, three-command repro, open source on HF), not the company. Founder background goes in the second paragraph or a "background available on request" footer.
5. **Attach:** the embargoed press release as a `.txt` or paste inline. Do NOT attach patent numbers, internal benchmarks, or anything not in the public release.
6. **Track:** keep a separate sent-log (date, reporter, embargo confirmed Y/N, response). Do not put it in this file — this file is the master template only.

---

## TIER 1 — Specialist AI / infrastructure beat (priority pitches)

These reporters cover model compression, quantization, on-device LLMs, and open-source ML releases as a regular beat. They are the right venue for a technical-first story.

| # | Reporter | Outlet | Beat | Why relevant |
|---|---|---|---|---|
| 1 | **Will Knight** | WIRED | AI / frontier models / hardware | Long track record covering frontier-model infrastructure and the hardware-cost economics of inference; this story IS the hardware-economics story. |
| 2 | **Cade Metz** | The New York Times | AI industry / research | Covers consequential AI-industry technical milestones for a general-business audience; "frontier inference on consumer hardware" is exactly his lane. |
| 3 | **Karen Hao** | The Atlantic / freelance | AI policy, infrastructure, energy | Has written extensively about the datacenter and energy footprint of large models; consumer-GPU inference directly hits her thesis. |
| 4 | **Stephanie Palazzolo** | The Information | AI infrastructure / startups | Beat is enterprise AI infra; The Information's audience (PMs, founders, investors) cares about cost-displacement stories and reproducible benchmarks. |
| 5 | **Carl Franzen** | VentureBeat | AI / generative AI products | Covers open-source AI releases and infra tooling weekly; reliable for a fair, detail-forward write-up of an HF release with a repro flow. |
| 6 | **Sharon Goldman** | Fortune Eye on AI / freelance | AI industry, open source | Strong track record covering open-source-vs-closed-source AI tension and developer-tooling releases; the open-source angle fits her newsletter cadence. |
| 7 | **Emilia David** | The Verge | AI / consumer-facing AI | Covers the consumer-AI-tooling angle; "this runs on a gaming GPU" is a Verge-shaped framing. |

---

## TIER 2 — Broader AI / developer / open-source coverage

These reporters cover AI broadly and will pick up the story if Tier 1 confirms interest. Pitch ~12 hours before lift, after at least one Tier 1 has expressed interest.

| # | Reporter | Outlet | Beat | Why relevant |
|---|---|---|---|---|
| 8  | **Devin Coldewey** | TechCrunch | AI / startups / open source | Reliable for a clean write-up of an open-source release with reproducible numbers; reads model cards before publishing. |
| 9  | **Kyle Wiggers** | TechCrunch | AI research / model releases | Covers HF / model-release news regularly; will engage with a technical eval-config disclosure rather than just the headline. |
| 10 | **Tiernan Ray** | ZDNET | Enterprise AI / infrastructure | Long-form, detail-tolerant infra coverage for an enterprise audience; a good fit for the regulated-industry angle. |
| 11 | **Maria Deutscher** | SiliconANGLE | Enterprise infrastructure / AI | Covers enterprise AI infrastructure and OSS releases for a CTO/infra-architect audience; the on-prem / no-cloud-dependency angle reads natively to that audience. |
| 12 | **Ryan Daws** | AI News (artificialintelligence-news.com) | AI industry trade press | Trade-press fit; reliably covers open-source ML announcements with a reproducibility focus and links back to the HF model card. |

---

## TIER 3 — Concurrent-with-lift (no embargo advantage)

These outlets do not need an embargo head start. Send the public release at 09:00 PT lift.

- **The Register** — AI / infra desk (general). UK-skewed audience; The Register's writers tend to enjoy "consumer GPU eats datacenter" framings on technical merit.
- **Hacker News (Show HN)** — Sip submits the HF model card link himself, NOT a press-release link. The HN front page does not respond well to PR-style writing; it responds to "here is the artifact, here is the repro, here is what's interesting".
- **r/LocalLLaMA** — Sip posts the HF link with a short technical-first note. This is the highest-leverage developer audience for this specific release. Avoid PR voice entirely.

---

## DO NOT pitch

- **Generalist tech tabloids** that will rewrite the headline as "Veteran-founded startup beats Meta" — that framing is wrong on facts (this is a NousResearch fine-tune of an openly released base architecture, not a head-to-head with Meta) and damaging to credibility.
- **Any outlet that has previously published unsourced "world's first" claims about quantization** — they will overstate the result and force a correction cycle.
- **Crypto / Web3 publications** — wrong audience.
- **Any reporter who has previously broken an embargo** Sip is aware of — keep a personal blacklist; do not put names in this file.

---

## Boilerplate cold-email template (paste into mail client manually)

```
Subject: Embargoed until 9 AM PT Friday: 405B model compressed to a single 32 GB consumer GPU

Hi <FIRST NAME>,

Quick heads-up on a release going public Friday morning that fits your beat.

Sipsa Labs is releasing a 5-bit compressed pack of the 405-billion-parameter Hermes-3-Llama-3.1-405B model that reloads on a single 32 GB consumer GPU. Bit-exact reconstruction; reproducible by any researcher in three commands. Open-source runner on PyPI, model card on Hugging Face.

The full embargoed release is below. Embargo lifts 2026-05-09 09:00 PT.

Happy to set up a 15-minute call before lift if helpful — I can walk through the eval setup, the v3 pack format, or the regulated-industry use cases the founder is talking to.

Best,
<SIP>

—

[paste full press release body here]
```

---

## Post-lift follow-up (do not pre-write)

After 09:00 PT lift:
1. Wait 30 minutes. Do not re-pitch in that window.
2. Anyone who confirmed interest pre-lift gets a single follow-up at lift+1h with the live HF link and a one-line "if you want a repro session, I'm at <booking-link>".
3. Anyone who did not respond pre-lift does NOT get a follow-up. Move on.
4. Track final coverage in a separate file. Do not edit this template after send.

---

## Contact-info verification before send

Verify each reporter's current outlet and email address on the day of send. Reporters change beats and outlets often; this list reflects beats as of 2026-05-08 and may need updating. Use the outlet's masthead or the reporter's own bylined contact link as the source of truth — not aggregator databases.
