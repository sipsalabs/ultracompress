# sipsalabs.com — /pricing + /status deploy walkthrough (v2.5)

**Date:** 2026-05-10
**Status:** PUBLIC-bound, INTERNAL deploy doc. Local files written; no `vercel` push yet.
**Target Sip workflow:** one-click `vercel --prod --yes` after eyeball pass.
**Selective-disclosure class:** INTERNAL (this doc never ships publicly).

---

## What was built (and where)

| File | Lines | Purpose |
|---|---|---|
| `C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site\pricing.html` | 798 | Five-tier pricing + 2 service blocks + Audit "Coming Q3 2026" + lossless demo + license + FAQ. Self-contained, inline CSS, system fonts only. |
| `C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site\status.html` | 447 | Four-surface infrastructure status (api / HF / PyPI / GitHub) + 30-day incident block + founder direct line + email subscribe placeholder. |
| `C:\Users\scamd\ultracompress\docs\PUBLIC_DRAFT_PRICING_STATUS_DEPLOY_NOTES_2026_05_10.md` | this file | Sip's deploy walkthrough (you are reading it). |

Both HTML files use `cleanUrls: true` from existing `vercel.json` — so they will live at `/pricing` and `/status` (no trailing `.html`).

---

## Pre-deploy verification checklist

Run these eyeball checks **before** typing `vercel --prod --yes`. Each one is ~10 seconds.

### 1. URLs return 200 locally

```powershell
# from C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site
# spin up a tiny static server and walk the URL surface
python -m http.server 8000
# open in browser:
#   http://localhost:8000/pricing.html
#   http://localhost:8000/status.html
```

Expected: both pages load, no console errors, fonts fall back cleanly to system stack (no FOUT spinner).

### 2. Tier prices match pricing v2 spec

Pull up `docs/PUBLIC_DRAFT_PRICING_PAGE_v2_5tier_2026_05_10.md` side-by-side with rendered `/pricing` and confirm the 5-tier table matches:

| Spec value | Page renders | OK? |
|---|---|---|
| Verifier | $0 | ☐ |
| Self-Serve API | From $0.10 / M tok in | ☐ |
| Pro Inference | From $499 / month | ☐ |
| Enterprise SaaS | From $50,000 / year | ☐ |
| On-Prem Deploy | From $250,000 / year | ☐ |

Do the same drill on the per-model self-serve table (Phi-3.5-MoE / Qwen3-8B / Qwen3-14B / Mixtral-8x7B / Hermes-3-405B). Every PPL ratio cited (`1.0044×`, `1.0040×`, `1.0037×`) traces to `BENCHMARKS_2026_05_10.json` `verified_records[]` — already cross-checked while building.

### 3. Stripe wiring path notes

`/pricing` does **not** post anywhere or render Stripe Checkout directly. The "Get a key — $5 free credits" CTA points to `/inference/signup` (already live, already wired to Stripe per `docs/INTERNAL_STRIPE_SELFSERVE_TURNKEY_2026_05_10.md`). The "Notify me" form on the Audit block uses `mailto:` — see "Intentionally left out" below.

### 4. Mobile responsive eyeball

Resize browser to ~360px wide. Confirm:
- 5-tier table on `/pricing` collapses to per-row cards (each row has its own labeled cells)
- Self-serve per-model table also collapses to readable cards
- `/status` service rows stack vertically with status pill below the row
- Footers reflow to single column
- Nav links wrap without overflow

### 5. Accessibility quick pass

- All `<a>` open-in-new-tab links carry `rel="noopener"`
- Form inputs have `<label>` (visually hidden via `position: absolute; left: -9999px;` — screen readers still read it)
- Status dots are `aria-hidden="true"`, with text status carrying the meaning
- Tables have `<caption>`-equivalent `aria-label`
- Color contrast: cyan (`#22d3ee`) and green (`#4ade80`) on black both pass WCAG AA at body size

---

## Deploy commands

```powershell
# from C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site
vercel --prod --yes
```

Vercel will detect no build step (matches existing `buildCommand: null` in `vercel.json`), upload the static assets, and promote to production. Typical wall time: 30-60 seconds.

---

## Post-deploy URL check

Run these two `curl` commands within 60 seconds of the Vercel deploy completing:

```powershell
# expect HTTP/2 200 on both
curl -I https://sipsalabs.com/pricing
curl -I https://sipsalabs.com/status
```

If either returns 404 or 308 → see "Vercel routing gotcha" below.

Then eyeball:
- https://sipsalabs.com/pricing — does the 5-tier table render? Hermes row visually emphasized?
- https://sipsalabs.com/status — does the green "All systems operational" pill animate?
- Click around: `/pricing → /status` link in footer works? `/status → /pricing` link works?

---

## Rollback command

If anything looks wrong on production (broken link, mis-rendered table, leaked internal copy):

```powershell
vercel rollback
```

Vercel CLI will prompt to select the previous production deployment. Pick the one immediately before your `--prod --yes` push. Rollback completes in ~10 seconds. Old URLs return immediately.

For a more surgical rollback (revert only one file rather than the whole deployment), `git revert` the commit on the `sipsalabs-site` repo and re-deploy — but for two new files like this, full `vercel rollback` is cleaner.

---

## Vercel routing gotcha (read before deploying)

`vercel.json` has `cleanUrls: true` — that means:
- `/pricing.html` is served at `https://sipsalabs.com/pricing` (no `.html`).
- `/status.html` is served at `https://sipsalabs.com/status`.
- Hitting `https://sipsalabs.com/pricing.html` directly will 308-redirect to `/pricing`. Same for status. (This is the intended behavior.)

The two new files do **not** need to live in `/pricing/index.html` or `/status/index.html` subdirectories — `cleanUrls` handles the URL rewriting at the edge. Putting them in subdirs would also work but is one more level of nesting for no gain.

If the post-deploy `curl` returns 404, check that `vercel ls` shows the two files in the deployment artifact list. If not, re-deploy with `vercel --prod --yes --force` to bust the build cache.

---

## What this deploy intentionally LEAVES OUT (defer to v2.1 / Phase 2)

**The "Notify me" form on the Audit block and the "Subscribe" form on /status are NOT wired to a real backend.** Both use `mailto:` action attributes — submissions will pop the user's mail client with a prefilled email to `founder@sipsalabs.com`. This is intentional for v2:

- v2 (this deploy) = static demand-capture. Sip processes inbound manually, ~5/week is fine.
- v2.1 (next deploy) = wire to Resend audience (or BetterStack status page) once we see >20 signups/week.

The HTML hooks are already in place: each form has a stable `aria-label` and the submit buttons use semantic `<button type="submit">`, so v2.1 only needs to flip the `action=` URL and add a `fetch()` POST handler. No re-template.

A separate v0.2 of `/status` will replace the four hardcoded green dots with live HTTP probes (the data-attributes `data-check="api"`, `data-check="hf"`, `data-check="pypi"`, `data-check="github"` are already in the HTML for the v0.2 script to find — no re-template needed). For now the dots are honest-but-static: green because all four surfaces are operational at deploy time.

---

## Commit message (suggested)

Recommend Sip use this exact message when committing the two HTML files to the `sipsalabs-site` repo:

```
add /pricing and /status pages

- /pricing: five-tier table (Verifier $0 → On-Prem $250K/yr) +
  Compression-as-a-Service + Custom Architecture + Audit "Coming Q3"
  tier anchor with Notify me form. FAQ in <details> blocks. Mobile-
  responsive collapse to cards <820px. Inline CSS, system fonts.
- /status: four-surface status (api / HF / PyPI / GitHub) + 30-day
  incident block + founder direct-line + Subscribe placeholder. v0.1
  hand-published; data-check hooks in place for v0.2 live probes.

Both pages match the /inference visual system exactly. No build step,
no external CDN, no JS dependencies. cleanUrls handles routing.

Forms are mailto: placeholders for v2 — Resend audience wiring in
Phase 2 once weekly volume justifies it.
```

(Single-author commit per the no-Co-Author discipline.)

---

## Charter compliance check (ULTRA-REVIEW RULE)

Both HTML files were audited against the charter constraint set:

- ☑ Zero recipe values exposed (no rank, lr, train_steps, seed, K=64, correction overlay, scalar quantization, k-cluster recovery, T1, calibration set, batch size).
- ☑ Zero internal Track A-G nomenclature on the public surface.
- ☑ Zero personal info (no `micipsa.ounner@gmail.com`, no `Missipssa Ounnar`, no street address, no DOB, no phone). Only `@sipsalabs.com` aliases.
- ☑ Every cited PPL number traces to `BENCHMARKS_2026_05_10.json`:
   - `1.0044×` (Qwen3-8B) → `verified_records[5].ppl_ratio = 1.0044`
   - `1.0040×` (Qwen3-14B) → `verified_records[3].ppl_ratio = 1.00403`
   - `1.0037×` (Mixtral-8x7B) → `verified_records[1].ppl_ratio = 1.00368`
   - `1.0026×` (Phi-3-mini) → `verified_records[0].ppl_ratio = 1.00262`
   - `1.0125×` (Llama-3.1-8B upper bound on the FAQ) → `verified_records[13].ppl_ratio = 1.0125`
- ☑ All comparison percentages are public list prices, dated to May 2026 in the footnote.
- ☑ "−44% vs Together" on Hermes-3-405B traces to public Together pricing card vs. our $2.50/M output rate.
- ☑ License framing matches existing /inference page exactly (BUSL on master, Apache on `legacy/0.5.x`).
- ☑ Patent numbers match USPTO public record (64/049,511 + 64/049,517 filed April 2026).
- ☑ "Coming Q3 2026" label is the only forward-looking claim on the Audit tier; no SLA language attached.
- ☑ Solo-founder direct-line on /status sets honest expectation ("within an hour during US business hours, four hours otherwise") — does not promise 24/7 pager coverage we don't have.
- ☑ /status incident block reads "No incidents in the last 30 days" — true at deploy time, not a forward claim.

No deploy step taken. Files are local-only until Sip runs `vercel --prod --yes`.

---

## Single-page deploy script (optional, copy-pasteable)

If Sip wants the whole flow as one PowerShell block:

```powershell
# 1. eyeball check
cd C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site
Start-Process "http://localhost:8000/pricing.html"
Start-Process "http://localhost:8000/status.html"
python -m http.server 8000
# Ctrl-C when done eyeballing

# 2. deploy
vercel --prod --yes

# 3. post-deploy verify
Start-Sleep -Seconds 30  # let the deploy propagate
curl -I https://sipsalabs.com/pricing
curl -I https://sipsalabs.com/status
Start-Process "https://sipsalabs.com/pricing"
Start-Process "https://sipsalabs.com/status"

# 4. rollback if needed
# vercel rollback
```

End of doc.

Codec internals + training procedure are patent-protected (USPTO 64/049,511 + 64/049,517).
