# sipsalabs.com homepage refresh — draft for Sip's review (2026-05-10)

**Status:** AUDIT + DRAFT only. Live homepage NOT modified. Sip reviews and runs `vercel --prod --yes` after approval.

**Source pulled:** `https://sipsalabs.com/` at 2026-05-10 ~06:15 UTC into `C:\Users\scamd\AppData\Local\Temp\homepage_2026_05_10.html` (17,034 bytes).

---

## Audit summary

| Check | Status | Notes |
|---|---|---|
| PPL ratios match verified ground truth | PASS | All 6 ratios match (Phi-3-mini 1.00262x, Mixtral-8x7B 1.00368x, Qwen3-1.7B-Base 1.00401x, Qwen3-14B 1.00403x, Yi-1.5-9B 1.00414x, Qwen3-8B 1.00440x) |
| PyPI version is v0.5.5 | **STALE** | Homepage shows v0.5.4 in **7 places** — needs update to v0.5.5 |
| Architecture count = 22 verified end-to-end | PASS | "22 architectures validated end-to-end at 5 bpw" appears throughout |
| Hermes-3-405B framing = "compressed eval landed, baseline pending" | **STALE** | Currently says "compression complete · HF upload in flight" and "PPL ratio TBD (upload + eval in flight)" — eval is now running per tweet drafts, framing should reflect that |
| Patent notice link present | **MISSING** | Hero badge + footer say "Patent pending" but no link to a `/patents` or `/patent-notice` page |
| Sponsor / design partners section present | **MISSING** | Sections are hero, projects, about, contact, footer — no design-partners block |
| Brand assets v5.3 deployed | LIKELY OK | Both `/favicon.svg` (1,478 B) and `/og-image.png` (475,539 B) return 200 OK from the live origin; cache `Age: 0` on og-image suggests fresh deploy. No version stamp embedded in HTML to verify v5.3 specifically — flag for Sip if the assets need a hash or build-stamp comment |

**Deferred (out of scope for this audit):** sponsor/design-partners section and patent-notice page would be material additions, not edits — too large to draft inline without spec from Sip on copy + URL targets. Documented as gaps below; Sip decides whether to add as part of this refresh or schedule separately.

---

## Edit 1 — bump PyPI v0.5.4 → v0.5.5 (7 spots)

**Justification:** v0.5.5 ships tonight. Every reference to "0.5.4" or "v0.5.4" or "ultracompress==0.5.4" needs to become 0.5.5.

### Edit 1a — `<title>` description meta (line 7)

**OLD:**
```html
  <meta name="description" content="Sipsa Labs ships UltraCompress: mathematically lossless reconstruction of W_base for 5-bit transformer weights. v0.5.4 live on PyPI (adds `uc bench` for TTFT/TPS/peak-VRAM). 22 architectures validated end-to-end at 5 bpw — including Qwen3-14B at PPL ratio 1.00403x ...
```

**NEW (only the version token changes):**
```html
  <meta name="description" content="Sipsa Labs ships UltraCompress: mathematically lossless reconstruction of W_base for 5-bit transformer weights. v0.5.5 live on PyPI. 22 architectures validated end-to-end at 5 bpw — including Qwen3-14B at PPL ratio 1.00403x ...
```

(Remove "(adds `uc bench` for TTFT/TPS/peak-VRAM)" — that was the v0.5.4 release-note hook. v0.5.5 needs its own one-line hook from Sip — placeholder `<V055_RELEASE_NOTE>` for Sip to fill in.)

### Edit 1b — og:description (line 9)

**OLD:**
```html
  <meta property="og:description" content="UltraCompress v0.5.4 live on PyPI (adds `uc bench`). 22 architectures validated end-to-end at 5 bpw. Phi-3-mini 1.00262x · Qwen3-1.7B-Base 1.00401x · Qwen3-14B 1.00403x (essentially tied at 14B scale — scale-invariant codec) · Yi-1.5-9B 1.00414x · Qwen3-8B 1.00440x (8B record). Mixtral-8x7B 1.00368x (best MoE, live)." />
```

**NEW:**
```html
  <meta property="og:description" content="UltraCompress v0.5.5 live on PyPI. 22 architectures validated end-to-end at 5 bpw. Phi-3-mini 1.00262x · Qwen3-1.7B-Base 1.00401x · Qwen3-14B 1.00403x (scale-invariant codec) · Yi-1.5-9B 1.00414x · Qwen3-8B 1.00440x (8B record). Mixtral-8x7B 1.00368x (best MoE, live)." />
```

### Edit 1c — twitter:description (line 17)

**OLD:**
```html
  <meta name="twitter:description" content="UltraCompress v0.5.4 on PyPI (ships `uc bench`). 22 architectures validated at 5 bpw. NEW: Qwen3-14B 1.00403x (essentially tied with the small-decoder record at 14B scale — scale-invariant codec). Qwen3-8B 1.00440x. Mixtral-8x7B 1.00368x (best MoE, live)." />
```

**NEW:**
```html
  <meta name="twitter:description" content="UltraCompress v0.5.5 on PyPI. 22 architectures validated at 5 bpw. Qwen3-14B 1.00403x (scale-invariant codec). Qwen3-8B 1.00440x. Mixtral-8x7B 1.00368x (best MoE, live)." />
```

### Edit 1d — UltraCompress project tag (line 128)

**OLD:**
```html
            <span class="project-tag">Flagship · v0.5.4 · Lossless</span>
```

**NEW:**
```html
            <span class="project-tag">Flagship · v0.5.5 · Lossless</span>
```

### Edit 1e — UltraCompress project description (line 132)

**OLD (the inline reference):**
```
... <strong>v0.5.4 ships <code>uc bench &lt;packed_dir&gt;</code></strong> — measure TTFT, tokens/sec, decode-TPS, and peak VRAM on any UC-packed model. ... Customers reproduce locally: <code>pip install ultracompress==0.5.4</code>, <code>hf download ...
```

**NEW:**
```
... <strong>v0.5.5 live on PyPI</strong> — <V055_RELEASE_NOTE>. ... Customers reproduce locally: <code>pip install ultracompress==0.5.5</code>, <code>hf download ...
```

(Sip fills `<V055_RELEASE_NOTE>` with the v0.5.5 changelog headline — likely something concise about what shipped. If the release is purely Hermes-405B-related, "ships Hermes-3-405B reproducibility hooks" or similar.)

### Edit 1f — Status meta-item line 135

**OLD:**
```html
            <span class="meta-item"><span class="meta-key">Status</span> Live · <a href="https://pypi.org/project/ultracompress/0.5.4/" target="_blank" rel="noopener">pip install ultracompress==0.5.4</a> (adds <code>uc bench</code> — TTFT / TPS / peak-VRAM on any UC pack) · Two USPTO patents filed</span>
```

**NEW:**
```html
            <span class="meta-item"><span class="meta-key">Status</span> Live · <a href="https://pypi.org/project/ultracompress/0.5.5/" target="_blank" rel="noopener">pip install ultracompress==0.5.5</a> · Two USPTO patents filed</span>
```

### Edit 1g — Reproduce-a-record meta-item line 154

**OLD:**
```html
            <span class="meta-item"><span class="meta-key">Reproduce a record</span> <code>pip install ultracompress==0.5.4 huggingface_hub[cli]</code> · <code>hf download SipsaLabs/qwen3-14b-uc-v3-bpw5 --local-dir ./qwen3-14b-uc</code> · <code>uc bench ./qwen3-14b-uc</code> · <code>uc verify ./qwen3-14b-uc</code></span>
```

**NEW:**
```html
            <span class="meta-item"><span class="meta-key">Reproduce a record</span> <code>pip install ultracompress==0.5.5 huggingface_hub[cli]</code> · <code>hf download SipsaLabs/qwen3-14b-uc-v3-bpw5 --local-dir ./qwen3-14b-uc</code> · <code>uc bench ./qwen3-14b-uc</code> · <code>uc verify ./qwen3-14b-uc</code></span>
```

---

## Edit 2 — reframe Hermes-3-405B status (2 spots)

**Justification:** Per `HERMES_405B_BASELINE_RUN_STATUS_2026_05_09.md` and the tweet drafts, the Hermes-405B PPL eval (PID 34224) is currently running on cuda:1. The current homepage framing ("HF upload in flight" + "upload + eval in flight") is now ~9.5 hours stale. User specified the framing should be **"compressed eval landed, baseline pending"** — neither triumph nor missing piece.

### Edit 2a — UltraCompress project description (line 132)

**OLD (the Hermes-405B clause):**
```
Hermes-3-Llama-3.1-405B compression complete on a single 32 GB consumer GPU (126 layers / 250 GB v3 pack); HF upload in flight via the resilient uploader.
```

**NEW:**
```
Hermes-3-Llama-3.1-405B compressed end-to-end on a single 32 GB consumer GPU (126 layers / 250 GB v3 pack); compressed-PPL eval landed, baseline run pending.
```

### Edit 2b — Coverage meta-item (line 136)

**OLD:**
```
Hermes-3-405B (compression complete · HF upload in flight)
```

**NEW:**
```
Hermes-3-405B (compressed eval landed · baseline pending)
```

### Edit 2c — Records-table footnote (line 152)

**OLD:**
```
Phi-3.5-MoE PPL eval pending · Hermes-3-405B PPL ratio TBD (upload + eval in flight).
```

**NEW:**
```
Phi-3.5-MoE PPL eval pending · Hermes-3-405B compressed-PPL eval landed · baseline pending.
```

(Note: when the bf16 baseline lands Sun morning and the ratio is computable, this becomes a real records-table row — the stub for that row lives at `docs/HOMEPAGE_HERMES_ROW_STUB.md` per Task 4d.)

---

## Gaps flagged but NOT auto-drafted

### Gap 1 — Patent notice link

The hero badge ("Patent pending — USPTO Filed 2026-04") and footer ("Built with intent. Patent pending.") both reference patents but neither links anywhere. Per the user audit checklist, the homepage **should now have a patent notice link**.

**Recommendation:** Sip drafts a `/patents.html` page (or adds an anchor link to a patent-notice section on the homepage) with USPTO numbers 64/049,511 and 64/049,517 and the Apr-25 filing date. Then update line 75:

```html
        <span class="badge-label">Patent pending — USPTO Filed 2026-04</span>
```

→ wrap in `<a href="/patents">...</a>` once the page exists.

This is **out of scope** for tonight's automated draft because it requires copy decisions Sip should make (which numbers to disclose publicly, what jurisdictions to claim, what the "method details held privately" boundary actually is on the patent page). Flagged for Sip's morning queue.

### Gap 2 — Sponsor / design partners section

Homepage has no design-partners block. Per the user audit checklist, one **should now exist**. The natural location would be a new `<section id="partners">` between the projects section (currently line 191 closing) and the about section (line 193).

**Recommendation:** Sip drafts the partner copy before adding the section. Premature scaffolding without confirmed partners would either look like a placeholder ("Featured partners: TBD") which is worse than no section at all, or claim partners who haven't agreed to be named publicly. Flagged for Sip's morning queue.

### Gap 3 — v0.5.5 release-note hook

Two of the v0.5.4 → v0.5.5 edits above leave a `<V055_RELEASE_NOTE>` placeholder. Sip needs to write the one-line v0.5.5 changelog headline. If v0.5.5 is purely Hermes-3-405B-reproducibility-related, sample candidates:

- "ships Hermes-3-405B reproducibility hooks"
- "405B-class lossless artifacts on a single consumer GPU"
- "stable on-disk layout for 100B+ packs"

Sip picks based on what actually shipped in the v0.5.5 release.

---

## Deployment command (after Sip's review and edit-application)

```powershell
cd C:\Users\scamd\ultracompress\sipsalabs-site
vercel --prod --yes
```

(Replace `sipsalabs-site` with the actual website repo path if different. Sip knows the path; the audit didn't traverse the website source tree because the goal was content correctness against the live HTML, not file-tree mapping.)

**Pre-deploy verification:**
1. Open the modified `index.html` in a browser locally first.
2. Confirm 7 v0.5.4 → v0.5.5 swaps applied cleanly (no truncated descriptions).
3. Confirm 3 Hermes-405B status reframes applied cleanly.
4. Confirm `<V055_RELEASE_NOTE>` placeholder filled in (search for `<V055_RELEASE_NOTE>` and confirm zero matches).

**Post-deploy verification:**
1. `curl -s https://sipsalabs.com/ | grep -c "0.5.5"` should return ≥ 7.
2. `curl -s https://sipsalabs.com/ | grep -c "0.5.4"` should return 0.
3. `curl -s https://sipsalabs.com/ | grep "Hermes-3-405B"` should show the new "compressed eval landed · baseline pending" framing.

---

## ⚡ NEW (2026-05-10 night) — Mistral v10 1.0055× breakthrough

**Status**: VERIFIED tonight. PPL ratio = **1.0055×** (baseline 6.8910 → compressed 6.9287, n=50, FineWeb-edu held-out tail, seed=42). Method: hidden-MSE LOCAL per-layer training objective (recipe patent-protected).

**Public-facing impact**: Mistral-7B-v0.3 promotes from "1.05× best, cure refuted 4×" to **"verified sub-1% drift"**. Now joins the elite production-lossless tier alongside:

- Hermes-3-Llama-3.1-405B: 1.0066×  
- Qwen3-1.7B-Base: 1.00401×
- Qwen3-14B: 1.00403×
- Qwen3-8B: 1.00440×
- **Mistral-7B-v0.3: 1.00548× ← NEW**
- Phi-3-mini-4k-instruct: 1.00262× (seq_len=128 caveat)

### Edit 5 — bump Mistral row in homepage results table

If the homepage has a Mistral row (look for "Mistral-7B" or "1.05" in the live HTML), bump it from 1.05x to **1.0055×**. Add caveat: "verified 2026-05-10, n=50 FineWeb-edu seq_len=1024 seed=42".

### Edit 6 — meta description PPL claims

Update the `<meta name="description">` PPL list to include Mistral 1.0055×. Suggested wording (drop-in for OG card + meta):

> "UltraCompress v0.6.1 — 22 architectures validated at 5 bpw. Mistral-7B-v0.3 NEW: 1.00548× verified tonight, joins the sub-1% drift production tier with Hermes-3-405B (1.0066×), Qwen3-14B (1.00403×), Qwen3-8B (1.00440×)."

### Edit 7 — version bump (the original v0.5.4→v0.5.5 edit needs to skip to v0.6.1)

The original draft says v0.5.5, but we shipped v0.6.1 tonight. Update all PyPI version references:
- `v0.5.4` (OLD homepage) → `v0.6.1` (TARGET)
- `ultracompress==0.5.4` → `ultracompress==0.6.1`
- `pypi.org/project/ultracompress/0.5.4/` → `pypi.org/project/ultracompress/0.6.1/`

### Sip-action runbook (post-homepage-edit)

```bash
# 1. Pull latest live homepage
curl -s https://sipsalabs.com/ -o /c/Users/scamd/AppData/Local/Temp/sipsa_live.html

# 2. Apply edits 1-7 above (sed or text editor)
# 3. Re-deploy via Vercel CLI from the homepage source dir (likely a separate repo)
cd path/to/sipsalabs-www && vercel --prod --yes

# 4. Verify LIVE
curl -s https://sipsalabs.com/ | grep -E "1\.0055|Mistral|0\.6\.1"
```

Note: the homepage source repo location is not in `C:\Users\scamd\ultracompress\` — Sip knows where it lives.

