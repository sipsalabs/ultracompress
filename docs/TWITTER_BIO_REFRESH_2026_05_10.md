# Twitter @SipsaLabs bio refresh — 2026-05-10

**Why this file exists:** Auto-update via Chrome MCP form-fill is blocked by the same anti-automation pattern that froze the LinkedIn auto-post earlier today (see `LINKEDIN_HERMES_405B_PASTE_READY_2026_05_10.md`). Manual paste takes ~30 seconds and avoids leaving the bio in a half-edited state on a public surface.

**To update (Sun 5/10):**
1. https://x.com/settings/profile
2. Replace the Bio field with the picked candidate below
3. Save
4. Done

---

## Picked candidate (148 chars / 160 max)

```
first 405B-class lossless 5-bit compression on a single 32GB consumer GPU. hermes-3-405b @ 1.0066x PPL. 22 archs verified. pip install ultracompress
```

**Why this one wins:**
- Leads with the defensible "first" claim, scoped exactly the way the LinkedIn post scopes it ("first 5-bit lossless 405B that reloads on a single 32 GB consumer GPU") — same hedge, same defensibility.
- Names the actual headline ratio (`1.0066x`) tied to the actual headline artifact (`hermes-3-405b`).
- "22 archs verified" anchors the breadth claim that backs up the depth claim.
- Closes with the install command — the surface's job is to get a `pip install` from a profile visit.

---

## Stale bio (currently live, replace)

```
Compression infrastructure for LLMs. First lossless 5-bit transformer compression. 8 archs on HF + Mamba SSM. pip install ultracompress
```

(8 archs is now 22, no 405B headline, no 1.0066x — material drift.)

---

## Other candidates considered

**Candidate B (163 chars — over 160 limit, rejected):**
```
lossless 5-bit transformer compression. hermes-3-llama-3.1-405b @ 1.0066x PPL on a single 32GB GPU. 22 architectures, SHA-256 verifiable. pip install ultracompress
```

**Candidate C (147 chars — viable backup if A reads too cluttered):**
```
bit-identical 5-bit compression for LLMs. 405B class @ 1.0066x PPL on one 32GB GPU. 22 architectures verified end-to-end. pip install ultracompress
```

---

## Verification source for the 1.0066x figure

`docs/_HERMES_405B_LAB_NOTEBOOK_ENTRY_FILLED.md` — full HYPOTHESIS→CONCLUSION lab notebook entry. Compressed PPL 5.069230 / baseline PPL 5.035783 / ratio 1.006642x. Streaming per-layer eval, n=50, seq_len=1024, FineWeb-edu, seed 42, cuda:1 RTX 5090. Tweet 8/n already LIVE on @SipsaLabs as of 2026-05-10 ~07:55 MDT carrying the same number — bio refresh just brings the profile into alignment with what the timeline already says.
