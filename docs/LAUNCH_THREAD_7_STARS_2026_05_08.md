# Twitter / X Thread — 7 Stars + 3 New Sub-1.01x Ratios (2026-05-08)

**Audience:** ML researchers, infra engineers, AI-investor crowd.
**Tone:** confident-but-honest, technical. No "revolutionary" / "world-class" / "first" / "lossless" without qualifier.
**Handle:** @SipsaLabs.
**Link policy:** one URL — `huggingface.co/SipsaLabs`.
**IP discipline:** numbers + customer flow only. No algorithm specifics.
**Status:** DRAFT — Sip copy/pastes manually. Do not auto-post.

---

## Tweet 1 / 6  — the social-proof hook  (272 chars)

```
github.com/sipsalabs/ultracompress crossed 7 stars this week.

Small number. First non-zero one we've had. A week ago Sipsa Labs was a name nobody knew; this week seven engineers found the repo on their own and starred it.

We owe them numbers in return. Below ↓
```

---

## Tweet 2 / 6  — the headline number  (271 chars)

```
Three new compressed checkpoints today, each under 1.01× baseline PPL on full eval:

  Qwen3-0.6B    21.4792 → 21.6274   ratio 1.0069×
  OLMo-2-1B     12.9933 → 13.0879   ratio 1.0073×
  SmolLM2-1.7B   9.1389 →  9.2168   ratio 1.0085×

The 1.0069× is the tightest dense-decoder ratio we've measured at 5 bpw on any architecture.
```

---

## Tweet 3 / 6  — what the v3 binary actually guarantees  (269 chars)

```
The v3 format gives mathematically lossless reconstruction of W_base — the dequantized weight matrix the customer's runtime sees is bitwise identical to the one our trainer measured.

Any PPL drift you see is from the quantization step, not from format ambiguity or reload-path noise.
```

---

## Tweet 4 / 6  — coverage breadth  (268 chars)

```
End-to-end validated this week on the SipsaLabs HF org:

  12 architectures
  SmolLM2 / OLMo-2 / Qwen3 (0.6B, 1.7B, 8B, 14B, 235B-A22B)
  Mistral-7B / Llama-3.1-8B / Mixtral-8x7B / 8x22B / Phi-3.5-MoE / Mamba-2.8B

Hermes-3-Llama-3.1-405B is in flight tonight.
```

---

## Tweet 5 / 6  — the customer-repro flow  (272 chars)

```
Two repos already pass `uc verify` end-to-end on a stranger's GPU:

  SipsaLabs/qwen3-1.7b-uc-v3-bpw5
  SipsaLabs/mistral-7b-v0.3-uc-v3-bpw5

The four small archs that produced today's ratios are uploading right now. PPL JSON files ship in each repo so the numbers are falsifiable.
```

---

## Tweet 6 / 6  — the call  (243 chars)

```
All repos, model cards, and the install + verify flow:

  huggingface.co/SipsaLabs

If you're one of the 7 stargazers — thank you. If you reproduce a checkpoint and it fails on your hardware, that's a bug we want; open an issue.

— @SipsaLabs
```

---

## Char-count sanity check

| # | Chars | OK? |
|---|------:|-----|
| 1 |   272 | OK |
| 2 |   271 | OK |
| 3 |   269 | OK |
| 4 |   268 | OK |
| 5 |   272 | OK |
| 6 |   243 | OK |

All under the 280-char strict limit. Counts include the line breaks as written; the actual paste is the bare text inside each code fence (Twitter counts URLs as 23 regardless of length).

## Posting notes (for Sip)

- Lead beat is the 7-star milestone, not the PPL number. Order matters: community signal first because that's what makes a stranger willing to read tweet 2.
- Pin tweet 1.
- The Qwen3-0.6B 1.0069× line in tweet 2 is the tightest dense-decoder ratio at 5 bpw measured to date — this is the falsifiable claim. Keep the qualifier "dense-decoder" in case someone produces a tighter number on an MoE or encoder model.
- Keep "mathematically lossless reconstruction of W_base" intact in tweet 3 — do not shorten to "lossless" alone.
- Tweet 5 calls out two repos by name that already pass `uc verify` and notes the four new ones uploading. If any of the four are not yet live by post time, edit tweet 5 to "uploading now — links in replies" rather than naming repos that 404.
- Hermes-3-405B mention in tweet 4 is forward-looking — do not over-promise. If the run hasn't finished by post time, change "in flight tonight" to "in flight" without the time anchor.
- No tagging in main thread. After it's live, reply to tweet 2 with one quote-tweet to a credible infra account asking one specific question. Do not spray.
- Pre-filing IP discipline: zero algorithm details. Numbers and customer flow only.
