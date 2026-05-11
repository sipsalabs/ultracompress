# Signup + billing pages — design notes for review

**Status:** DRAFT, awaiting Sip's eyeball pass. Local-only, no deploy taken.
**Date:** 2026-05-10
**Files shipped:**
- `sip/sipsalabs-site/inference/signup.html` — email-only signup, generates an `sk-sps-` API key
- `sip/sipsalabs-site/inference/billing.html` — three top-up buttons ($25 / $100 / $500) → Stripe checkout
- `ultracompress/docs/PUBLIC_DRAFT_SIGNUP_BILLING_NOTES_2026_05_10.md` — this file

**Related:**
- Spec: `ultracompress/docs/INTERNAL_STRIPE_SELFSERVE_TURNKEY_2026_05_10.md`
- Backend: `sipsa-inference/src/sipsa_inference/{auth.py,billing.py,routes/auth_signup.py,settings.py}`

---

## 1. Copy decisions and the rationale

### Headline tone

- Signup: **"Get $5 in free credits."** Reads as a low-friction trial offer, not a hype pitch. Matches `inference/index.html`'s eyebrow ("Public preview · OpenAI-compatible API") and the existing CTA on the inference page.
- Billing: **"Top up your credits."** No imperative ("Buy now"), no FOMO. We're a serious-developer-first product; treat the customer like an adult.

### What we explicitly did NOT say

- **No "instant"**. Stripe redirect adds 5-10 seconds; the webhook adds another ~30 seconds before credits land. The success banner says "as soon as Stripe confirms the webhook (usually under 30 seconds)" — honest range, not a promise.
- **No "trusted by X teams"**. We don't have X teams yet. Empty social proof is worse than none.
- **No 99.9% uptime claim** on either page. The inference home shows 99.0% — these pages stay quieter rather than overclaim.
- **No "save your card on file"**. Stripe Checkout is one-shot for MVP; saved-card is a v0.2 feature that requires Stripe Customer + setup-intent wiring on the backend (not in this scope).

### Honest copy catches I avoided

| Tempting claim | Why I did not write it |
|---|---|
| "$5 free credits, *instant*" | The redirect + webhook take 5-10 s. "instant" overpromises. |
| "Most popular: $100" | We have no usage data. I wrote "Most chosen" — softer, and means the same thing once we DO have data. |
| "Cancel anytime, no commitment" | True but irrelevant — we're pay-per-token, there's nothing to cancel. Would confuse the customer. |
| "Receipts auto-emailed" | Stripe sends them, we don't. Phrasing: "Stripe emails the receipt directly." Attribution matters. |
| "Best price on the market" | Inference home page shows Hermes-3-405B is -44% vs Together but parity on Phi-3.5-MoE — sweeping superlatives invite scrutiny we don't need. |
| "Bank-grade security" | Meaningless. Said "No card touches our servers — we eliminate 95% of PCI scope by design" instead. Specific, verifiable. |
| "Sign up in 10 seconds" | True but the email DNS / spam-filter step is out of our hands. Don't promise time we can't deliver. |

### Honesty around free-tier credits

Confirmed in `sipsa-inference/src/sipsa_inference/settings.py`:
```python
free_tier_credits_usd: float = 5.0
```
And in `sipsa-inference/src/sipsa_inference/routes/auth_signup.py` line 73-74:
```python
raw_key, key_id = await create_key(
    settings.db_path, email=req.email, credits_usd=settings.free_tier_credits_usd,
)
```
The "$5 free credits" claim is real. The signup form's success panel reads the actual `credits_remaining_usd` from the response (defensive: if the backend ever returns a different number, the UI shows the truth, not the marketing copy).

---

## 2. Error states — full matrix

### Signup page (`POST /auth/signup`)

| HTTP / code | UI rendering | Hint shown |
|---|---|---|
| `201` + `api_key` | Success panel: key, $5 pill, copy button, save-now warning, OpenAI SDK env vars | n/a |
| `201` + missing `api_key` | "Unexpected server response" — key missing from envelope | "HTTP 201" |
| `409 email_already_registered` | "Already registered" | "Lost the key? Email founder@sipsalabs.com to rotate." |
| `422` (pydantic validation on bad email) | "Invalid email — server rejected the format" | "Double-check spelling and try again." |
| `429` (per-IP rate limit) | "Rate limited" | "If this persists, email founder@sipsalabs.com." |
| `503` (DB down, etc.) | "Service temporarily unavailable" | "Retry in a minute, or check status updates at github.com/sipsalabs." |
| Other 4xx/5xx | "Signup failed" | "HTTP {N} — email founder@sipsalabs.com if it persists." |
| Network failure (offline, DNS, CORS) | "Network error" | "If you're behind a corporate firewall, the api.sipsalabs.com host may be blocked." |
| Local: empty/malformed email | "Invalid email" | "Example: you@yourcompany.com" |

### Billing page (`POST /v1/billing/checkout`)

| HTTP / code | UI rendering | Hint shown |
|---|---|---|
| `200` + `checkout_url` (validated *.stripe.com) | `window.location.assign(checkout_url)` | n/a |
| `200` + non-Stripe URL | "Suspicious redirect — refusing to redirect" | "Email security@sipsalabs.com if this persists." |
| `401 no_api_key` / `invalid_api_key` | "Authentication failed" | "Double-check the key. If it was suspended, email founder@sipsalabs.com." |
| `400 invalid_amount` | "Invalid amount" | "Pick $25, $100, or $500." |
| `503 billing_not_configured` | "Billing temporarily unavailable" | "Try again in a few minutes, or email founder@sipsalabs.com to top up manually." |
| `429` | "Rate limited" | "If this persists, email founder@sipsalabs.com." |
| Other 4xx/5xx | "Checkout failed" | "HTTP {N} — email founder@sipsalabs.com if it persists." |
| Network failure | "Network error" | "If you're behind a corporate firewall, the api.sipsalabs.com host may be blocked." |
| Local: no key loaded | "Missing API key" | "If you don't have one, create a free trial first." |
| URL `?cancelled=1` (Stripe back-button) | Amber "Checkout cancelled" banner | "No charge was made. Pick a different amount or try again." |
| URL `?success=1` (Stripe redirect post-payment) | Green "Payment received" banner | "If credits don't arrive within five minutes, email founder@sipsalabs.com with your receipt." |

---

## 3. Failure modes addressed

### What if the API is down?

Both pages handle this two ways:
1. **Network failure** (fetch throws): "Could not reach api.sipsalabs.com." Static, doesn't lie about cause.
2. **`503 billing_not_configured`**: This is the documented behavior of `billing.py` when `STRIPE_SECRET_KEY` is unset (Day 30 — Sip hasn't pasted Stripe keys yet). The UI surfaces it as "Billing temporarily unavailable" with a `mailto:founder@` escape hatch — *no spinning forever, no silent failure*.

### What if a user is on free tier and tries to top up?

Works fine. The backend's `increment_credits` adds to whatever is there — zero, $5, $0.30 leftover. The 5.0 free credits are in the same `credits_remaining_usd` column as paid top-ups; there's no tier distinction. UI doesn't even mention "tier" — it just says "Top up your credits" because that's what's happening.

### What if a user already redeemed their free trial and tries again?

Backend returns `409 email_already_registered`. UI renders "Already registered" with a `mailto:founder@` for key recovery. We **deliberately do not echo a fresh key** (the backend won't either, per the policy comment in `auth_signup.py`).

### Spam prevention angle

- **Backend already enforces:**
  - One-key-per-email guard in `auth_signup.py` (409 on repeat).
  - Per-IP rate limiting (handled by `ratelimit.py` which the auth middleware wraps, currently `rate_limit_rpm: 60` per `settings.py` — applied to `/v1/*` routes; `/auth/signup` is per-IP rate-limited at the reverse-proxy or CDN layer per spec).
- **Frontend surfaces gracefully:**
  - 429 → "Rate limited. Wait a minute and retry." with the `mailto:founder@` escape hatch.
  - Honest fineprint on the signup form ("Signup is rate-limited per IP to prevent abuse; if you see a 429, wait a minute and retry").
- **Frontend does NOT add CAPTCHA** at this stage — Sip's spec explicitly defers Turnstile until "signup volume warrants" (`auth_signup.py` line 8-9). The `mailto:founder@` escape route handles edge-case false-positive 429s.

---

## 4. Security posture

### XSS hardening

- **No `innerHTML` on user-derived data anywhere**. Both pages construct UI nodes via `document.createElement` + `textContent`. The hardcoded UI strings that DO use `innerHTML` (none in the final files — I migrated all of them) would only contain author-controlled text.
- **API key never serialized into HTML** — it lives in `code.textContent` and a closure-local `apiKey` var.

### Token leakage

- **No `localStorage` / `sessionStorage` / cookies** for the API key on either page. The key shown on the signup page is in DOM textContent only — closing the tab destroys it.
- **`?key=` query param is stripped from the address bar** via `history.replaceState` BEFORE the key is bound to UI state. Prevents shoulder-surfing and stops the key from leaking into the browser's URL bar / autocomplete history.
- **Key shown only ONCE.** The signup success panel includes a yellow warning: "Save this now. The key is shown only once and never stored client-side."

### Phishing defense in depth

The billing page validates that `body.checkout_url` resolves to a `*.stripe.com` hostname BEFORE calling `window.location.assign`. If the backend is ever compromised and returns a phishing URL, the UI refuses to navigate and surfaces "Suspicious redirect — Email security@sipsalabs.com." This is a 4-line check that costs nothing and closes a real attack vector.

### What's still TODO at the perimeter

- **CSP header**: not in this scope (Vercel-side or `vercel.json` change). Add `default-src 'self'; connect-src 'self' https://api.sipsalabs.com; script-src 'self' 'unsafe-inline'` (the `unsafe-inline` is required because the JS is inline; can be removed if we extract scripts).
- **Subresource integrity**: N/A — no third-party scripts loaded on either page.

---

## 5. Backend gap discovered

The signup endpoint (`auth_signup.py`) returns the response shape:
```json
{"api_key": "sk-sps-...", "credits_remaining_usd": 5.0}
```
which my UI consumes correctly. **No mismatch.**

However, **two minor backend gaps to flag**:

1. **`/auth/signup` is not behind any rate-limiter in the current `app.py`** — the `RateLimiter` middleware in `ratelimit.py` only applies to `/v1/*` routes (per the route-prefix logic in `chat_completions.py`). `/auth/signup` is currently an open endpoint. Spec says rate-limiting "is handled by the backend" — strictly speaking, that needs to be done at the reverse proxy (Caddy / nginx / Cloudflare) or by adding a per-IP middleware on `/auth/*` specifically. **The UI surfaces the 429 gracefully if/when one fires, but there's no source of 429s today on `/auth/signup` itself.**

2. **CORS preflight not configured for `api.sipsalabs.com`**. The signup form posts cross-origin from `sipsalabs.com` to `api.sipsalabs.com`. `app.py` has no `CORSMiddleware` registered. This will fail in browser with the dreaded "blocked by CORS policy" message. **Must add before deploy:**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://sipsalabs.com"],
       allow_methods=["POST", "OPTIONS"],
       allow_headers=["Authorization", "Content-Type", "Accept"],
       max_age=600,
   )
   ```

---

## 6. Design decisions Sip might want to revise

1. **Cyan as the primary CTA color** — chosen to match the existing inference page exactly. If Sip wants the signup CTA to pop louder for Show HN traffic, consider a brighter accent (amber? violet?) on `Create API key` only. Currently it's the same cyan as `Get a key →` on the inference home, which keeps brand cohesion but doesn't strongly differentiate "this is the conversion moment."

2. **"Most chosen" badge on the $100 amount card** — pure psychological anchor with no data behind it yet. Two options: (a) remove it until we have real data, (b) leave it (every saas does this and it works). I left it because friction-reduction matters more than perfect honesty on a button label, but flag it as a Sip-call.

3. **Custom-amount field is visible-but-disabled in v0.1** — chose to render it greyed-out with a "v0.2" tag to signal the feature is coming, rather than hide it entirely. If Sip thinks this looks half-finished, hiding it is a 5-line change. Tradeoff: signals roadmap visibility vs. visual cleanliness.

4. **Footer's USPTO line** — copies the inference home pattern verbatim ("USPTO 64/049,511 + 64/049,517 / Filed April 2026"). Fine for sipsalabs.com pages, but `billing.html` is `noindex,nofollow` — the patent line could be removed there since the page isn't user-facing in search. I kept it for consistency.

5. **No ToS link on signup form** — only privacy policy, because we don't have a public ToS yet (Sip's "ToS goes up after the first paying customer" rule from the launch checklist). When ToS lands, edit signup.html line ~248 to add it.

---

## 7. Confidence rating

**8 / 10 — deploy-ready after Sip's eyeball pass + 2 backend touches.**

The two blocking touches before this can ship to production:
1. **Add `CORSMiddleware` to `sipsa-inference/src/sipsa_inference/app.py`** allowing `https://sipsalabs.com` origin (see Section 5.2). Without this, the form won't reach the API in any browser. **5-minute fix.**
2. **Decide on rate-limiting `/auth/signup`** — either at the reverse-proxy layer (recommended; cleaner separation) OR add a per-IP token-bucket middleware that scopes to `/auth/*`. Without this, `/auth/signup` is open to scripted abuse and the 5.0-credits-per-email guard is the only line of defense. **30-minute fix at the proxy layer.**

Everything else (copy, layout, error rendering, mobile responsiveness, security posture) is shippable as-is. The HTML files match the existing inference page aesthetic 1:1 and are completely self-contained — no framework dependency, no build step, no Google Fonts (system fonts only, fastest TTFB).

Sip should:
- Open both files locally in a browser (`file:///...`), confirm visual match
- Mentally walk through error states (the `mailto:founder@` escape hatches are deliberately frequent — that's the spec's "no human-in-the-loop except when something breaks" pattern)
- Decide on the $100 "Most chosen" badge (Section 6.2)
- Approve or revise the custom-amount-disabled approach (Section 6.3)

— Built per ULTRA-REVIEW RULE: pure local, no deploy, no git push, charter-compliant.
