# IP Hardening + Sponsorship Setup (2026-05-09)

Two 30-min lifts to harden the public sipsalabs/ultracompress repo's IP posture and turn on visible support channels.

## What was added/modified

Three files changed in this pass:

1. **`PATENT_NOTICE.md`** (new, repo root) — Standalone document declaring USPTO provisionals 64/049,511 + 64/049,517 cover the GSQ + V18-C + streaming pipeline methods. Spells out the Apache 2.0 grant for as-published source vs. commercial-license ask for productization. Contact: founder@sipsalabs.com.
2. **`README.md`** (modified) — Added a 2-line callout block right under the badges, before "Current state". One line links to the patent notice; one line links to GitHub Sponsors + paid POC email. Kept the engineer-voice tone of the existing README; no corporate hedging.
3. **`.github/FUNDING.yml`** (new) — Activates the "Sponsor" button on the repo page. Lists `sipsalabs` as the org GitHub Sponsor account and `https://sipsalabs.com/sponsor` as a custom URL placeholder.

`LICENSE` was **not** touched — Apache 2.0 stays exactly as it was. The patent notice supplements the license, it does not replace or override it.

## Manual steps Sip needs to take

### 1. Activate GitHub Sponsors for the org

`.github/FUNDING.yml` only takes effect once the `sipsalabs` org actually has Sponsors enabled. Steps:

1. Go to **https://github.com/sponsors/signup**
2. Apply as the **SipsaLabs org** (NOT the personal account — the org needs its own Stripe Connect or fiscal-host setup, and donations should land in the company entity not personal income)
3. Approval takes 1–7 days
4. Once approved, the "Sponsor" button appears on the `sipsalabs/ultracompress` repo automatically
5. The custom URL `https://sipsalabs.com/sponsor` in FUNDING.yml is a placeholder — point it at a real sponsor page later (or remove it)

The org Sponsors application will require a payout method. With Stripe Atlas EIN expected May 13–15, this is fine to start now (the application is reviewed during that window).

### 2. Commit and push the three files

From `C:\Users\scamd\ultracompress` run:

```bash
git add PATENT_NOTICE.md README.md .github/FUNDING.yml
git commit -m "add patent notice + sponsor setup"
git push origin main
```

Subagent did NOT commit — left for Sip to review the diffs and push manually.

## Honest caveat on the patent notice

A patent notice in a README is a **soft deterrent**. It does not stop bad actors. What it actually does:

- Establishes that you are paying attention and the IP is documented
- Creates a paper trail (and a date stamp) if something egregious shows up later — useful for demonstrating "willful infringement" in a commercial dispute
- Sets the tone that this is a real company with patent counsel, not a hobby project — discourages casual reskinning by firms with legal departments
- Gives honest commercial users a clear contact path (founder@sipsalabs.com) so they can self-select into a license conversation

What it does **not** do:

- Stop a determined competitor from reimplementing the methods in another language
- Substitute for actual patent enforcement (cease-and-desist, litigation) when the time comes
- Provide any technical access control over the code

Treat it as a posture signal, not a security control. Real protection comes from (a) the granted patent itself (still pending), (b) paying attention to who uses it, and (c) being willing to enforce when productization happens at scale.
