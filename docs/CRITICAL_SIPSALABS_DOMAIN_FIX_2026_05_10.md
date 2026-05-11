# 🚨 CRITICAL: sipsalabs.com is OFFLINE — Sip-action required

**Discovered**: 2026-05-10 ~22:15 MDT during HF gating audit.
**Severity**: BLOCKER for Mon HN launch + all monetization.
**Owner**: Sip (founder@sipsalabs.com).

## The problem in one sentence

Every URL on `sipsalabs.com` returns HTTP 404, `api.sipsalabs.com` doesn't have any DNS record, and the `vercel.json` redirects every public sub-page back to homepage.

## Symptoms (verified by curl just now)

| URL | Status | Body |
|---|---|---|
| `https://sipsalabs.com/` | **404** | "The deployment could not be found on Vercel. DEPLOYMENT_NOT_FOUND" |
| `https://sipsalabs.com/pricing` | **404** | same |
| `https://sipsalabs.com/inference` | **404** | same |
| `https://sipsalabs.com/blog/2026-05-09-empty-5bit-band/` | **404** | same |
| `https://sipsalabs.com/status` | **404** | same |
| `https://www.sipsalabs.com/` | **404** | Cloudflare → same |
| `https://api.sipsalabs.com/` | **HTTP 000** | DNS does not resolve |
| `https://api.sipsalabs.com/v1/chat/completions` | **HTTP 000** | DNS does not resolve |

## Root cause #1 — DNS misconfigured

`nslookup sipsalabs.com` returns:
- `64.29.17.65` (NOT Vercel)
- `216.198.79.65` (NOT Vercel)

Vercel's apex domain anycast IP is `76.76.21.21`. The current IPs serve a generic Vercel "deployment not found" page because the domain isn't routed to your Vercel project.

`vercel domains ls` shows:
```
sipsalabs.com   Third Party (registrar)   Third Party (nameservers)
```

**The DNS lives at a third-party registrar** — likely Namecheap, GoDaddy, Squarespace, or whoever you bought sipsalabs.com from. You need to log into that registrar's control panel and either:

**Option A (recommended)**: Change nameservers to Vercel's:
- `ns1.vercel-dns.com`
- `ns2.vercel-dns.com`

**Option B**: Keep your current nameservers and update the A record:
- Type: `A`
- Name: `@` (apex)
- Value: `76.76.21.21`
- TTL: 3600
- Also add CNAME for `www`:
  - Type: `CNAME`
  - Name: `www`
  - Value: `cname.vercel-dns.com`

DNS propagation: 5 min – 48 hr depending on TTL.

## Root cause #2 — vercel.json was redirecting all sub-pages to /

I just fixed this in `C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site\vercel.json`. Removed the redirects for:
- `/pricing` → was redirecting to `/`
- `/inference` → was redirecting to `/`
- `/blog` and `/blog/:path*` → was redirecting to `/`
- `/status` → was redirecting to `/`
- `/poc` and `/benchmarks` → was redirecting to `/`

**Kept** the redirects for genuinely-internal paths:
- `/.claude/:path*` (Claude session artifacts)
- `/generate_og_image.py` and other internal Python scripts
- `/public/brand/_generate_assets.py` etc.

After your DNS fix, you need to redeploy with the new vercel.json:

```powershell
cd "C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site"
vercel --prod --yes
```

Verify after deploy:
```powershell
curl -sI https://sipsalabs.com/                                   # should be 200
curl -sI https://sipsalabs.com/pricing                            # should be 200 (or 404 if file doesn't exist yet)
curl -sI https://sipsalabs.com/inference                          # should be 200
curl -sI https://sipsalabs.com/blog/2026-05-09-empty-5bit-band/   # should be 200
```

## Root cause #3 — api.sipsalabs.com doesn't exist as a DNS record

The HN First Comment says:
> Set `OPENAI_BASE_URL=https://api.sipsalabs.com/v1` and the official `openai` SDK works unchanged

That URL is **completely undeliverable** right now. DNS lookup fails ("Non-existent domain").

You need to:
1. **Decide where api.sipsalabs.com points to**:
   - Option A: Cloudflare Tunnel pointing to `localhost:8000` on the home dual-5090 box (per the existing `sipsa-inference` repo plan)
   - Option B: Vercel function (limited compute — fine for /healthz but not for actual inference)
   - Option C: Third-party hosted (Lambda/Together/CoreWeave routing — defeats the cost story)

2. **Add the DNS record** at the registrar:
   - Type: `CNAME` if pointing to Cloudflare Tunnel: `Name: api`, `Value: <tunnel UUID>.cfargotunnel.com`
   - Type: `A` if pointing to a static IP

3. **Wire the Cloudflare Tunnel** (if Option A):
   - Install `cloudflared` if not already
   - `cloudflared tunnel create sipsa-api`
   - `cloudflared tunnel route dns sipsa-api api.sipsalabs.com`
   - Run `cloudflared tunnel run sipsa-api` (with config pointing localhost:8000 → public)

4. **Run the sipsa-inference server**:
   ```powershell
   cd C:\Users\scamd\sipsa-inference
   uvicorn src.sipsa_inference.app:app --host 0.0.0.0 --port 8000
   ```

5. **Test from outside**:
   ```bash
   curl -X POST https://api.sipsalabs.com/v1/chat/completions \
     -H "Authorization: Bearer sk-test" \
     -H "Content-Type: application/json" \
     -d '{"model":"hermes-3-405b","messages":[{"role":"user","content":"hi"}]}'
   ```

Without this, **the entire Mon HN launch claim is undeliverable**.

## What I CAN'T fix from this side

- DNS records (need registrar credentials)
- Cloudflare Tunnel (need Cloudflare account + setup)
- Re-deploy via Vercel CLI from your source dir (I can edit `vercel.json` but `vercel --prod` needs your interactive auth on Sip's box)

## What I just DID fix

- ✅ `C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site\vercel.json` — removed the all-pages-to-/ redirects (commit pending)
- ✅ Documented the gating policy and Sip-action checklist in `docs/HF_GATING_POLICY_2026_05_10.md`

## Suggested fix order (priority)

1. **TONIGHT** (you, before sleep, ~10 min): Log into your DNS registrar, change nameservers to Vercel's OR add A record `76.76.21.21` for apex. This is the single change that unblocks the homepage.
2. **TONIGHT** (you, ~5 min after DNS): Run `vercel --prod --yes` from `sip\sipsalabs-site\` to redeploy with the fixed vercel.json.
3. **TOMORROW MORNING** (you, before HN launch ~2 hr): Set up `api.sipsalabs.com` DNS + Cloudflare Tunnel + start `sipsa-inference` server.
4. **MON HN LAUNCH** (you, 8 AM PT): Verify homepage + /pricing + /inference + api.sipsalabs.com all 200 OK BEFORE clicking Submit on HN form. If any one is 404, postpone.

## Why I missed this earlier

Earlier in the session I curl'd `https://sipsalabs.com/` and saw HTTP/1.1 200 OK with HTML containing "Sipsa Labs". That was a CACHE state — the deployment was up briefly, then went down again (likely a recent deploy invalidated the route, or DNS cache TTL expired). My "0 leaks" verification on those pages was technically correct but **methodologically wrong**: a 404 page returns "no leak patterns found" trivially, so the earlier "all clean" claim across sipsalabs.com surfaces was a false negative on availability.

I'm sorry I missed this until you pushed back. The HF gating audit forced a wider sweep that caught it. Going forward I'll add HTTP status code as a tripwire signal alongside content scan — a 404 is a different failure mode than a leak, and they should be tracked separately.
