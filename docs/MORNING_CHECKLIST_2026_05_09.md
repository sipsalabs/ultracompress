# MORNING CHECKLIST — Saturday 2026-05-09

**Owner:** Sip
**Trigger:** Hermes-405B compression run finishes overnight (~midnight Friday → wake Saturday morning)
**Total elapsed:** ~6-8 hours wall, ~90 min hands-on
**Read time:** 3 min. Then execute top-to-bottom. Do not deviate.

---

## RULE OF THE DAY

Hermes-405B is the news. Every channel push depends on a clean PPL eval (Step 2). If PPL ratio > 1.030x, **STOP**. Debug instead of launching. The science is the moat.

---

## 0. Wake-up sanity check (5 min, ~07:30 PT)

- [ ] `Test-Path 'C:\Users\scamd\ultracompress\scripts\overlay\_e2e_hermes_3_405b_v3\layer_125.pt'` → must return `True`
- [ ] If `False`: Hermes run failed mid-stream. **HALT all launch activity.** Open `scripts/overlay/_e2e_hermes_3_405b_v3/_run.log` (last 100 lines), diagnose, reschedule run. Do NOT post anywhere.
- [ ] `curl -I https://sipsalabs.com` → 200 OK
- [ ] `gh repo view sipsalabs/ultracompress` → public, no surprise outage

**Done criterion:** layer_125.pt exists AND sipsalabs.com is up AND repo is public. All three required to proceed.

---

## 1. Pack + verify Hermes-405B (10-15 min, ~07:45 PT)

### Pack
```powershell
cd C:\Users\scamd\ultracompress
python -c "from ultracompress.pack_v3 import pack_e2e_dir_v3; pack_e2e_dir_v3('scripts/overlay/_e2e_hermes_3_405b_v3', '_packed_hermes_3_405b_v3', bpw=5, block_size=64)"
```

### Verify with full hash audit
```powershell
uc verify _packed_hermes_3_405b_v3 --compute-hashes
```

### Capture for downstream paste
```powershell
Get-FileHash _packed_hermes_3_405b_v3\layer_000.uc -Algorithm SHA256 | Select-Object -ExpandProperty Hash
```
**Save this hash. You will paste it 3 times today (README, tweet, press release).**

**Expected pack size:** ~120 GB
**Done criterion:** verify exits 0 AND layer_000.uc SHA256 captured to clipboard/notepad.

---

## 2. Run PPL eval (30-40 min, ~08:00 PT — kick off, then move to Step 3)

```powershell
python scripts/overlay/eval_compressed_only.py `
  --model hermes-3-405b `
  --compressed_dir scripts/overlay/_e2e_hermes_3_405b_v3 `
  --n_eval 30 `
  --seq_len 1024 `
  --device cuda:0 `
  --out_json docs/PPL_EVAL_hermes-3-405b_2026_05_09.json
```

**Expected:** baseline_ppl ≈ 4.91, compressed_ppl ≈ 4.95, ppl_ratio ≈ 1.007x

**HARD GATE — read the JSON when it completes:**
- ratio ≤ 1.015x → green, full launch
- 1.015x < ratio ≤ 1.030x → yellow, launch but lead with "preliminary, in-flight tuning"
- ratio > 1.030x → **RED. STOP. Do not post. Debug.**

Capture three numbers: `baseline_ppl`, `compressed_ppl`, `ppl_ratio`. Paste into README + tweet + press release.

**Done criterion:** JSON written AND ratio ≤ 1.030x AND three numbers captured.

---

## 3. Fill placeholders + start HF upload (5 min hands-on, ~08:40 PT)

```powershell
# Open template, fill 3 placeholders: baseline_ppl, compressed_ppl, layer_000.uc SHA256
code _packed_hermes_3_405b_v3_README_TEMPLATE.md
# Save As → _packed_hermes_3_405b_v3\README.md
```

### Kick off upload (runs 2-4 hours in background, watchdog auto-retries)
```powershell
python scripts/overlay/_hf_upload_simple.py _packed_hermes_3_405b_v3 SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5
```

**Done criterion:** README has zero `{{PLACEHOLDER}}` strings remaining AND upload process is running (PID captured).

---

## 4. Coordinated multi-channel push (20 min, all in one 60-min window, ~08:30-09:30 PT)

> **Sequence intent:** Tweet first (algorithm picks up engagement), LinkedIn 30 min later (different audience), press embargo 09:00, reporter outreach 09:00:01. HN follow-up at Step 5.

### 4a. Tweet thread @SipsaLabs (08:00-09:30 PT optimal)
- **Source:** `docs/LAUNCH_THREAD_HERMES_405B_2026_05_09.md`
- **Target:** twitter.com/SipsaLabs
- **Steps:** Open file → copy first tweet → paste in compose → Post → repeat for thread (paste each tweet as Reply to previous). Pin first tweet.
- **Done:** thread is live, pinned, ratio number is in tweet 2.

### 4b. LinkedIn long-form (08:30-10:00 PT optimal)
- **Source:** `docs/LAUNCH_LINKEDIN_HERMES_405B_2026_05_09.md`
- **Target:** linkedin.com/company/sipsalabs (post as the org, not personal)
- **Steps:** "Start a post" → paste full text → add 1 image (HF model card screenshot) → Post.
- **Done:** post is live on Sipsa Labs company page.

### 4c. HuggingFace org bio refresh
- **Wait until** Hermes commit lands (4a/4b can happen in parallel; 4c depends on Step 3 upload progressing past first commit ~30 min in)
- ```powershell
  python _refresh_hf_org_bio.py
  ```
- **Done:** huggingface.co/SipsaLabs bio shows Hermes-405B in the model list.

### 4d. sipsalabs.com homepage redeploy
```powershell
cd C:\Users\scamd\sip\sipsalabs-site
vercel --prod
```
- **Done:** sipsalabs.com loads with Hermes-405B in the "Compressed models" section.

### 4e. sipsalabs.com/blog deploy
- Same `vercel --prod` from Step 4d covers it (blog is in the same repo).
- **Verify:** browse to `sipsalabs.com/blog/2026-05-08-eighteen-architectures/` → 200 OK.
- **Done:** blog post resolves publicly.

### 4f. Press release (BusinessWire / PRNewswire)
- **Source:** `docs/PRESS_RELEASE_HERMES_405B_2026_05_09.md`
- **Steps:** Log into BusinessWire (or PRNewswire) → New Release → paste body → set embargo: **09:00 PT today** → Schedule → confirm.
- **Done:** release is scheduled with embargo time visible.

### 4g. Reporter outreach (12 emails, send AFTER 09:00 PT embargo lifts)
- **Source:** `docs/PRESS_RELEASE_DISTRIBUTION_LIST_2026_05_09.md`
- **Tier 1 (7 specialist reporters):** send immediately at 09:00:01 PT.
- **Tier 2 (5 generalists):** send at 09:30 PT.
- **Use:** sipsalabs@gmail.com (NOT personal Gmail — see global memory: no personal info on public surfaces).
- **Done:** all 12 sent, no bounces.

### 4h. Investor update — top 5
- **Source:** `docs/INVESTOR_UPDATE_TEMPLATE_2026_05_08.md`
- **Steps:** Personalize the {{name}} / {{firm}} / {{prev_thread_ref}} blocks for 5 priority investors. Send from sipsalabs@gmail.com.
- **Done:** 5 personalized emails sent.

### 4i. Cold outreach — 5 named targets
- **Source folder:** `docs/OUTREACH_2026_05_08/`
- **Targets:** Tri Dao, Albert Gu, Yi Tay, Lambda, NASA HPSC
- **Steps:** open each .md → copy body → paste in compose with subject line from frontmatter → send from sipsalabs@gmail.com.
- **Done:** 5 sent.

---

## 5. HN Show v2 follow-up post (10 min, ~10:00 PT)

- **Source:** `docs/SHOW_HN_V2_2026_05_08.md`
- **Action:** Submit as a NEW HN post (per the doc's recommendation — do NOT reply-thread the existing 4-point post).
- **Steps:**
  1. news.ycombinator.com/submit
  2. Title from doc frontmatter, URL = HF model card link
  3. Click Submit
  4. **Within 60 sec:** post the "first comment from OP" block (also in the doc) as a top-level comment on your own post.
- **Then:** monitor for replies for 90 min. HN's front-page algorithm rewards quick OP responses. Reply with numbers + repro steps. Do not engage trolls beyond that.

**Done:** post submitted AND OP comment landed AND first reply (if any) answered.

---

## 6. Patent batch filing (60-90 min, USPTO EFS-Web, ~11:00 PT)

- **Runbook:** `docs/MAY_9_FILING_RUNBOOK.md`
- **Batch:** 5 provisionals, $325 micro fee total
  1. DSR-Q
  2. Track A v4 supplement
  3. Track B v2 supplement
  4. Track C v2
  5. UC-IR Layer-2

### Per-PDF check (do this for all 5 before logging into USPTO)
```powershell
pandoc <source.md> --pdf-engine=xelatex -o <out.pdf>
# Open each PDF, eyeball page 1, last page, no broken figures
```

### File via EFS-Web
- efs.uspto.gov → log in → Provisional Application → upload PDF → micro entity → pay → **capture provisional number** for each.
- **Log all 5 provisional numbers** to `docs/PATENT_PROVISIONALS_LOG.md` immediately.

**Done:** 5 numbers logged. Total spend $325. Both already-filed apps (64/049,511 + 64/049,517) untouched.

---

## 7. Watch for Atlas EIN (passive — check every ~2 hours)

- **Window:** Day 2-3 of 1-7 day Atlas turnaround. Email arrives random business hours.
- **When EIN lands:** immediately file SAM.gov UEI registration. UEI unblocks NASA + AFWERX SBIR submissions next week.
- **Action:** check sipsalabs@gmail.com inbox at 11:00 / 13:00 / 15:00 / 17:00.

**Done if EIN arrives today:** UEI submission filed same day.

---

## 8. Throughout the day: monitor + respond

- **HN comments:** reply within 90 min if engagement spikes (numbers + repro flow only — don't argue)
- **Twitter @ replies:** reply within 30 min during business hours
- **LinkedIn comments:** reply within 60 min
- **Reporter responses:** 24-72hr typical; reply same-day if any land
- **Inbound on sipsalabs@gmail.com:** triage hourly
- **HF org analytics:** check hf.co/SipsaLabs at 14:00 + 18:00 for traffic spike

---

## 9. End-of-day status pulse (8 PM ~20:00 PT)

```powershell
cd C:\Users\scamd\ultracompress
python verify_all_committed.py
# Target: 17/17 PASS by EOD if all uploads landed

git pull --rebase origin master
git push sipsalabs master

# Update LAB-NOTEBOOK with predicted vs actual outcomes for the day
code docs/LAB-NOTEBOOK.md
```

- **Investor follow-up email** (1-line, "shipped what we promised") to the same 5 from Step 4h.

**Done:** notebook entry written, repo synced, follow-up sent.

---

## DO NOT — anti-tunnel-vision protections

1. **DO NOT post anything publicly** if Hermes-405B PPL ratio > 1.030x. The science is the moat. Killed launch > broken claim.
2. **DO NOT skip the SHA256 audit** in Step 1. It is the falsifiability anchor that lets reporters / academics verify independently.
3. **DO NOT click Send on Step 4g cold emails** before the press release embargo lifts at 09:00 PT. Pre-embargo leaks burn the BusinessWire relationship.
4. **DO NOT engage HN trolls** with anything except numbers + repro flow. No tone-matching, no defensiveness.
5. **DO NOT discuss algorithm internals** beyond what is already in the open-source `pack_v3.py`. Patent filings (Step 6) are still in the 12-month provisional window. Loose lips kill priority.
6. **DO NOT use personal Gmail (`micipsa.ounner@gmail.com`)** anywhere public today. Use sipsalabs@gmail.com for all outbound. (Per global memory: zero personal info on public surfaces.)
7. **DO NOT spend money** beyond the $325 USPTO batch in Step 6 (cash-constrained pre-funding mode until YC June 5 OR first revenue).

---

## End-state by 21:00 PT

- Hermes-405B compressed, verified, on HuggingFace
- 9 surfaces have shipped coordinated content
- 5 patent provisionals filed (8 total now on record)
- Investor + reporter inbox triaged
- LAB-NOTEBOOK updated
- Repo synced

If all 9 sections checked: tomorrow (Sunday) is a rest day. Earned.
