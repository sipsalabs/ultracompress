# PUBLIC DEPLOY RUNBOOK — Tonight (2026-05-10)

**Status:** All local code is Charter-clean. PUBLIC surfaces (PyPI / GitHub / sipsalabs.com / HF) are still the OLD leaky state. This runbook walks Sip through the 6 Sip-only actions in the order Codex GPT-5.5 prescribed.

**Time estimate:** ~45-60 min total Sip-time. Highest-leverage 60 minutes of the entire week.

**Strategic axiom (Codex's words, paraphrased):** *Local clean ≠ live clean. The internet is the urgent part. Stop generating more private proof until the public story is clean, credible, and non-leaky.*

---

## STEP 0 — Pre-flight verification (2 min)

Before touching anything live, confirm the local state is what you think it is:

```powershell
cd C:\Users\scamd\ultracompress
Select-String -Path README.md, CHANGELOG.md, ultracompress\*.py -Pattern "GSQ|V18-?C|Track G|grid\[codes\]|low-rank (production-tuned)|train_steps=|`U`|`V`|Cure A4|5\.125"
```

Expected output: only matches inside the public class name `CorrectionLinearV18C` (catch #20 v0.6 rename pending). If you see anything else, **STOP** and fix before publishing.

---

## STEP 1 — Stale failing test (DONE 2026-05-10)

`test_postprocess.py:144` previously asserted `"TRACK G POST-PROCESS RESULTS"`. Now asserts the cleaned `"POST-PROCESS RESULTS"`. **41/41 tests pass.** Verify in the cli repo:

```powershell
cd C:\Users\scamd\OneDrive\Desktop\Projects\sip\ultracompress-cli
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pytest tests -q
# Expected: 41 passed, 1 warning
```

✅ DONE. Ready for build.

---

## STEP 2 — Build + release v0.5.6 to PyPI (~10 min)

The cleaned package code + scrubbed docstrings + `weights_only=True` security fix are all bumped to v0.5.6. CHANGELOG entry written. Build + push:

```powershell
# A. Main repo (source of canonical PyPI publish)
cd C:\Users\scamd\ultracompress
Remove-Item dist\* -Force -ErrorAction SilentlyContinue
python -m build
python -m twine check dist\*
# Expected: PASSED for both .tar.gz and .whl

# B. Verify the wheel doesn't ship leak strings
python -c "import zipfile; z = zipfile.ZipFile([f for f in __import__('os').listdir('dist') if f.endswith('.whl')][0]); [print(n) for n in z.namelist() if n.endswith('.py')]"
# Confirm: only ultracompress/*.py + tests/*.py — no scripts/overlay/, no docs/INTERNAL_*

# C. Upload to PyPI (requires PYPI_TOKEN env var)
python -m twine upload dist\*
# Expected: View at: https://pypi.org/project/ultracompress/0.5.6/
```

**Verify after upload (~30 sec for PyPI to refresh):**

```powershell
pip index versions ultracompress
# Expected output includes: 0.5.6 (top of list)

# Cold-install test in fresh venv:
python -m venv \tmp\uc_test
\tmp\uc_test\Scripts\python -m pip install ultracompress==0.5.6
\tmp\uc_test\Scripts\python -c "import ultracompress; help(ultracompress)"
# Expected: clean docstring, no GSQ/correction overlay/low-rank (production-tuned) leaks
```

⚠️ **IF the build fails with version conflict (twine already-uploaded error):** the previous v0.5.6 metadata is stuck. Bump to v0.5.7 (edit pyproject.toml + __init__.py + CHANGELOG header).

⚠️ **DO NOT publish ultracompress-cli to PyPI** under that name yet — name collision with main package. Per catch #20: rename the cli repo to `uc-cli` first (separate v0.6 work).

---

## STEP 3 — Push cleaned main repo to sipsalabs/main on GitHub (~10 min)

```powershell
cd C:\Users\scamd\ultracompress

# A. Verify what will be pushed
git status
git diff --stat HEAD

# B. Stage the Charter-clean files
git add README.md CHANGELOG.md pyproject.toml ultracompress\__init__.py ultracompress\pack.py ultracompress\pack_v3.py ultracompress\api_v3.py ultracompress\api_v3_memory_aware.py ultracompress\cli.py ultracompress\verify.py

# C. Commit with explicit Charter-cleanup message
git commit -m "v0.5.6: Charter cleanup + RCE fix (catch #12+#14+#17 closure)

- Scrub recipe-level disclosure from public docstrings + CLI help
- Fix CVE-class torch.load(weights_only=False) on all 3 callsites
- Bump version: 0.5.5 -> 0.5.6
- Reword 'lossless' -> 'bit-exact pack reconstruction'
- README rewritten without internal codenames or equations
- Tests: 41/41 pass, ruff clean

Codec internals remain patent-protected (USPTO 64/049,511 + 64/049,517).
Closes catches #12, #14, #16, #17 (per docs/CATCH_*_AUDIT_2026_05_10.md).
"

# D. CRITICAL — verify scripts/overlay/ is NOT in this commit
git diff --stat HEAD~1 | Select-String "scripts/overlay"
# Expected: NO matches. If there are matches, scripts/overlay/ leaked into main again — STOP and `git rm` those before push.

# E. Push to public mirror
git push origin master:main
# (Or: git push sipsalabs main if you have a separate remote for the public mirror)
```

**Verify after push:** Open https://github.com/sipsalabs/ultracompress in browser. README should show v0.5.6 + the cleaned content. The Reproducibility section should NOT show scalar quantization + correction overlay + low-rank + production KL distillation.

⚠️ **scripts/overlay/ MUST NOT BE PUBLIC.** Run before push: `git ls-files scripts/overlay | head -5`. If anything returns, you have a leak surface that needs `git rm -r scripts/overlay/` + commit BEFORE the v0.5.6 push.

---

## STEP 4 — Deploy cleaned sipsalabs.com (~5 min)

You already cleaned the local site (deleted Athena/Quant blocks, rewrote homepage, added Vercel redirects). Just push to production:

```powershell
cd C:\Users\scamd\OneDrive\Desktop\Projects\sip\sipsalabs-site

# A. Verify only canonical files exist
ls *.html
# Expected: index.html, privacy.html (only)

# B. Verify vercel.json redirects are in place
Get-Content vercel.json | Select-String "redirects"
# Expected: 1 match (the redirects array)

# C. Deploy
vercel --prod
# Expected: production URL printed; visit and verify homepage loads cleanly
```

**Verify after deploy (~1 min cache purge):**
- https://sipsalabs.com — should show your cleaned homepage (no Athena, no Quant Trading, no 405B headline at the top)
- https://sipsalabs.com/benchmarks — should redirect to /
- https://sipsalabs.com/inference — should redirect to /
- https://sipsalabs.com/pricing — should redirect to /

---

## STEP 5 — Refresh HuggingFace cards (~15 min)

The 11 LIVE HF cards still leak (per catch #13). Replace with the 11 clean cards in `docs/HF_CARDS_REFRESH_2026_05_10/`:

```powershell
cd C:\Users\scamd\ultracompress\docs\HF_CARDS_REFRESH_2026_05_10

# Verify clean cards exist + are tripwire-clean
Select-String -Path *.md -Pattern "GSQ|V18-?C|Track G|grid\[codes\]|low-rank (production-tuned)|train_steps=|`U`|`V`"
# Expected: no matches

# Apply each card (run the APPLY.md script or do per-card):
# Per-card pattern:
huggingface-cli upload SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5 .\qwen3-1.7b-base.md README.md --commit-message "v0.5.6: Charter-clean refresh (catch #13)"
huggingface-cli upload SipsaLabs/qwen3-8b-uc-v3-bpw5 .\qwen3-8b.md README.md --commit-message "v0.5.6: Charter-clean refresh (catch #13)"
# ...repeat for all 11 cards per APPLY.md
```

**Worst offenders to fix FIRST** (per catch #13 audit):
1. SipsaLabs/yi-1.5-9b-uc-v3-bpw5 — leaks the full reconstruction equation + scalar quantization K=32 + per-block 64 + low-rank + correction overlay + FineWeb-edu + n_eval=50/seq_len=1024/seed=42
2. All 11 hf_publish_v9arch/* cards — leak correction overlay + scalar quantization + `uc compress --rank 32` reproduce CLI

**Plus org bio:** `huggingface-cli repo-files-update SipsaLabs --repo-type=org` (per the org bio refresh doc).

---

## STEP 6 — Re-tripwire against LIVE surfaces (~5 min)

After steps 2-5 land, verify the public internet is actually clean:

```powershell
# A. PyPI long_description (the README from step 3 should now render here)
curl -s https://pypi.org/pypi/ultracompress/json | python -c "import json, sys; d = json.load(sys.stdin); desc = d['info']['description']; bad = [t for t in ['GSQ', 'correction overlay', 'Track G', 'scalar_dequantize(codes)', 'low-rank (production-tuned)', 'train_steps=', '`U`', '`V`', 'Cure A4'] if t in desc]; print('LEAKS:', bad if bad else 'CLEAN')"

# B. GitHub README (renders the same content as the local README.md you just pushed)
curl -s https://raw.githubusercontent.com/sipsalabs/ultracompress/main/README.md | python -c "import sys; t = sys.stdin.read(); bad = [s for s in ['GSQ', 'correction overlay', 'Track G', 'scalar_dequantize(codes)', 'low-rank (production-tuned)', 'train_steps=', '`U`', '`V`', 'Cure A4'] if s in t]; print('LEAKS:', bad if bad else 'CLEAN')"

# C. sipsalabs.com homepage
curl -s https://sipsalabs.com/ | python -c "import sys; t = sys.stdin.read(); bad = [s for s in ['Athena', 'Quant Trading', 'GSQ', 'correction overlay', 'Track G', 'scalar_dequantize(codes)'] if s in t]; print('LEAKS:', bad if bad else 'CLEAN')"

# D. HuggingFace org page + a sample card
curl -s https://huggingface.co/SipsaLabs | python -c "import sys; t = sys.stdin.read(); bad = [s for s in ['GSQ', 'correction overlay', 'Track G', 'scalar_dequantize(codes)'] if s in t]; print('ORG BIO LEAKS:', bad if bad else 'CLEAN')"
curl -s https://huggingface.co/SipsaLabs/qwen3-1.7b-base-uc-v3-bpw5/raw/main/README.md | python -c "import sys; t = sys.stdin.read(); bad = [s for s in ['GSQ', 'correction overlay', 'Track G', 'scalar_dequantize(codes)', 'low-rank (production-tuned)', 'train_steps=', '`U`', '`V`'] if s in t]; print('SAMPLE CARD LEAKS:', bad if bad else 'CLEAN')"
```

**Expected output: `LEAKS: CLEAN` for ALL FOUR.** If any fails, the corresponding step's deploy didn't take — re-run that step.

---

## STEP 7 — Lock the door behind you (NEW Charter rule)

Add a pre-publication tripwire to your shell profile so this can NEVER happen again:

```powershell
# Add to $PROFILE (run: notepad $PROFILE):
function uc-tripwire {
    param([string]$Path = ".")
    $hits = Select-String -Path "$Path\**\*.md","$Path\**\*.html","$Path\**\*.py" -Pattern "GSQ|V18-?C|Track G|grid\[codes\]|low-rank (production-tuned)|train_steps=|`U`|`V`|Cure A4|W_base = absmax" -ErrorAction SilentlyContinue
    if ($hits) {
        Write-Host "RECIPE LEAK DETECTED:" -ForegroundColor Red
        $hits | Select-Object Filename, LineNumber, Line | Format-Table -AutoSize
        return $false
    }
    Write-Host "TRIPWIRE CLEAN" -ForegroundColor Green
    return $true
}
```

Then before EVERY public push: `uc-tripwire .` from the repo root. If it returns false, abort the push.

**Charter Ultra-Review Addendum (paste into `docs/SELECTIVE_DISCLOSURE_CHARTER_2026_05_10.md` §4):**

> **Rule 6 (added 2026-05-10 post-catch-#12 through #17):** Before any `git push` to a public mirror, `vercel --prod`, `huggingface-cli upload`, OR `twine upload`, run the tripwire grep `uc-tripwire .` from the repo root. If it returns ANY match outside the patent-protected class names (`CorrectionLinearV18C` and its `MemoryAware` subclass — pending v0.6 rename), STOP. Fix the leak. Re-run tripwire. Only push when tripwire returns CLEAN.

---

## What you've now closed (via tonight's combined work)

| Catch | Surface | Status |
|---|---|---|
| #12 | README + sipsalabs.com recipe leaks | ✅ CLOSED (README swapped + comment-block stripped + site rewrite + redirects) |
| #14 | PyPI/CLI source leaks | ✅ CLOSED (docstrings scrubbed + version aligned + ruff passes + 41/41 tests) |
| #16 | ultracompress-cli RCE + Charter | ✅ CLOSED (weights_only=True + Track G + correction overlay strings stripped) |
| #17 | GitHub repo (in-package leaks) | ✅ CLOSED for ultracompress/ — scripts/overlay/ STILL NEEDS Sip git rm before next push |
| #13 | HF cards (11 LIVE leak) | ❌ Pending step 5 above |
| #15 | Mon launch + agent-output leaks | ⏳ Pending (Reddit/LinkedIn redact before fire) |

---

## What remains AFTER this runbook

1. **scripts/overlay/ → private repo** (or `git rm -r` from main before any push) — biggest remaining leak surface
2. **8 scripts/overlay/artifacts/*.json files** — full recipe in machine-readable JSON; delete or move
3. **v9 Mistral cure decision** — landed at 1.1114 (worst yet); fires v10 Option B per pre-staged plan; runner ready at `scripts/overlay/hidden_mse_mistral_7b_v10.py`
4. **Class-rename v0.6** — `CorrectionLinearV18C` → `CorrectionLinear` with deprecated alias (catch #20)
5. **NeurIPS paper trim** — 8784 words → 6500 (1-page over 9-page hard cap)
6. **HN launch tomorrow** — only fire AFTER steps 2-6 above PASS tripwire, otherwise the front-page traffic scrapes the leaky version

---

**One-line summary for Sip:** *Steps 1-6 take ~45 min. After they're done, the public internet matches the cleaned local code. Then you can fire HN safely.*
