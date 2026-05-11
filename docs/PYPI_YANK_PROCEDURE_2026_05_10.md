# PyPI Yank Procedure — Sip-action

**Date**: 2026-05-10 night
**Reason**: v0.5.6 + v0.6.0 had Charter leaks (recipe equation in docstrings, V18-C class names in public API). v0.6.1 supersedes both and is fully Charter-clean.

## What to yank

| Version | Why yank | Replacement |
|---|---|---|
| `0.5.6` | `W_base = absmax × grid[codes]` in verify.py docstring; `train_steps=1500` default; `gsq_codecs` legacy fallback string; `V18C` class name in public API | `0.5.5` (legacy Apache-2.0 last-known-clean) or `0.6.1` (BUSL-1.1 current) |
| `0.6.0` | `V18CCorrectedLinear` mention in README repo layout (METADATA leak); `seed.py` line 91 `rank=64` recipe leak; class name `CorrectionLinearV18C` still present in public API | `0.6.1` |

Do NOT yank `0.5.5` or earlier — they remain available for users on the legacy Apache-2.0 branch.

## How to yank (browser, ~3 min total)

### v0.5.6 yank
1. Open: <https://pypi.org/manage/project/ultracompress/release/0.5.6/>
2. Sign in as `sip786`
3. Scroll to bottom: "Yank release"
4. Reason text: `Superseded by 0.6.1; v0.5.6 contained internal codenames in public API. Existing pinned installs continue to work; new installs get 0.6.1.`
5. Click "Yank release"

### v0.6.0 yank
1. Open: <https://pypi.org/manage/project/ultracompress/release/0.6.0/>
2. Same flow as above
3. Reason text: `Superseded by 0.6.1; minor Charter cleanup. v0.6.0 packs continue to work; new installs get 0.6.1.`

## What yanking does (and doesn't)

- ✅ Removes the version from `pip install ultracompress` default resolution
- ✅ Anyone with `ultracompress==0.5.6` or `==0.6.0` pinned in requirements.txt CAN STILL INSTALL it (yanking doesn't delete)
- ✅ Adds a "yanked" badge on the PyPI page with the reason text
- ❌ Does NOT delete the wheel/sdist files (PyPI never deletes — too risky for downstream)
- ❌ Does NOT cancel existing installs

## Verification (post-yank)

```bash
pip install ultracompress  # Should resolve to 0.6.1 (latest non-yanked)
pip install ultracompress==0.5.6  # Should warn about yanked version + still install
```

## Live current state

- ✅ PyPI 0.6.1 LIVE: <https://pypi.org/project/ultracompress/0.6.1/>
- ✅ Latest non-yanked: 0.6.1
- ✅ Wheel content tripwire: ZERO leaks
- ✅ All 34 SipsaLabs HF model cards: ZERO leaks
- ✅ GitHub sipsalabs/main README: clean
- ✅ sipsalabs.com (homepage + inference + blog): clean

## Notes

- The yank is housekeeping, not strictly required for Charter compliance (since v0.6.1 is now `pip install ultracompress` default).
- Patent (USPTO 64/049,511 + 64/049,517) protects commercial use of the algorithm regardless of which version is installed.
- The legacy Apache-2.0 v0.5.x branch on GitHub continues to support v0.5.5 callers.
