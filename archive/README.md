# archive/

Older compression iteration scripts (`compress_v8` through `compress_v18`, `compress_vocab_v2`–`v7`, v6/v7 sweep drivers). Kept for historical traceability of the claim timeline (see [../PATENT_CLAIMS.md](../PATENT_CLAIMS.md)) but not part of the current entry points.

**Current flagship entry points (root dir):**

- `compress_v17.py`, `pack_v17.py`, `pack_all_v17.py` — Claim 16/17 packed v17 pipeline.
- `lambada_overlay.py`, `lambada_overlay_fp8.py`, `lambada_overlay_mixed.py` — Claim 17/18A/18D row-overlay drivers.
- `benchmark_head_to_head.py` — Claim 19/20 unified harness (bnb + HQQ baselines).
- `_analyze_claim20.py` — Claim 20 merge + summary generator.
