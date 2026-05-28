"""`uc catalog` — print the Sipsa Labs compressed model catalog.

Fetches the public ``/v1/models`` endpoint (no auth required) and prints
a formatted table. Adds the per-model tier label and CTAs so the user
can see at a glance which models are open, which are gated, and which
are POC-only — and what to do next for each tier.

If the API is unreachable (offline, censored network), falls back to a
URL pointer so the user can still reach the catalog.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

_API = "https://api.sipsalabs.com/v1/models"
_SITE_CATALOG = "https://sipsalabs.com/inference"
_ACCESS = "https://sipsalabs.com/access"
_POC = "https://sipsalabs.com/poc"


# Curated per-model metadata. The public /v1/models endpoint returns only
# the OpenAI-compatible id+object envelope, so we annotate client-side
# with PPL ratios + tier markers + approximate size. Everything here is
# already public on sipsalabs.com/inference and the GitHub README.
# PPL ratios below are the canonical verified records from
# docs/benchmarks.json verified_records[] (public auditor registry), rounded to 3 decimals
# with an "x" suffix. Updating this dict requires re-running canonical eval
# and updating the JSON source first; never edit by hand.
# Entries marked "ppl_pending" are SHA-256-verified shipped packs whose
# canonical PPL run is still in the queue.
_MODEL_META: dict[str, dict[str, str]] = {
    "sipsa-qwen3-0.6b":           {"params": "0.6B",  "ppl": "1.007x",      "tier": "free"},
    "sipsa-tinyllama-1.1b":       {"params": "1.1B",  "ppl": "1.003x",      "tier": "free"},
    "sipsa-smollm2-1.7b":         {"params": "1.7B",  "ppl": "1.008x",      "tier": "free"},
    "sipsa-qwen3-1.7b":           {"params": "1.7B",  "ppl": "1.008x",      "tier": "free"},
    "sipsa-qwen3-1.7b-base":      {"params": "1.7B",  "ppl": "1.004x",      "tier": "free"},
    "sipsa-olmo-2-1b":            {"params": "1.0B",  "ppl": "1.007x",      "tier": "free"},
    "sipsa-mamba-2.8b":           {"params": "2.8B",  "ppl": "1.006x*",     "tier": "free"},
    "sipsa-phi-3-mini-4k":        {"params": "3.8B",  "ppl": "1.003x",      "tier": "free"},
    "sipsa-llama-3.1-8b":         {"params": "8B",    "ppl": "1.012x",      "tier": "gated"},
    "sipsa-qwen3-8b":             {"params": "8B",    "ppl": "1.004x",      "tier": "gated"},
    "sipsa-mistral-7b-v0.3":      {"params": "7B",    "ppl": "1.005x",      "tier": "gated"},
    "sipsa-phi-4":                {"params": "14B",   "ppl": "1.005x",      "tier": "gated"},
    "sipsa-qwen3-14b":            {"params": "14B",   "ppl": "1.004x",      "tier": "gated"},
    "sipsa-qwen3-32b":            {"params": "32B",   "ppl": "ppl_pending", "tier": "gated"},
    "sipsa-mixtral-8x7b":         {"params": "46B",   "ppl": "1.004x",      "tier": "gated"},
    "sipsa-llama-3.1-70b":        {"params": "70B",   "ppl": "1.009x",      "tier": "gated"},
    "sipsa-phi-3.5-moe":          {"params": "42B",   "ppl": "1.001x",      "tier": "gated"},
    "sipsa-mixtral-8x22b":        {"params": "141B",  "ppl": "1.006x",      "tier": "gated"},
    "sipsa-qwen3-235b-a22b":      {"params": "235B",  "ppl": "1.004x",      "tier": "gated"},
    "sipsa-hermes-3-llama-3.1-405b": {"params": "405B", "ppl": "1.007x",    "tier": "gated"},
    # Three additions 2026-05-28 to bring catalog into sync with the 22
    # PPL-verified architectures + 1 ppl_pending = 23-entry chat catalog
    # (the 23rd architecture across 4 classes — DINOv2-Large ViT — is
    # cosine-verified and not chat-completable, so it is intentionally
    # NOT in this catalog; see sipsalabs.com/research for the ViT entry).
    "sipsa-yi-1.5-9b":             {"params": "9B",    "ppl": "1.004x",      "tier": "gated"},
    "sipsa-olmo-2-1b-base":        {"params": "1.0B",  "ppl": "1.007x",      "tier": "gated"},
    "sipsa-smollm2-1.7b-base":     {"params": "1.7B",  "ppl": "1.008x",      "tier": "gated"},
}

_TIER_LABEL = {
    "free":   "free      ",
    "gated":  "request   ",
    "custom": "POC ($5K) ",
}


def cmd_catalog(_args: Any = None) -> int:
    req = urllib.request.Request(
        _API,
        headers={"User-Agent": "ultracompress-cli", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        print(f"[offline] could not reach catalog API: {e}")
        print(f"Full catalog: {_SITE_CATALOG}")
        return 1

    models = data.get("data", [])
    print(f"{'Model':<37} {'Params':>8} {'PPL':>13}  {'Tier':<10}")
    print("-" * 72)
    free_count = 0
    has_ssm_caveat = False
    has_pending = False
    for m in models:
        mid = m.get("id", "?")
        meta = _MODEL_META.get(mid, {})
        params = meta.get("params", "?")
        ppl = meta.get("ppl", "?")
        tier = meta.get("tier", "gated")
        if tier == "free":
            free_count += 1
        if ppl.endswith("*"):
            has_ssm_caveat = True
        if ppl == "ppl_pending":
            has_pending = True
        label = _TIER_LABEL.get(tier, tier)
        print(f"{mid:<37} {params:>8} {ppl:>13}  {label}")

    if has_ssm_caveat:
        print("\n* SSM record uses architecture-compatible comparator; see")
        print("  github.com/sipsalabs/ultracompress/blob/main/docs/benchmarks.json comparator_note.")
    if has_pending:
        print("\nppl_pending: SHA-256-verified shipped pack; canonical PPL run queued.")

    total = len(models)
    print(
        f"\n{free_count} models free to try | "
        f"{total - free_count} available under engagement"
    )
    print("\nTry any free model:")
    print("  uc try <model-id>          # demo prompt, no signup")
    print("  uc try <model-id> --key sk-sps-...   # live, your own prompts")
    print(f"\nRequest a gated model:       {_ACCESS}")
    print(f"Phase 0 POC ($5K / 5 days):  {_POC}")
    return 0
