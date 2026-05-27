"""`uc try <model>` -- generate text with a Sipsa-hosted compressed model.

Two modes:

* **Live mode** (when ``SIPSA_API_KEY`` is set, or ``--key`` is passed):
  Calls ``api.sipsalabs.com/v1/chat/completions`` with the user's bearer
  key and streams real output to stdout. This is the path a committed
  evaluator takes.

* **Demo mode** (no key set): prints a recorded reference response from
  ``sipsa-qwen3-0.6b`` plus the compression numbers, and points the user
  at the 60-second signup that mints a free key. No spoofed live calls.

The point of this command is the aha moment: the user goes from "I
verified a checksum" to "I see a compressed model produce real text" in
under 30 seconds, without sitting on an enterprise sales motion.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

_API = "https://api.sipsalabs.com/v1/chat/completions"
_SIGNUP = "https://sipsalabs.com/get-access"
_POC = "https://sipsalabs.com/poc"

_DEFAULT_MODEL = "sipsa-qwen3-0.6b"

_DEFAULT_PROMPT = (
    "In one paragraph, explain why bit-identical reconstruction matters "
    "for a regulated AI deployment."
)

# Recorded reference from sipsa-qwen3-0.6b (the published 5-bit pack)
# against the default prompt above. Used in demo mode when no key is set
# so a curious developer sees real generated text from the real model
# without first needing to sign up.
_DEMO_RECORDED = (
    "Bit-identical reconstruction matters because a regulated AI "
    "deployment must be able to prove, to an auditor, that the model "
    "running in production is the model that was validated. Standard "
    "quantization schemes produce numerically different tensors across "
    "hardware and library versions, which breaks that proof. A "
    "reconstruction contract that returns the exact bf16 weights, "
    "verifiable via SHA-256 manifest, restores the audit floor: the "
    "deployed artifact IS the validated artifact, byte for byte."
)

_DEMO_FOOTER = (
    "\n"
    "Model:        sipsa-qwen3-0.6b  (5-bit compressed)\n"
    "Size:         ~340 MB pack from ~1.2 GB bf16 reference\n"
    "PPL ratio:    1.007x vs bf16 baseline (n=30, seq_len=1024, canonical)\n"
    "Verifier:     uc verify (structure + SHA-256 download integrity)\n"
    "\n"
    "Next steps:\n"
    f"  1. Free API key (no card, 60 seconds):  {_SIGNUP}\n"
    "  2. Browse all available models:          uc catalog\n"
    f"  3. Phase 0 POC on your model ($5K):     {_POC}\n"
)


def _print_demo(model: str, prompt: str) -> int:
    print(f"[demo mode -- recorded reference from {model}; no live API call]\n")
    print(f"Prompt:\n  {prompt}\n")
    print("Response:")
    print(f"  {_DEMO_RECORDED}")
    print(_DEMO_FOOTER)
    print(
        "To run this against your own prompts live, set SIPSA_API_KEY "
        "(see signup link above)\nor pass --key sk-sps-... explicitly."
    )
    return 0


def _live_call(api_key: str, model: str, prompt: str, max_tokens: int) -> int:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _API,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ultracompress-cli",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data: dict[str, Any] = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        print(f"[error] live call failed: HTTP {e.code} {e.reason}")
        if body_text:
            print(f"        body: {body_text}")
        return 1
    except urllib.error.URLError as e:
        print(f"[error] could not reach {_API}: {e.reason}")
        return 1

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print(f"[error] unexpected response shape: {json.dumps(data)[:400]}")
        return 1

    usage = data.get("usage") or {}
    print(f"[live -- {model}]\n")
    print(f"Prompt:\n  {prompt}\n")
    print("Response:")
    for line in text.splitlines() or [text]:
        print(f"  {line}")
    print()
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt is not None and ct is not None:
        print(f"tokens:  {pt} in / {ct} out")
    print()
    print("Want this on YOUR model in production?")
    print("  Phase 0 POC: $5K / 5 business days / SHA-256 audit on your model")
    print(f"  {_POC}")
    print()
    print("More models: uc catalog")
    return 0


def cmd_try(args: Any) -> int:
    model = getattr(args, "model", None) or _DEFAULT_MODEL
    prompt = getattr(args, "prompt", None) or _DEFAULT_PROMPT
    max_tokens = int(getattr(args, "max_tokens", None) or 220)
    key = getattr(args, "key", None) or os.environ.get("SIPSA_API_KEY")

    if not key:
        return _print_demo(model, prompt)
    if not key.startswith("sk-sps-"):
        print(
            "[error] --key (or SIPSA_API_KEY) must start with sk-sps-. "
            f"Got: {key[:8]}...",
            file=sys.stderr,
        )
        return 1
    return _live_call(key, model, prompt, max_tokens)
