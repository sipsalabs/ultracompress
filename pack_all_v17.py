"""pack_all_v17.py — pack every v17 fit we have into binary form and
emit a summary table. Proves the 2.40-bpw on-disk format generalises
across all 6 models in the Claim-16 envelope (3 families, 3 corpora,
1.1B → 8.2B)."""
from __future__ import annotations
import json, os, time
import torch
from pack_v17 import pack_fit

MODELS = [
    ("Qwen3-1.7B",   "v17_fit_qwen3_1.7b.pt", "qwen3_1.7b_cache.pt",    "v17_qwen3_1.7b.bin"),
    ("Qwen3-8B",     "v17_fit_8b.pt",         "qwen3_8b_cache.pt",      "v17_qwen3_8b.bin"),
    ("Mistral-7B",   "v17_fit_mistral.pt",    "mistral_7b_v0.3_cache.pt", "v17_mistral_7b.bin"),
    ("TinyLlama-1.1B", "v17_fit_tinyllama.pt", "tinyllama_1.1b_cache.pt", "v17_tinyllama.bin"),
    ("SmolLM2-1.7B", "v17_fit_smollm2.pt",    "smollm2_1.7b_cache.pt",  "v17_smollm2.bin"),
    ("OLMo-2-1B",    "v17_fit_olmo2.pt",      "olmo2_1b_cache.pt",      "v17_olmo2.bin"),
]

summary = []
for name, fit, teacher, out in MODELS:
    if not os.path.exists(fit) or not os.path.exists(teacher):
        print(f"[skip] {name}: missing {fit} or {teacher}")
        continue
    if os.path.exists(out):
        print(f"[skip-pack] {name}: {out} exists ({os.path.getsize(out):,} bytes)")
        # synthesize summary from existing
        v17 = torch.load(fit, map_location="cpu", weights_only=False)
        total_bpw_claim = (v17.get("global_bpw") or 0) + (v17.get("overhead_bpw") or 0)
        # Count params from the pack header
        with open(out, "rb") as f:
            import struct
            f.read(6)
            (hl,) = struct.unpack("<I", f.read(4))
            header = json.loads(f.read(hl).decode("utf-8"))
        params = sum(w["O"] * w["I"] for w in header["weights"])
        size = os.path.getsize(out)
        bpw = size * 8 / params
        summary.append({
            "name": name, "fit": fit, "bin": out,
            "bytes": size, "params": params,
            "bpw_disk": bpw, "bpw_claim": total_bpw_claim,
            "a_attn": v17.get("a_attn"), "a_mlp": v17.get("a_mlp"),
        })
        continue
    t0 = time.time()
    print(f"\n{'='*64}\n[pack] {name}\n{'='*64}")
    info = pack_fit(fit, teacher, out, device="cuda:0")
    v17 = torch.load(fit, map_location="cpu", weights_only=False)
    summary.append({
        "name": name, "fit": fit, "bin": out,
        "bytes": info["bytes"], "params": info["params"],
        "bpw_disk": info["bpw_disk"],
        "bpw_claim": (v17.get("global_bpw") or 0) + (v17.get("overhead_bpw") or 0),
        "a_attn": v17.get("a_attn"), "a_mlp": v17.get("a_mlp"),
        "wall_sec": time.time() - t0,
    })
    torch.cuda.empty_cache()

print("\n" + "="*80)
print(f"{'Model':<18} {'Params':>14} {'Bytes':>14} {'bpw_disk':>10} {'bpw_claim':>10} {'delta':>8}")
print("="*80)
for s in summary:
    d = s["bpw_disk"] - s["bpw_claim"]
    print(f"{s['name']:<18} {s['params']:>14,} {s['bytes']:>14,} "
          f"{s['bpw_disk']:>10.4f} {s['bpw_claim']:>10.4f} {d:>+8.4f}")
print("="*80)

with open("pack_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n[pack] wrote pack_summary.json")
