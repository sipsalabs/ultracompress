"""Run pack_v17 round-trip verification on every model in pack_summary.json
that hasn't been verified yet (Qwen3-8B already done). Writes
verify_all_results.json."""
from __future__ import annotations
import json, gc, time, traceback, os
import torch
from pack_v17 import verify_roundtrip

MODEL_IDS = {
    "Qwen3-1.7B":     ("Qwen/Qwen3-1.7B",                "qwen3_1.7b_cache.pt",         "wikitext103_test_qwen3.pt"),
    "Qwen3-8B":       ("Qwen/Qwen3-8B",                  "qwen3_8b_cache.pt",           "wikitext103_test_qwen3.pt"),
    "Mistral-7B":     ("mistralai/Mistral-7B-v0.3",      "mistral_7b_v0.3_cache.pt",    "wikitext103_test_mistral.pt"),
    "TinyLlama-1.1B": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_1.1b_cache.pt",  "wikitext103_test_tinyllama.pt"),
    "SmolLM2-1.7B":   ("HuggingFaceTB/SmolLM2-1.7B",     "smollm2_1.7b_cache.pt",       "wikitext103_test_smollm2.pt"),
    "OLMo-2-1B":      ("allenai/OLMo-2-0425-1B",         "olmo2_1b_cache.pt",           "wikitext103_test_olmo2.pt"),
}


def main():
    with open("pack_summary.json") as f:
        entries = json.load(f)

    # Skip Qwen3-8B — already verified in verify_8b.log.
    skip = {"Qwen3-8B"}

    # Process small models first so big ones don't block early harvest.
    order = {"OLMo-2-1B": 0, "TinyLlama-1.1B": 1, "SmolLM2-1.7B": 2,
             "Qwen3-1.7B": 3, "Mistral-7B": 4, "Qwen3-8B": 5}
    entries.sort(key=lambda e: order.get(e["name"], 99))

    results = []
    out_path = "verify_all_results.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    done = {r["name"] for r in results}

    for e in entries:
        name = e["name"]
        if name in skip or name in done:
            print(f"[verify_all] skip {name}")
            continue
        mid, teacher, toks = MODEL_IDS[name]
        print(f"\n==================== {name} ====================")
        t0 = time.time()
        try:
            r = verify_roundtrip(
                teacher_path=teacher, v17_path=e["fit"], pack_path=e["bin"],
                model_id=mid, tokens_path=toks, n=16, seq_len=128,
                device="cuda:0",
            )
            wall = time.time() - t0
            rec = {"name": name, "model_id": mid, "wall_s": round(wall, 1),
                   "ppl_original": r["ppl_original"],
                   "ppl_packed":   r["ppl_packed"],
                   "rel_diff":     r["rel_diff"]}
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            rec = {"name": name, "model_id": mid, "error": repr(ex)}
        results.append(rec)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[verify_all] wrote {out_path} ({len(results)} entries)")
        torch.cuda.empty_cache(); gc.collect()

    print("\n==================== summary ====================")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:16s}  ERROR  {r['error'][:60]}")
        else:
            print(f"  {r['name']:16s}  ppl_a={r['ppl_original']:.4f}  "
                  f"ppl_b={r['ppl_packed']:.4f}  diff={r['rel_diff']*100:.4f}%  "
                  f"({r['wall_s']}s)")


if __name__ == "__main__":
    main()
