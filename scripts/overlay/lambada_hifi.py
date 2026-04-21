"""lambada_hifi.py -- Evaluate higher-fidelity v17 fits on LAMBADA.

Parallel of lambada_all.py but pointing at the v17hi_fit_*.pt artefacts
produced by fit_v17_hifi.py. Reuses lambada_test_<model>.pt tokens cached
by the original run.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

# import the full pipeline
from lambada_all import tokenize_if_missing, run_one

MODELS = [
    # (display_name, hf_model_id, teacher_cache, v17hi_fit, tokens_path)
    ("OLMo-2-1B",      "allenai/OLMo-2-0425-1B",              "olmo2_1b_cache.pt",         "v17hi_fit_olmo2.pt",       "lambada_test_olmo2.pt"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "tinyllama_1.1b_cache.pt",   "v17hi_fit_tinyllama.pt",   "lambada_test_tinyllama.pt"),
    ("Qwen3-1.7B",     "Qwen/Qwen3-1.7B",                     "qwen3_1.7b_cache.pt",       "v17hi_fit_qwen3_1.7b.pt",  "lambada_test_qwen3.pt"),
    ("SmolLM2-1.7B",   "HuggingFaceTB/SmolLM2-1.7B",          "smollm2_1.7b_cache.pt",     "v17hi_fit_smollm2.pt",     "lambada_test_smollm2.pt"),
    ("Mistral-7B",     "mistralai/Mistral-7B-v0.3",           "mistral_7b_v0.3_cache.pt",  "v17hi_fit_mistral.pt",     "lambada_test_mistral.pt"),
    ("Qwen3-8B",       "Qwen/Qwen3-8B",                       "qwen3_8b_cache.pt",         "v17hi_fit_8b.pt",          "lambada_test_qwen3.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="lambada_hifi_results.json")
    args = ap.parse_args()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done = {r["name"] for r in results if "error" not in r}

    for name, mid, teacher, v17hi, tokens in MODELS:
        if not os.path.exists(v17hi):
            print(f"[skip] {name}: {v17hi} not present")
            continue
        if name in done:
            print(f"[skip] {name}: already in {args.out}")
            continue
        try:
            tokenize_if_missing(mid, tokens)
            rec = run_one(name, mid, teacher, v17hi, tokens, args.n, args.seq_len, args.device)
            rec["fit"] = v17hi
            rec["tier"] = "hifi"
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            rec = {"name": name, "model_id": mid, "error": repr(ex), "fit": v17hi}
        results = [r for r in results if r["name"] != name] + [rec]
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[lambada_hifi] wrote {args.out} ({len(results)})", flush=True)

    print("\n================ hifi summary ================")
    print(f"{'model':18s} {'teacher_ppl':>12s} {'v17hi_ppl':>10s} {'ratio':>7s} "
          f"{'tchT1':>7s} {'hiT1':>7s} {'T1_ret':>8s}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:16s}  ERROR")
            continue
        print(f"{r['name']:18s} {r['teacher_ppl']:>12.4f} {r['v17_ppl']:>10.4f} "
              f"{r['ppl_ratio']:>7.3f} {r['teacher_t1']*100:>6.2f}% {r['v17_t1']*100:>6.2f}% "
              f"{r['t1_ret']*100:>7.2f}%")


if __name__ == "__main__":
    main()
