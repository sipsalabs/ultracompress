"""fit_v17_hifi.py -- Higher-fidelity Claim 16 fits at boosted codebook capacity.

The 2.40 bpw baseline fits use role_K = (K1=2048, K2=256) per role (19 bits
per D=8 group). This driver refits the same models at role_K = (4096, 1024)
(22 bits per group, ~2.80 bpw on disk) to demonstrate the bpw-vs-retention
dial that Claim 16 provides. All other knobs (D=8, alpha=0.25, iters=6,
beam=8, rotation seed) are held fixed -- only the codebook capacity changes.

Only the 4 small models (<=2B) are fit here to keep wall under an hour;
7B/8B can use the same driver with --only mistral or --only qwen3_8b.
"""
from __future__ import annotations
import argparse, json, os, time, traceback
import torch

from compress_v17 import v17_compress

ROLE_K_HIFI = {
    "q_proj":    (4096, 1024),
    "k_proj":    (4096, 1024),
    "v_proj":    (4096, 1024),
    "o_proj":    (8192, 2048),   # same 2x-on-o_proj asymmetric upgrade as baseline
    "gate_proj": (4096, 1024),
    "up_proj":   (4096, 1024),
    "down_proj": (4096, 1024),
}

MODELS = [
    # (short_name, teacher_cache, act_pt, out_fit)
    ("olmo2",     "olmo2_1b_cache.pt",       "v17_activations_olmo2.pt",     "v17hi_fit_olmo2.pt"),
    ("tinyllama", "tinyllama_1.1b_cache.pt", "v17_activations_tinyllama.pt", "v17hi_fit_tinyllama.pt"),
    ("qwen3_1.7b","qwen3_1.7b_cache.pt",     "v17_activations.pt",           "v17hi_fit_qwen3_1.7b.pt"),
    ("smollm2",   "smollm2_1.7b_cache.pt",   "v17_activations_smollm2.pt",   "v17hi_fit_smollm2.pt"),
    ("mistral",   "mistral_7b_v0.3_cache.pt","v17_activations_mistral.pt",   "v17hi_fit_mistral.pt"),
    ("qwen3_8b",  "qwen3_8b_cache.pt",       "v17_activations_8b.pt",        "v17hi_fit_8b.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="comma-separated short names to fit")
    ap.add_argument("--summary", default="v17hi_fit_summary.json")
    args = ap.parse_args()

    wanted = set(s.strip() for s in args.only.split(",") if s.strip()) or None
    summary = []
    if os.path.exists(args.summary):
        with open(args.summary) as f:
            summary = json.load(f)
    done = {r["name"] for r in summary if "error" not in r}

    for name, teacher, act, out in MODELS:
        if wanted and name not in wanted:
            continue
        if name in done and os.path.exists(out):
            print(f"[skip] {name}")
            continue
        print(f"\n====== {name} ({teacher}) -> {out} ======", flush=True)
        t0 = time.time()
        try:
            r = v17_compress(teacher, act, ROLE_K_HIFI, args.D,
                             alpha=args.alpha, iters=args.iters, beam=args.beam,
                             device=args.device)
            r["wall_sec"] = time.time() - t0
            torch.save(r, out)
            rec = {
                "name": name, "out": out,
                "global_bpw": float(r["global_bpw"]),
                "overhead_bpw": float(r["overhead_bpw"]),
                "total_bpw": float(r["global_bpw"]) + float(r["overhead_bpw"]),
                "rel_w_final_mean": float(r["rel_w_final_mean"]),
                "rel_w_final_max":  float(r["rel_w_final_max"]),
                "wall_sec": float(r["wall_sec"]),
                "role_K": {k: list(v) for k, v in ROLE_K_HIFI.items()},
                "D": args.D, "alpha": args.alpha,
            }
            print(f"[{name}] DONE bpw={rec['total_bpw']:.4f} "
                  f"rel_w mean={rec['rel_w_final_mean']:.4f} "
                  f"max={rec['rel_w_final_max']:.4f} "
                  f"wall={rec['wall_sec']:.0f}s", flush=True)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            rec = {"name": name, "error": repr(ex)}
        summary = [s for s in summary if s["name"] != name] + [rec]
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
