"""Wave-45 auto-finisher.

Polls results/claim21_finetune_delta_{olmo2_1b,smollm2_1.7b}.json every
60s. When both exist and are stable (unchanged size across two polls),
runs the summary script and the master verifier, then stages a commit
and writes wave45_ready.flag.  Does NOT push.  Does NOT edit
PATENT_CLAIMS.md (that is done manually with the confirmed numbers).
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

# Resolve repo root from this file's location so the daemon is portable.
ROOT = Path(__file__).resolve().parent.parent.parent
PAIRS = [
    ("olmo",  ROOT / "results" / "claim21_finetune_delta_olmo2_1b.json"),
    ("smol",  ROOT / "results" / "claim21_finetune_delta_smollm2_1.7b.json"),
    ("qwen3", ROOT / "results" / "claim21_finetune_delta_qwen3_1.7b.json"),
]


def stable(p: Path) -> int:
    if not p.exists():
        return -1
    return p.stat().st_size


def wait_for_both():
    prev = {p: -1 for _, p in PAIRS}
    stable_rounds = 0
    while True:
        cur = {p: stable(p) for _, p in PAIRS}
        all_exist = all(v > 0 for v in cur.values())
        all_stable = all(cur[p] == prev[p] for p in cur)
        if all_exist and all_stable and stable_rounds >= 1:
            return
        if all_exist and cur == prev:
            stable_rounds += 1
        else:
            stable_rounds = 0
        prev = cur
        with open(ROOT / "logs" / "wave45_finisher.log", "a",
                  encoding="utf-8") as f:
            parts = "  ".join(f"{name}={cur[p]:>12}"
                              for name, p in PAIRS)
            f.write(f"{time.strftime('%H:%M:%S')}  {parts}  "
                    f"stable_rounds={stable_rounds}\n")
        time.sleep(60)


def run(cmd, cwd=ROOT):
    with open(ROOT / "logs" / "wave45_finisher.log", "a",
              encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    with open(ROOT / "logs" / "wave45_finisher.log", "a",
              encoding="utf-8") as f:
        f.write(res.stdout + "\n")
        if res.stderr:
            f.write("[stderr]\n" + res.stderr + "\n")
        f.write(f"[returncode={res.returncode}]\n")
    if res.returncode != 0:
        (ROOT / "wave45_FAILED.flag").write_text(
            f"cmd={cmd}\nreturncode={res.returncode}\n",
            encoding="utf-8")
        sys.exit(res.returncode)
    return res


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    # Append, don't truncate -- preserve forensic context across restarts.
    with open(ROOT / "logs" / "wave45_finisher.log", "a",
              encoding="utf-8") as f:
        f.write(f"=== wave45 finisher started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    wait_for_both()

    with open(ROOT / "logs" / "wave45_finisher.log", "a",
              encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] both JSONs stable; running summary\n")

    run([sys.executable, r"scripts\overlay\claim21_finetune_delta_summary.py"])
    run([sys.executable, r"scripts\overlay\claim21_master_verify.py",
         "--device", "cuda:0", "--rho", "0.010",
         "--out", r"results\claim21_master_verify.json"])
    run([sys.executable, r"scripts\overlay\claim21_master_verify_summary.py"])

    # Parse and log the headline numbers for easy grep.
    try:
        summ = json.loads((ROOT / "results" /
                           "claim21_finetune_delta_summary.json")
                          .read_text(encoding="utf-8"))
        c = summ["cohort"]
        with open(ROOT / "logs" / "wave45_finisher.log", "a",
                  encoding="utf-8") as f:
            f.write("\n=== WAVE 45 COHORT HEADLINE ===\n")
            f.write(f"  params         : {c['n_params_total']:,}\n")
            f.write(f"  bf16 total     : {c['bf16_bytes_total']:,}\n")
            f.write(f"  brotli11(base) : {c['brotli11_base_total']:,}\n")
            f.write(f"  brotli11(ft)   : {c['brotli11_ft_total']:,}\n")
            f.write(f"  brotli11(delta): {c['brotli11_delta_total']:,}\n")
            f.write(f"  delta vs ft    : {c['delta_vs_ft_ratio']:.3f}x\n")
            f.write(f"  delta vs base  : {c['delta_vs_base_ratio']:.3f}x\n")
            f.write(f"  ft_bpB         : {c['ft_bpB_brotli11']:.4f}\n")
            f.write(f"  delta_bpB      : {c['delta_bpB_brotli11']:.4f}\n")
    except Exception as e:
        with open(ROOT / "logs" / "wave45_finisher.log", "a",
                  encoding="utf-8") as f:
            f.write(f"parse err: {e}\n")

    (ROOT / "wave45_ready.flag").write_text(
        "ready — run git add + commit manually after adding PATENT_CLAIMS "
        "wave-45 subsection\n", encoding="utf-8")

    with open(ROOT / "logs" / "wave45_finisher.log", "a",
              encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] DONE\n")


if __name__ == "__main__":
    main()
