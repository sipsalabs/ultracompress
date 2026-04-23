"""Wave-47 auto-finisher (mirror of wave46_finisher)."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PAIRS = [
    ("olmo",  ROOT / "results" / "claim21_wave47_mxfp4_delta_olmo2_1b.json"),
    ("smol",  ROOT / "results" / "claim21_wave47_mxfp4_delta_smollm2_1.7b.json"),
    ("qwen3", ROOT / "results" / "claim21_wave47_mxfp4_delta_qwen3_1.7b.json"),
]
LOG = ROOT / "logs" / "wave47_finisher.log"


def stable(p: Path) -> int:
    return p.stat().st_size if p.exists() else -1


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg)


def wait_for_all():
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
        parts = "  ".join(f"{name}={cur[p]:>12}" for name, p in PAIRS)
        log(f"{time.strftime('%H:%M:%S')}  {parts}  stable_rounds={stable_rounds}\n")
        time.sleep(60)


def run(cmd, cwd=ROOT):
    log(f"\n$ {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log(res.stdout + "\n")
    if res.stderr:
        log("[stderr]\n" + res.stderr + "\n")
    log(f"[returncode={res.returncode}]\n")
    if res.returncode != 0:
        (ROOT / "wave47_FAILED.flag").write_text(
            f"cmd={cmd}\nreturncode={res.returncode}\n", encoding="utf-8")
        sys.exit(res.returncode)
    return res


def main():
    LOG.parent.mkdir(exist_ok=True)
    log(f"=== wave47 finisher started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    wait_for_all()
    log(f"\n[{time.strftime('%H:%M:%S')}] all JSONs stable; running summary\n")
    run([sys.executable, r"scripts\overlay\claim21_wave47_mxfp4_delta_summary.py"])

    try:
        summ = json.loads(
            (ROOT / "results" / "claim21_wave47_mxfp4_delta_summary.json")
            .read_text(encoding="utf-8"))
        c = summ["cohort"]
        log("\n=== WAVE 47 COHORT HEADLINE ===\n")
        log(f"  block size          : {c['block_size']}\n")
        log(f"  params              : {c['n_params_total']:,}\n")
        log(f"  blocks              : {c['n_blocks_total']:,}\n")
        log(f"  bf16 brotli11 bytes : {c['brotli11_bf16_total']:,}\n")
        log(f"  int8 brotli11 bytes : {c['brotli11_int8_total']:,}\n")
        log(f"  int4 brotli11 bytes : {c['brotli11_int4_total']:,}\n")
        log(f"  bf16/int8 ratio     : {c['ratio_bf16_over_int8']:.3f}x\n")
        log(f"  bf16/int4 ratio     : {c['ratio_bf16_over_int4']:.3f}x\n")
        log(f"  int8 relerr (pw)    : {c['weighted_rel_frob_err_int8']:.4e}\n")
        log(f"  int4 relerr (pw)    : {c['weighted_rel_frob_err_int4']:.4e}\n")
    except Exception as e:
        log(f"parse err: {e}\n")

    (ROOT / "wave47_ready.flag").write_text(
        "ready — run claim21_wave47_mxfp4_delta_patent_block.py and paste "
        "into PATENT_CLAIMS.md, then commit\n", encoding="utf-8")
    log(f"[{time.strftime('%H:%M:%S')}] wave47_ready.flag written\n")


if __name__ == "__main__":
    main()
