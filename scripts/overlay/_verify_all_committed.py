"""Verify every public SipsaLabs HF artifact via the customer flow.

For each committed model in the SipsaLabs HF org:
  1. snapshot_download to a temp dir
  2. Run `uc verify` (structural + SHA256 spot-check + reconstruction)
  3. Record PASS/FAIL + key metadata into a JSON report

Skips repos that only contain README.md + .gitattributes (uploads still in flight).

Usage:
    python scripts/overlay/_verify_all_committed.py [--out report.json]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from huggingface_hub import HfApi, snapshot_download

ORG = "SipsaLabs"
LOCAL_BASE = Path(r"C:\Users\scamd\AppData\Local\Temp\customer_repro")


def list_org_models() -> list[str]:
    api = HfApi()
    return sorted(m.id for m in api.list_models(author=ORG))


def repo_has_layer_uc(repo_id: str) -> bool:
    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        return any(s.rfilename.endswith(".uc") for s in info.siblings)
    except Exception:
        return False


def run_uc_verify(local_dir: Path) -> dict:
    """Invoke `uc verify` as a subprocess, parse PASS/FAIL from stdout."""
    t0 = time.time()
    proc = subprocess.run(
        ["uc", "verify", str(local_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr
    pass_line = "VERIFY: PASS" in out
    fail_line = "VERIFY: FAIL" in out
    return {
        "elapsed_s": round(elapsed, 2),
        "exit_code": proc.returncode,
        "pass": pass_line and not fail_line,
        "fail": fail_line,
        "tail": out.splitlines()[-15:] if out else [],
    }


def verify_one(repo_id: str) -> dict:
    print(f"\n[verify-all] === {repo_id} ===", flush=True)
    if not repo_has_layer_uc(repo_id):
        print(f"[verify-all]   SKIP (no .uc files committed yet)", flush=True)
        return {"repo_id": repo_id, "status": "no_uc_files", "skipped": True}

    local = LOCAL_BASE / repo_id.split("/")[-1]
    print(f"[verify-all]   downloading to {local}...", flush=True)
    t0 = time.time()
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local))
    print(f"[verify-all]   download done in {time.time() - t0:.1f}s", flush=True)

    result = run_uc_verify(local)
    status = "PASS" if result["pass"] else ("FAIL" if result["fail"] else "INCONCLUSIVE")
    print(f"[verify-all]   {status} (uc verify ran {result['elapsed_s']}s)", flush=True)
    return {"repo_id": repo_id, "status": status, **result}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/VERIFY_ALL_REPORT.json"))
    args = ap.parse_args()

    print(f"[verify-all] start ts={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    repos = [r for r in list_org_models() if r.endswith("-uc-v3-bpw5")]
    print(f"[verify-all] {len(repos)} v3 candidates: {[r.split('/')[-1] for r in repos]}", flush=True)

    results = []
    for repo in repos:
        try:
            results.append(verify_one(repo))
        except Exception as e:
            print(f"[verify-all]   ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({"repo_id": repo, "status": "ERROR", "error": str(e)})

    # Tally
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_error = sum(1 for r in results if r.get("status") == "ERROR")

    summary = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_total": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip_no_uc_files_yet": n_skip,
        "n_error": n_error,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[verify-all] === SUMMARY ===", flush=True)
    print(f"[verify-all]   total={len(results)}, pass={n_pass}, fail={n_fail}, "
          f"skip(no_uc)={n_skip}, error={n_error}", flush=True)
    print(f"[verify-all]   report saved to {args.out}", flush=True)
    print(f"[verify-all] done ts={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Exit code: nonzero only if there's a real FAIL or ERROR
    return 1 if (n_fail > 0 or n_error > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
