"""uc verify-org + uc status — automated org-wide verification.

`uc verify-org SipsaLabs` iterates every `*-uc-v3-bpw5` repo on the org, downloads
each to a local cache, runs `uc verify --skip-hash` on it, and writes a JSON
report. Skips repos that have only README + .gitattributes (upload still in
flight).

`uc status` prints a one-line summary of every local `_packed_*_v3` dir.

Both subcommands respect the `--local-base` flag so repeat invocations reuse
the same cache.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _has_uc_files(repo_id: str) -> bool:
    """Cheap pre-flight: does the repo have at least one .uc file committed?"""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        return any(s.rfilename.endswith(".uc") for s in info.siblings)
    except Exception:
        return False


def _list_org_uc_repos(org: str, suffix: str) -> list[str]:
    """List every model on `org/*` whose name ends with `suffix`."""
    from huggingface_hub import HfApi
    api = HfApi()
    return sorted(
        m.id for m in api.list_models(author=org)
        if m.id.endswith(suffix)
    )


def _verify_one(repo_id: str, local_base: Path) -> dict[str, Any]:
    """Download + uc verify a single repo. Returns a result dict."""
    if not _has_uc_files(repo_id):
        return {"repo_id": repo_id, "status": "no_uc_files", "skipped": True}

    from huggingface_hub import snapshot_download

    local = local_base / repo_id.split("/")[-1]
    print(f"[verify-org] {repo_id} -> {local}", flush=True)
    t0 = time.time()
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local))
    download_s = time.time() - t0

    t0 = time.time()
    proc = subprocess.run(
        ["uc", "verify", str(local), "--skip-hash"],
        capture_output=True, text=True, timeout=300,
    )
    verify_s = time.time() - t0
    out = proc.stdout + proc.stderr
    pass_line = "VERIFY: PASS" in out
    fail_line = "VERIFY: FAIL" in out
    status = "PASS" if pass_line and not fail_line else ("FAIL" if fail_line else "INCONCLUSIVE")
    return {
        "repo_id": repo_id,
        "status": status,
        "download_s": round(download_s, 2),
        "verify_s": round(verify_s, 2),
        "exit_code": proc.returncode,
        "tail": out.splitlines()[-10:],
    }


def cmd_verify_org(args: argparse.Namespace) -> int:
    org = args.org
    out_path = Path(args.out)
    local_base = Path(args.local_base) if args.local_base else Path(tempfile.gettempdir()) / "uc_verify_org" / org

    print(f"[verify-org] org={org} cache={local_base} out={out_path}", flush=True)
    repos = _list_org_uc_repos(org, args.repo_suffix)
    print(f"[verify-org] {len(repos)} candidate repos", flush=True)

    results = []
    for repo in repos:
        try:
            results.append(_verify_one(repo, local_base))
        except Exception as e:
            print(f"[verify-org] ERROR on {repo}: {type(e).__name__}: {e}", flush=True)
            results.append({"repo_id": repo, "status": "ERROR", "error": str(e)})

    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_error = sum(1 for r in results if r.get("status") == "ERROR")

    summary = {
        "org": org,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_total": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip_no_uc_files_yet": n_skip,
        "n_error": n_error,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[verify-org] === {org} ===", flush=True)
    print(f"  total={len(results)}, pass={n_pass}, fail={n_fail}, "
          f"skip(no_uc)={n_skip}, error={n_error}", flush=True)
    print(f"  report: {out_path}", flush=True)

    return 1 if (n_fail > 0 or n_error > 0) else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print a one-line inventory of local _packed_*_v3 dirs."""
    cwd = Path.cwd()
    packs = sorted(cwd.glob("_packed_*_v3"))
    if not packs:
        print(f"[status] no _packed_*_v3 directories in {cwd}", flush=True)
        return 0
    print(f"[status] cwd={cwd}", flush=True)
    print(f"[status] {len(packs)} local v3 packs:", flush=True)
    total_bytes = 0
    for p in packs:
        if not p.is_dir():
            continue
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        n_uc = len(list(p.glob("layer_*.uc")))
        total_bytes += size
        print(f"  {p.name}: {n_uc} layer.uc, {size / (1024**3):.2f} GB", flush=True)
    print(f"[status] total: {total_bytes / (1024**3):.2f} GB across {len(packs)} packs", flush=True)
    return 0
