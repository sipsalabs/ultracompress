"""claim21_sweep.py -- run entropy_code_overlay across all available models.

Produces results/claim21_sweep.json with one row per (model, rho) pair.
Requires teacher+v17hi artifacts already cached on disk.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRIPT = HERE / "entropy_code_overlay.py"
OUTDIR = REPO / "results"
OUTDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
from entropy_code_overlay import MODEL_CONFIGS  # noqa: E402

MODELS = [
    "qwen3_1.7b", "qwen3_8b",
    "smollm2_1.7b", "tinyllama", "olmo2_1b", "mistral_7b",
]
RHOS = [0.003, 0.03]

results = []
for m in MODELS:
    teacher_pt, v17_pt = MODEL_CONFIGS[m]
    if not (REPO / teacher_pt).exists() or not (REPO / v17_pt).exists():
        print(f"[skip] {m}: missing artifacts")
        continue
    for rho in RHOS:
        out = OUTDIR / f"claim21_entropy_{m}_rho{rho:g}.json"
        if out.exists():
            print(f"[skip] {out.name} exists")
        else:
            print(f"[run] {m} rho={rho}")
            rc = subprocess.call([sys.executable, str(SCRIPT),
                                  "--model", m, "--rho", str(rho),
                                  "--device", "cuda:0",
                                  "--out", str(out)])
            if rc != 0:
                print(f"  FAILED rc={rc}")
                continue
        results.append(json.loads(out.read_text()))

(OUTDIR / "claim21_sweep.json").write_text(json.dumps(results, indent=2))
print(f"[sweep] {len(results)} rows -> {OUTDIR/'claim21_sweep.json'}")
