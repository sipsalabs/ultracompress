"""
Compile all UltraCompress experimental results into a single structured JSON.
Reads checkpoints, training logs, and eval outputs to build a comprehensive record.
CPU-only — does not load models onto GPU.

Usage:
    python compile_results.py [--output results_compiled.json]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch


def scan_checkpoints(root: Path) -> list[dict]:
    """Find all checkpoint files and extract metadata."""
    results = []
    for ckpt_dir in sorted(root.glob("checkpoints*")):
        if not ckpt_dir.is_dir():
            continue
        for pt_file in sorted(ckpt_dir.glob("*.pt")):
            info: dict = {
                "path": str(pt_file.relative_to(root)),
                "directory": ckpt_dir.name,
                "filename": pt_file.name,
                "size_mb": round(pt_file.stat().st_size / 1024 / 1024, 1),
            }
            # Extract step from filename
            step_match = re.search(r"step(\d+)", pt_file.name)
            if step_match:
                info["step"] = int(step_match.group(1))
            elif "best" in pt_file.name:
                info["step"] = "best"

            # Load checkpoint metadata (CPU only)
            try:
                ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    for key in ["step", "t10", "t1", "loss", "best_t10", "config"]:
                        if key in ckpt:
                            info[f"ckpt_{key}"] = ckpt[key]
                    if "model_state_dict" in ckpt:
                        n_params = sum(
                            p.numel() for p in ckpt["model_state_dict"].values()
                        )
                        info["params"] = n_params
                        info["params_m"] = round(n_params / 1e6, 2)
            except Exception as e:
                info["load_error"] = str(e)

            results.append(info)
    return results


def parse_training_logs(root: Path) -> list[dict]:
    """Parse all .log files for training metrics."""
    results = []
    for log_file in sorted(root.glob("*_output.log")):
        experiment = log_file.stem.replace("_output", "")
        entries: list[dict] = []

        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Match step lines: Step  5000/100000: loss=41.3256  T1=32.0%  T10=61.4%
                step_match = re.search(
                    r"Step\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+T1=([\d.]+)%\s+T10=([\d.]+)%",
                    line,
                )
                if step_match:
                    entries.append({
                        "step": int(step_match.group(1)),
                        "total_steps": int(step_match.group(2)),
                        "loss": float(step_match.group(3)),
                        "t1": float(step_match.group(4)),
                        "t10": float(step_match.group(5)),
                    })

                # Match FINAL lines
                final_match = re.search(
                    r"FINAL:\s+T1=([\d.]+)%\s+T10=([\d.]+)%",
                    line,
                )
                if final_match:
                    entries.append({
                        "step": "final",
                        "t1": float(final_match.group(1)),
                        "t10": float(final_match.group(2)),
                    })

                # Match HellaSwag
                hs_match = re.search(
                    r"HellaSwag.*?(\d+\.?\d*)%",
                    line,
                )
                if hs_match:
                    entries.append({
                        "step": "hellaswag",
                        "hellaswag": float(hs_match.group(1)),
                    })

        if entries:
            best_t10 = max(
                (e["t10"] for e in entries if "t10" in e), default=0
            )
            best_t1 = max(
                (e["t1"] for e in entries if "t1" in e), default=0
            )
            results.append({
                "experiment": experiment,
                "log_file": log_file.name,
                "n_entries": len(entries),
                "best_t10": best_t10,
                "best_t1": best_t1,
                "entries": entries,
            })
    return results


def compile_known_results() -> dict:
    """Compile all known experimental results from the project history."""
    return {
        "scaling_experiments": [
            {
                "teacher": "Qwen3-0.6B",
                "data": "random",
                "steps": 15_000,
                "t1": 44,
                "t10": 56,
                "compression": "60x",
                "frr_params": "7.35M",
            },
            {
                "teacher": "Qwen3-0.6B",
                "data": "random",
                "steps": 50_000,
                "t10": 63,
                "compression": "60x",
                "frr_params": "7.35M",
            },
            {
                "teacher": "Qwen3-0.6B",
                "data": "random",
                "steps": 100_000,
                "t1": 48,
                "t10": 65,
                "compression": "60x",
                "frr_params": "7.35M",
            },
            {
                "teacher": "Qwen3-0.6B",
                "data": "real_text",
                "steps": 15_000,
                "t10": 60,
                "compression": "60x",
                "frr_params": "7.35M",
            },
            {
                "teacher": "Qwen3-1.7B",
                "data": "random",
                "steps": 15_000,
                "t10": 61,
                "compression": "52x",
                "frr_params": "29.4M",
            },
            {
                "teacher": "Qwen3-1.7B",
                "data": "random",
                "steps": 100_000,
                "t10": 67,
                "compression": "52x",
                "frr_params": "29.4M",
            },
            {
                "teacher": "Qwen3-1.7B",
                "data": "real_text",
                "steps": 10_000,
                "t1": 47,
                "t10": 62.4,
                "compression": "52x",
                "frr_params": "29.4M",
                "note": "best real-text result, 5x faster than random",
            },
        ],
        "e2e_compression": [
            {
                "pipeline": "FRR only (FP32)",
                "t1": 36,
                "t10": 55,
                "compression": "60x",
                "size_mb": 29,
            },
            {
                "pipeline": "FRR + Q8",
                "t1": 36,
                "t10": 55,
                "compression": "240x",
                "size_mb": 12,
                "quality_drop": "-0.2%",
            },
            {
                "pipeline": "FRR + Q4",
                "t1": 32,
                "t10": 56,
                "compression": "479x",
                "size_mb": 7,
                "quality_drop": "+0.8%",
            },
            {
                "pipeline": "FRR + Q2 + entropy",
                "t1": 35,
                "t10": 53,
                "compression": "959x",
                "size_mb": 1.8,
                "quality_drop": "-1.5%",
            },
        ],
        "benchmarks": {
            "teacher_0.6b": {
                "wikitext2_ppl": 1202.8,
                "hellaswag": 29.0,
            },
            "frr_100k_60x": {
                "wikitext2_ppl": 1521.1,
                "hellaswag": 26.5,
                "hellaswag_retention": 91.4,
            },
            "frr_100k_300sample": {
                "hellaswag": 25.0,
                "hellaswag_retention": 83.3,
                "note": "300-sample eval, more reliable",
            },
        },
        "inference_speed": {
            "device": "RTX 5090",
            "results": [
                {"seq_len": 32, "teacher_tps": 613, "frr_tps": 2073, "speedup": 3.38},
                {
                    "seq_len": 128,
                    "teacher_tps": 2624,
                    "frr_tps": 8041,
                    "speedup": 3.06,
                },
                {
                    "seq_len": 256,
                    "teacher_tps": 5223,
                    "frr_tps": 16403,
                    "speedup": 3.14,
                },
            ],
        },
        "ablation": {
            "per_layer_modulation": "+21% T10 (41% → 62%)",
            "real_text_vs_random": "+4% T10 at 15K steps",
            "hidden_state_supervision": "-6% T10 (harmful for FRR)",
            "temperature_annealing": "oscillation ±3-5% at 1.7B, partially noise",
            "selective_student_trustgate": "-8.7% at 3K, -1.9% at 6K (harmful)",
            "dendritic_neurons": "-6% T10",
            "multi_block_3": "-5% T10, no benefit over single block",
            "lora_adapters": "+3% T10",
        },
        "all_approaches": [
            {"name": "FRR (1 shared block)", "t1": 44, "t10": 62, "compression": "42x"},
            {"name": "FRR + Q2 E2E", "t1": 35, "t10": 53, "compression": "959x"},
            {"name": "HWI (holographic)", "t1": 35, "t10": 57, "compression": "76x"},
            {"name": "Swarm (16 experts)", "t1": 32, "t10": 52},
            {"name": "Program synthesis", "t1": 38, "t10": 58},
            {
                "name": "Genome + hidden sup",
                "t1": 44,
                "t10": 63,
                "compression": "37x",
            },
            {"name": "FRR from scratch", "accuracy": 80.7},
            {"name": "BitNet ternary", "t1": 36, "t10": 57, "compression": "6x"},
            {"name": "PHM (hypercomplex)", "t1": 30, "t10": 50, "compression": "168x"},
            {
                "name": "Ultimate pipeline",
                "cosine_sim": 0.994,
                "compression": "Q2",
            },
        ],
        "records": {
            "best_t1_1.7b": {"value": 47, "how": "1.7B real text 10K steps"},
            "best_t10_1.7b": {"value": 67, "how": "1.7B random tokens 100K steps"},
            "best_t10_0.6b": {"value": 65, "how": "0.6B random tokens 100K steps"},
            "best_hellaswag_retention": {"value": 83.3, "how": "0.6B FRR 100K, 300-sample"},
            "best_compression": {"value": "959x", "how": "FRR + Q2 + entropy"},
            "best_param_efficiency": {
                "value": "5.5x",
                "how": "FRR 25.5% HS at 7.3M vs Standard 26.5% at 42M",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile all UltraCompress results")
    parser.add_argument(
        "--output",
        type=str,
        default="results_compiled.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning {root}...")

    # Gather all data
    checkpoints = scan_checkpoints(root)
    print(f"  Found {len(checkpoints)} checkpoints")

    logs = parse_training_logs(root)
    print(f"  Found {len(logs)} training logs")

    known = compile_known_results()

    # Compile
    compiled = {
        "compiled_at": datetime.now().isoformat(),
        "project_root": str(root),
        "checkpoints": checkpoints,
        "training_logs": logs,
        "known_results": known,
        "summary": {
            "total_checkpoints": len(checkpoints),
            "total_logs": len(logs),
            "best_t10_overall": 67,
            "best_compression": "959x",
            "teachers_tested": ["Qwen3-0.6B", "Qwen3-1.7B"],
            "teachers_ready": ["Qwen3-8B"],
        },
    }

    output_path = root / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, default=str)

    print(f"\nResults compiled to {output_path}")
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  Training logs: {len(logs)}")
    print(f"  Known results: {len(known)} categories")

    # Print summary
    print("\n--- ALL-TIME RECORDS ---")
    for name, record in known["records"].items():
        print(f"  {name}: {record['value']} ({record['how']})")

    # Print best from each log
    if logs:
        print("\n--- BEST FROM TRAINING LOGS ---")
        for log in sorted(logs, key=lambda x: x["best_t10"], reverse=True):
            print(f"  {log['experiment']}: T10={log['best_t10']}% T1={log['best_t1']}%")


if __name__ == "__main__":
    main()
