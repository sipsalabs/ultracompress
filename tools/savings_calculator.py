#!/usr/bin/env python3
"""UltraCompress savings calculator.

v0.1 quoteable: storage and egress savings.
v0.2 planning only: steady-state inference-memory savings (the v0.1 reference
loader inflates compressed weights at load time, so v0.1 inference memory
matches the inflated form, not the on-disk form).

Estimates storage, bandwidth, and (v0.2-projected) inference-memory savings for
a model fleet when migrating from FP16 / NF4 / int8 baselines to UltraCompress
lossless 5-bit patent-pending compression reference artifacts.

Usage:
    python savings_calculator.py
    python savings_calculator.py --params 1.7e9 --models 100
    python savings_calculator.py --json

For sales conversations: quote the storage + egress columns directly. Treat the
v0.2 GPU memory ceiling column as a planning estimate, not a v0.1 deliverable.
The v0.2 native lossless 5-bit kernel path ships in Q3 2026, gated on patent
prosecution timing.

Adjust customer-specific cost assumptions before quoting; the defaults below
are conservative public estimates as of April 2026.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict


# ---- Tunable per-customer assumptions ----

# Storage (S3 Standard us-east-1, April 2026)
STORAGE_USD_PER_GB_MONTH = 0.023

# Egress bandwidth (CloudFront US/EU, April 2026)
EGRESS_USD_PER_GB = 0.085

# H100 / A100 inference memory price proxy (USD per GB-hr; conservative)
GPU_MEMORY_USD_PER_GB_HOUR = 0.012


# ---- Compression baselines (bytes per parameter) ----
BASELINES = {
    "fp16": 2.000,
    "int8 (bitsandbytes)": 1.000,
    "NF4 (bitsandbytes)": 0.500,
    "HQQ 4-bit g64": 0.563,
    "UltraCompress 5 bpw (lossless)": 0.625,  # 5 bits / 8 bits per byte
}

ULTRACOMPRESS_BPP = BASELINES["UltraCompress 5 bpw (lossless)"]


@dataclass
class FleetSavings:
    """Per-baseline savings vs UltraCompress 5 bpw (lossless)."""

    baseline_name: str
    baseline_bytes_per_param: float
    fleet_baseline_gb: float
    fleet_uc_gb: float
    storage_savings_usd_year: float
    egress_savings_usd_per_full_redownload: float
    v02_inference_memory_ceiling_usd_year_per_replica: float
    compression_ratio: float


def compute_savings(params_per_model: float, model_count: int,
                    pulls_per_model_per_year: int = 12,
                    inference_replicas: int = 1) -> dict:
    """Compute storage + bandwidth + inference-memory savings vs each baseline.

    Args:
        params_per_model: Number of trainable parameters per model (e.g., 1.7e9).
        model_count: How many distinct model variants in the fleet.
        pulls_per_model_per_year: Egress events per model per year (deploys, updates, CI).
        inference_replicas: Number of in-memory replicas across the fleet at any time.
    """
    total_params = params_per_model * model_count

    fleet_uc_bytes = total_params * ULTRACOMPRESS_BPP
    fleet_uc_gb = fleet_uc_bytes / (1024 ** 3)

    rows: list[FleetSavings] = []
    for name, bpp in BASELINES.items():
        if name == "UltraCompress 5 bpw (lossless)":
            continue
        fleet_baseline_bytes = total_params * bpp
        fleet_baseline_gb = fleet_baseline_bytes / (1024 ** 3)
        delta_gb = fleet_baseline_gb - fleet_uc_gb

        storage_savings_year = delta_gb * STORAGE_USD_PER_GB_MONTH * 12
        egress_savings_per_pull = delta_gb * EGRESS_USD_PER_GB
        egress_savings_year = egress_savings_per_pull * pulls_per_model_per_year
        inference_savings_year = (
            delta_gb * GPU_MEMORY_USD_PER_GB_HOUR * 24 * 365 * inference_replicas
        )

        rows.append(FleetSavings(
            baseline_name=name,
            baseline_bytes_per_param=bpp,
            fleet_baseline_gb=round(fleet_baseline_gb, 2),
            fleet_uc_gb=round(fleet_uc_gb, 2),
            storage_savings_usd_year=round(storage_savings_year, 2),
            egress_savings_usd_per_full_redownload=round(egress_savings_per_pull, 2),
            v02_inference_memory_ceiling_usd_year_per_replica=round(inference_savings_year, 2),
            compression_ratio=round(bpp / ULTRACOMPRESS_BPP, 2),
        ))

    return {
        "inputs": {
            "params_per_model": params_per_model,
            "model_count": model_count,
            "pulls_per_model_per_year": pulls_per_model_per_year,
            "inference_replicas": inference_replicas,
            "total_params": total_params,
        },
        "ultracompress": {
            "bytes_per_param": ULTRACOMPRESS_BPP,
            "fleet_gb": round(fleet_uc_gb, 2),
        },
        "vs_baselines": [asdict(r) for r in rows],
        "assumptions": {
            "storage_usd_per_gb_month": STORAGE_USD_PER_GB_MONTH,
            "egress_usd_per_gb": EGRESS_USD_PER_GB,
            "gpu_memory_usd_per_gb_hour": GPU_MEMORY_USD_PER_GB_HOUR,
        },
        "notes": [
            "Storage and egress savings reflect the on-disk artifact size (5 bpw lossless vs the baseline). "
            "Continuous inference-time memory savings are NOT a current deliverable — the reference loader "
            "inflates compressed weights at load time, so steady-state inference memory matches the inflated form. "
            "The 'inference_memory_savings_usd_year_per_replica' field below is therefore a v0.2 (Q3 2026) projection, "
            "gated on the quantized-runtime kernel path shipping with native lossless 5-bit inference. Use it as a v0.2 "
            "ceiling estimate, not a current quote.",
            "Storage assumes a single canonical artifact per model. Multi-region replication multiplies savings linearly.",
            "Egress assumes one full redownload per pull event. Differential / patch updates further reduce egress.",
            "Customers with custom storage tiers (Glacier, S3 IA, on-prem) should substitute their own per-GB rate.",
        ],
    }


def render_human_readable(result: dict) -> str:
    """Pretty-print for sales conversations."""
    lines = []
    inp = result["inputs"]
    lines.append(f"\n=== UltraCompress fleet savings ===\n")
    lines.append(f"Fleet: {inp['model_count']} models × {inp['params_per_model']:,.0f} params each")
    lines.append(f"Total parameters: {inp['total_params']:,.0f}\n")
    lines.append(f"UltraCompress fleet on disk: {result['ultracompress']['fleet_gb']:.2f} GB\n")
    lines.append(f"{'Baseline':<28} {'Ratio':>8} {'Fleet GB':>10} {'Storage/yr':>14} {'Egress/pull':>14} {'v0.2 GPU ceil/yr':>18}")
    lines.append("-" * 96)
    for row in result["vs_baselines"]:
        lines.append(
            f"{row['baseline_name']:<28} "
            f"{row['compression_ratio']:>7.2f}x "
            f"{row['fleet_baseline_gb']:>10.2f} "
            f"${row['storage_savings_usd_year']:>13,.2f} "
            f"${row['egress_savings_usd_per_full_redownload']:>13,.2f} "
            f"${row['v02_inference_memory_ceiling_usd_year_per_replica']:>17,.2f}"
        )
    lines.append("\n  (v0.1 quoteable: Storage/yr and Egress/pull. v0.2 GPU ceiling is a planning estimate only.)")
    lines.append("\nAssumptions:")
    for k, v in result["assumptions"].items():
        lines.append(f"  {k}: ${v}")
    lines.append("\nNotes:")
    for note in result["notes"]:
        lines.append(f"  - {note}")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="UltraCompress savings calculator.")
    p.add_argument("--params", type=float, default=7e9,
                   help="Parameters per model (default: 7e9 = 7B)")
    p.add_argument("--models", type=int, default=100,
                   help="Number of distinct models in the fleet (default: 100)")
    p.add_argument("--pulls-per-year", type=int, default=12,
                   help="Pulls / redownloads per model per year (default: 12)")
    p.add_argument("--replicas", type=int, default=1,
                   help="In-memory replicas at any time (default: 1)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of human-readable.")
    args = p.parse_args()

    result = compute_savings(
        params_per_model=args.params,
        model_count=args.models,
        pulls_per_model_per_year=args.pulls_per_year,
        inference_replicas=args.replicas,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_human_readable(result))


if __name__ == "__main__":
    main()
