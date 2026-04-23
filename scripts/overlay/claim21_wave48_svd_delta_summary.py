"""claim21_wave48_svd_delta_summary.py -- aggregator."""
from __future__ import annotations

import json
import os
from pathlib import Path


PAIRS = [
    ("olmo2_1b",    "results/claim21_wave48_svd_delta_olmo2_1b.json"),
    ("smollm2_1.7b","results/claim21_wave48_svd_delta_smollm2_1.7b.json"),
    ("qwen3_1.7b",  "results/claim21_wave48_svd_delta_qwen3_1.7b.json"),
]


def main() -> None:
    root = Path(".")
    per_model: dict = {}
    tot_params = 0
    tot_lr_params = 0
    tot_bf16 = 0
    tot_br_bf16 = 0
    tot_br_lr  = 0
    relerr_list: list[tuple[int, float]] = []
    frac = None

    print()
    print("=" * 100)
    print("CLAIM 21 WAVE 48 SVD-LOWRANK DELTA COHORT SUMMARY")
    print("=" * 100)
    hdr = (f"{'model':<16} {'params':>14} {'lr_params':>14} {'lr_frac':>8} "
           f"{'br(bf16Δ)':>14} {'br(lowrank)':>14} {'ratio':>8} {'relerr':>10}")
    print(hdr)
    print("-" * len(hdr))

    for name, path in PAIRS:
        p = root / path
        if not p.exists():
            print(f"  MISSING: {path}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if frac is None:
            frac = d.get("rank_frac")
        nparams    = d["n_params"]
        nlr        = d["n_params_lowrank"]
        br_bf16    = d["brotli_11_bytes"]["bf16_delta"]
        br_lr      = d["brotli_11_bytes"]["lowrank_bf16_U_Vt"]
        relerr     = d["rel_frobenius_reconstruction_error"]
        lr_frac    = d["lowrank_param_fraction"]
        ratio      = d["ratio_bf16_delta_over_lowrank"]
        per_model[name] = {
            "base_repo":  d["base_repo"],
            "ft_repo":    d["ft_repo"],
            "n_params":   nparams,
            "n_params_lowrank": nlr,
            "lowrank_param_fraction": lr_frac,
            "brotli11_bytes_bf16_delta": br_bf16,
            "brotli11_bytes_lowrank":    br_lr,
            "rel_frobenius_err":         relerr,
            "ratio_bf16_over_lowrank":   ratio,
        }
        tot_params += nparams
        tot_lr_params += nlr
        tot_br_bf16 += br_bf16
        tot_br_lr  += br_lr
        relerr_list.append((nparams, relerr))
        print(f"{name:<16} {nparams:>14,} {nlr:>14,} {lr_frac*100:>7.2f}% "
              f"{br_bf16:>14,} {br_lr:>14,} {ratio:>7.2f}x {relerr:>10.3e}")

    print("-" * len(hdr))
    if tot_br_lr == 0:
        print("(no pairs parsed)")
        return

    def weighted(lst):
        num = sum(w * v for w, v in lst)
        den = sum(w for w, _ in lst)
        return num / den if den else 0.0

    cohort = {
        "n_pairs": len(per_model),
        "rank_frac": frac,
        "n_params_total": tot_params,
        "n_params_lowrank_total": tot_lr_params,
        "lowrank_param_fraction": tot_lr_params / tot_params,
        "brotli11_bf16_delta_total": tot_br_bf16,
        "brotli11_lowrank_total":   tot_br_lr,
        "ratio_bf16_over_lowrank":  tot_br_bf16 / tot_br_lr,
        "weighted_rel_frob_err":    weighted(relerr_list),
    }

    print()
    print(f"COHORT (rank_frac={frac}):")
    print(f"  total delta params         = {tot_params:,}")
    print(f"  total lowrank params       = {tot_lr_params:,}  "
          f"({100.0 * cohort['lowrank_param_fraction']:.2f}%)")
    print(f"  br(bf16 Δ) / br(lowrank)   = "
          f"{cohort['ratio_bf16_over_lowrank']:.3f}x")
    print(f"  params-weighted relerr     = "
          f"{cohort['weighted_rel_frob_err']:.3e}")

    out_path = root / "results" / "claim21_wave48_svd_delta_summary.json"
    obj = {"per_model": per_model, "cohort": cohort}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
