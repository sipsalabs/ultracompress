"""claim21_histogram_correlation.py -- cross-model byte-histogram correlation.

Wave 19 established that the per-stream byte entropy H is nearly
identical across models (idx_delta range 0.16 pp, fp8 range 1.71 pp).
H is a scalar summary; two distributions can share H exactly while
being very different shapes. This diagnostic goes further: it
measures the *full 256-bin distribution* similarity across all
pairs of models, per stream.

For each stream and each ordered pair (model_a, model_b) with
a < b, we compute, on the normalised (prob-mass) histograms:

  - Pearson correlation coefficient r in [-1, 1]
  - L1 distance (total-variation, in [0, 1])
  - Jensen-Shannon divergence in bits (in [0, 1])
  - symmetric KL in bits (sum of two directions)

A high r with small TV / JSD across every model pair means the
byte distribution is an encoding-level property, not a model-level
property. A low r or large JSD would indicate model-specific
structure that the scalar H happens to summarise to a similar
value by coincidence.

Input: results/claim21_fp8_histogram_<model>_rho0.01.json  (wave 19)
Output: results/claim21_histogram_correlation.txt
         results/claim21_histogram_correlation.json
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RES = REPO / "results"

MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
STREAMS = ["fp8", "idx_delta", "scale"]


def load_hists(rho: float = 0.01):
    out = {}
    for m in MODELS:
        p = RES / f"claim21_fp8_histogram_{m}_rho{rho:.2f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        streams = d["streams"]
        rec = {}
        for s in STREAMS:
            hist = np.array(streams[s]["histogram"], dtype=np.float64)
            total = float(hist.sum())
            if total > 0:
                p_vec = hist / total
            else:
                p_vec = hist
            rec[s] = {
                "hist": hist,
                "prob": p_vec,
                "n_bytes": int(streams[s]["n_bytes"]),
                "H": float(streams[s]["shannon_bits_per_byte"]),
            }
        out[m] = rec
    return out


def pearson_r(p: np.ndarray, q: np.ndarray) -> float:
    pm = p - p.mean()
    qm = q - q.mean()
    denom = math.sqrt((pm * pm).sum() * (qm * qm).sum())
    if denom <= 0:
        return float("nan")
    return float((pm * qm).sum() / denom)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def sym_kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))
    return _kl(p, q) + _kl(q, p)


def main():
    data = load_hists(0.01)
    present = list(data.keys())
    lines = []
    out_json = {"claim": 21, "experiment": "histogram_correlation",
                "rho": 0.01, "models": present, "streams": {}}

    lines.append("Claim-21 cross-model byte-histogram correlation")
    lines.append("=" * 82)
    lines.append("")
    lines.append(
        "Wave 19 established nearly identical Shannon H across models."
    )
    lines.append(
        "This diagnostic compares the full 256-bin byte distributions"
    )
    lines.append("pairwise. High r + low TV + low JSD => the distribution")
    lines.append("is a property of the encoding, not the model.")
    lines.append("")
    lines.append(f"Cohort: n={len(present)} models at rho=0.010")
    lines.append(f"Models: {present}")
    lines.append("")

    for s in STREAMS:
        lines.append(f"--- stream: {s} ---")
        header = "  pair".ljust(34) + "  r        TV      JSD (bits)   symKL (bits)"
        lines.append(header)
        rs, tvs, jsds, skls = [], [], [], []
        pair_rows = []
        for a, b in combinations(present, 2):
            pa = data[a][s]["prob"]
            pb = data[b][s]["prob"]
            r = pearson_r(pa, pb)
            tv = tv_distance(pa, pb)
            jsd = js_divergence_bits(pa, pb)
            skl = sym_kl_bits(pa, pb)
            rs.append(r); tvs.append(tv); jsds.append(jsd); skls.append(skl)
            label = f"  {a} vs {b}"
            lines.append(
                f"{label.ljust(34)}  {r: .5f}  {tv: .4f}  {jsd: .5f}      {skl: .5f}"
            )
            pair_rows.append({
                "a": a, "b": b, "pearson_r": r,
                "tv_distance": tv, "js_divergence_bits": jsd,
                "symmetric_kl_bits": skl,
            })
        if rs:
            rs_arr = np.array(rs); tvs_arr = np.array(tvs)
            jsds_arr = np.array(jsds); skls_arr = np.array(skls)
            lines.append("")
            lines.append(
                f"  pairs n={len(rs)}  "
                f"r mean={rs_arr.mean():.5f} min={rs_arr.min():.5f}  "
                f"TV mean={tvs_arr.mean():.4f} max={tvs_arr.max():.4f}  "
                f"JSD mean={jsds_arr.mean():.5f} max={jsds_arr.max():.5f}"
            )
        lines.append("")
        out_json["streams"][s] = {
            "n_pairs": len(rs),
            "pearson_r": {
                "mean": float(np.mean(rs)) if rs else None,
                "min": float(np.min(rs)) if rs else None,
                "max": float(np.max(rs)) if rs else None,
            },
            "tv_distance": {
                "mean": float(np.mean(tvs)) if tvs else None,
                "min": float(np.min(tvs)) if tvs else None,
                "max": float(np.max(tvs)) if tvs else None,
            },
            "js_divergence_bits": {
                "mean": float(np.mean(jsds)) if jsds else None,
                "max": float(np.max(jsds)) if jsds else None,
            },
            "symmetric_kl_bits": {
                "mean": float(np.mean(skls)) if skls else None,
                "max": float(np.max(skls)) if skls else None,
            },
            "pairs": pair_rows,
        }

    lines.append("=" * 82)
    lines.append("INTERPRETATION")
    lines.append("-" * 82)
    lines.append(
        "- Pearson r near +1 = models share byte-frequency shape."
    )
    lines.append(
        "- JSD near 0 = distributions are informationally close."
    )
    lines.append(
        "- If both hold across all pairs, scalar-H universality (wave 19)"
    )
    lines.append(
        "  is upgraded to full distribution-shape universality: a single"
    )
    lines.append(
        "  entropy-coder table trained on ANY model will be near-optimal"
    )
    lines.append(
        "  on EVERY model for this stream."
    )
    lines.append("")

    out_txt = RES / "claim21_histogram_correlation.txt"
    out_json_path = RES / "claim21_histogram_correlation.json"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"[wrote] {out_txt.relative_to(REPO)}")
    print(f"[wrote] {out_json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
