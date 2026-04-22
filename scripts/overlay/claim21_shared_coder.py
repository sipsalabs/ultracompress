"""Claim-21 wave 25: shared-coder cross-entropy measurement.

Wave 22 showed pairwise histogram r=0.99995 on idx_delta and r=0.99155
on fp8, and r=0.435 on scale. That is a property of DISTRIBUTIONS.
This wave upgrades it to a property of CODERS: for each stream and
each "training" model M_t, build the 256-bin Laplace-smoothed pmf from
M_t's byte histogram, then measure the CROSS-ENTROPY H(M_e, M_t) =
-sum p_e(b) * log2 q_t(b) on every "evaluation" model M_e. The diagonal
is the self-entropy (Shannon floor). Off-diagonal excess
    delta_bpB = H(M_e, M_t) - H(M_e, M_e)
is the bpB tax a static frequency coder trained on M_t pays when
shipped to M_e. Small delta => universal coder works.

Reads: results/claim21_fp8_histogram_<model>_rho0.01.json  (wave 19/22)
Writes: results/claim21_shared_coder.{txt,json}
"""
from __future__ import annotations
import json, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES  = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
STREAMS = ["fp8", "idx_delta", "scale"]
RHO = 0.010

def load_hist(model: str, stream: str):
    p = RES / f"claim21_fp8_histogram_{model}_rho0.01.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return list(d["streams"][stream]["histogram"])

def pmf_from_counts(counts, smoothing=1.0):
    tot = sum(counts) + 256 * smoothing
    return [(c + smoothing) / tot for c in counts]

def cross_entropy_bpB(p_eval_counts, q_train_counts):
    # p from counts directly (no smoothing), q Laplace-smoothed so logs are defined.
    n_eval = sum(p_eval_counts)
    if n_eval == 0:
        return 0.0
    q = pmf_from_counts(q_train_counts, smoothing=1.0)
    H = 0.0
    for c, qi in zip(p_eval_counts, q):
        if c == 0: continue
        p = c / n_eval
        H -= p * math.log2(qi)
    return H

def self_entropy_bpB(counts):
    n = sum(counts)
    if n == 0: return 0.0
    H = 0.0
    for c in counts:
        if c == 0: continue
        p = c / n
        H -= p * math.log2(p)
    return H

def main():
    out = {
        "claim": 21, "experiment": "shared_coder_cross_entropy",
        "rho": RHO, "models": MODELS, "streams": STREAMS,
        "per_stream": {},
    }
    lines = []
    lines.append("Claim-21 shared-coder cross-entropy (wave 25)")
    lines.append("=" * 86)
    lines.append("")
    lines.append("For each stream, cell (e, t) = bpB paid when coding model e's")
    lines.append("bytes with a Laplace-smoothed pmf trained on model t's bytes.")
    lines.append("Diagonal = Shannon self-entropy (floor). Row max over")
    lines.append("off-diagonal - diagonal = worst-case bpB tax of a static")
    lines.append("universal coder trained on any single model.")
    lines.append("")
    lines.append(f"Cohort: n={len(MODELS)} at rho={RHO:.3f}")
    lines.append("")

    for stream in STREAMS:
        counts = {m: load_hist(m, stream) for m in MODELS}
        H_self = {m: self_entropy_bpB(counts[m]) for m in MODELS}
        M = len(MODELS)
        # H[e][t]
        H = [[0.0]*M for _ in range(M)]
        for i, me in enumerate(MODELS):
            for j, mt in enumerate(MODELS):
                H[i][j] = cross_entropy_bpB(counts[me], counts[mt])
        # Row-wise excess over self
        excess = [[H[i][j] - H[i][i] for j in range(M)] for i in range(M)]
        # Worst-case tax for each training model t = max over eval models e (e != t) of excess[e][t]
        tax_train = []
        for j, mt in enumerate(MODELS):
            worst = max(excess[i][j] for i in range(M) if i != j)
            mean  = sum(excess[i][j] for i in range(M) if i != j) / (M - 1)
            tax_train.append((mt, mean, worst))
        # Global worst
        worst_any = max(excess[i][j] for i in range(M) for j in range(M) if i != j)
        mean_any  = sum(excess[i][j] for i in range(M) for j in range(M) if i != j) / (M*(M-1))

        # Emit table
        lines.append(f"--- stream: {stream} ---")
        header = f"  {'eval \\ train':<22}" + "".join(f"{m:>14}" for m in MODELS) + f"{'self H':>12}"
        lines.append(header)
        for i, me in enumerate(MODELS):
            row = f"  {me:<22}"
            for j in range(M):
                mark = "*" if i == j else " "
                row += f"{mark}{H[i][j]:>13.4f}"
            row += f"{H_self[me]:>12.4f}"
            lines.append(row)
        lines.append("")
        lines.append(f"  Cross-entropy excess over self-entropy (delta bpB, off-diagonal):")
        lines.append(f"  {'train model':<22}{'mean delta bpB':>18}{'worst delta bpB':>20}")
        for mt, mn, wr in tax_train:
            lines.append(f"  {mt:<22}{mn:>18.5f}{wr:>20.5f}")
        lines.append(f"  global mean delta bpB (all off-diagonal) = {mean_any:.5f}")
        lines.append(f"  global worst delta bpB                   = {worst_any:.5f}")
        lines.append("")

        out["per_stream"][stream] = {
            "models": MODELS,
            "H_cross_bpB": H,
            "H_self_bpB": [H_self[m] for m in MODELS],
            "excess_bpB": excess,
            "per_train_model": [
                {"train": mt, "mean_delta_bpB": mn, "worst_delta_bpB": wr}
                for mt, mn, wr in tax_train
            ],
            "global_mean_excess_bpB": mean_any,
            "global_worst_excess_bpB": worst_any,
        }

    lines.append("=" * 86)
    lines.append("INTERPRETATION")
    lines.append("-" * 86)
    lines.append("- global worst delta bpB near 0 => a SINGLE static entropy")
    lines.append("  coder built on any one model is near-optimal on every model")
    lines.append("  for this stream. This is the operational form of the wave-22")
    lines.append("  distribution-shape universality result.")
    lines.append("- Large delta on scale confirms the wave-22 finding that scale")
    lines.append("  distributions are model-specific and need per-model tables.")
    lines.append("")

    txt = "\n".join(lines) + "\n"
    (RES / "claim21_shared_coder.txt").write_text(txt, encoding="utf-8")
    (RES / "claim21_shared_coder.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_shared_coder.txt")
    print("[wrote] results/claim21_shared_coder.json")

if __name__ == "__main__":
    main()
