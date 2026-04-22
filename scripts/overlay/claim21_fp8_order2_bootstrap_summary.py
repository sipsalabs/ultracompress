"""Aggregator for wave-35 bootstrap two-pass coder (both contiguous and interleaved)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
RHO_TAG = "0.01"
ALPHAS = ["0.1", "0.01"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    d = load(RES / f"claim21_fp8_order2_bootstrap_rho{RHO_TAG}.json")
    rows = d["models"]

    br = {}
    oracle = {}
    for r in rows:
        m = r["model"]
        br[m] = float(load(RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json")["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        u = load(RES / f"claim21_fp8_order2_universal_rho{RHO_TAG}.json")
        for mr in u["models"]:
            if mr["test_model"] == m:
                oracle[m] = float(mr["by_alpha"]["0.01"]["oracle_static_bpB"])

    lines = []
    lines.append("Claim-21 wave 35: self-bootstrap order-2 coder fails to beat brotli-11")
    lines.append("=" * 124)
    lines.append("")
    lines.append("Tests whether a deployable coder can BOOTSTRAP order-2 counts from a")
    lines.append("fraction F of the fp8 stream (no side information) and code the")
    lines.append("remainder with the learned counts. Two sampling modes:")
    lines.append("  contiguous : first F of stream is the bootstrap (coded at brotli-11)")
    lines.append("  interleaved: every 1/F-th byte is the bootstrap (coded at brotli-11)")
    lines.append("Both modes charge brotli-11 on the bootstrap bytes. Combined rate =")
    lines.append("F * brotli + (1-F) * order2_tail. Oracle = full-stream static alpha=0.01.")
    lines.append("")

    def cohort_combined(mode, alpha, F):
        tN = 0
        s = 0.0
        for r in rows:
            n = int(r["n_bytes"])
            tN += n
            for e in r[mode]:
                if abs(e["fraction"] - F) < 1e-9:
                    tail = float(e["by_alpha"][alpha]["tail_bpB"])
                    s += (F * br[r["model"]] + (1 - F) * tail) * n
                    break
        return s / tN

    Fs = [e["fraction"] for e in rows[0]["bootstraps"]]
    tN = sum(int(r["n_bytes"]) for r in rows)
    sBR = sum(br[r["model"]] * int(r["n_bytes"]) for r in rows) / tN
    sOR = sum(oracle[r["model"]] * int(r["n_bytes"]) for r in rows) / tN

    for alpha in ALPHAS:
        lines.append(f"--- cohort combined rate @ alpha={alpha} (brotli-11={sBR:.4f}, oracle={sOR:.4f}) ---")
        hdr = f"  {'mode':<14}" + "".join(f"{f'F={F}':>12}" for F in Fs)
        lines.append(hdr)
        for mode in ("bootstraps", "interleaved"):
            label = "contiguous" if mode == "bootstraps" else "interleaved"
            line = f"  {label:<14}"
            for F in Fs:
                line += f"{cohort_combined(mode, alpha, F):>12.4f}"
            lines.append(line)
        lines.append("")

    lines.append("=" * 124)
    lines.append("INTERPRETATION")
    lines.append("-" * 124)
    lines.append(
        "- At EVERY fraction F in {0.05, 0.1, 0.25, 0.5, 0.75} and at BOTH alphas")
    lines.append(
        "  (0.1 and 0.01), the combined rate is WORSE than pure brotli-11. The best")
    lines.append(
        "  combined rate (alpha=0.1, F=0.75, contiguous) is ~6.64 bpB cohort -- still")
    lines.append(
        "  above brotli-11 shipping at 6.558 bpB.")
    lines.append(
        "- Contiguous vs interleaved bootstrap are within 0.01-0.03 bpB of each other")
    lines.append(
        "  at every (F, alpha) combination. This RULES OUT non-stationarity of the")
    lines.append(
        "  fp8 stream as the failure mode. The failure is fundamental: the 65,536 x")
    lines.append(
        "  256 order-2 context table requires essentially the FULL stream to populate")
    lines.append(
        "  adequately. With 50-75% of bytes the table has too many sparsely-covered")
    lines.append(
        "  contexts where smoothing dominates the coded length.")
    lines.append(
        "- IMPLICATION: wave-34's implied 'two-pass decode' shipping path does NOT")
    lines.append(
        "  work -- there is no way to code the payload at sub-brotli-11 rate using only")
    lines.append(
        "  order-2 counts derived from the payload itself, regardless of sampling order.")
    lines.append(
        "  The -0.155 bpB theoretical gap from wave 31 is REALIZABLE only by shipping")
    lines.append(
        "  the order-2 context table as side information (~32 MiB per model, 0.8% of")
    lines.append(
        "  a 4 GiB checkpoint, cheap in absolute terms) or by training a compact neural")
    lines.append(
        "  context model that emits the table from model metadata.")
    lines.append(
        "- CLAIM-21 CONTEXT-CODER DESIGN SPACE (waves 30-35 summary):")
    lines.append(
        "    wave 30: brotli-11 uses order >= 2 context (rules out order-1 coders)")
    lines.append(
        "    wave 31: Shannon H2 floor = 6.40 bpB, -0.155 bpB vs brotli-11")
    lines.append(
        "    wave 32: order-3 estimator sample-limited; order-2 is the safe floor")
    lines.append(
        "    wave 33: naive adaptive Laplace-1 fails by +0.50 bpB cohort")
    lines.append(
        "    wave 34: oracle static alpha=0.01 hits 94% of floor (-0.146 bpB)")
    lines.append(
        "             universal cross-model priors fail (plateau at +0.14 vs brotli)")
    lines.append(
        "    wave 35: self-bootstrap fails regardless of sampling order")
    lines.append(
        "  => sub-brotli-11 order-2 fp8 coder REQUIRES side-information priors.")
    lines.append("")

    out_txt = RES / "claim21_fp8_order2_bootstrap.txt"
    out_json = RES / "claim21_fp8_order2_bootstrap_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    summary = {"claim": 21, "wave": 35, "experiment": "fp8_order2_bootstrap_summary",
               "rho": float(RHO_TAG), "cohort_brotli11_bpB": sBR,
               "cohort_oracle_bpB": sOR, "per_alpha": {}}
    for alpha in ALPHAS:
        summary["per_alpha"][alpha] = {
            "contiguous": {str(F): cohort_combined("bootstraps", alpha, F) for F in Fs},
            "interleaved": {str(F): cohort_combined("interleaved", alpha, F) for F in Fs},
        }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
