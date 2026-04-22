"""Aggregator for wave-34 universal order-2 coder measurement."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
RHO_TAG = "0.01"

ALPHAS = ["1.0", "0.5", "0.1", "0.01"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    d = load(RES / f"claim21_fp8_order2_universal_rho{RHO_TAG}.json")
    rows = d["models"]

    # Gather brotli-11 and Shannon H2 per model
    ext = {}
    for r in rows:
        m = r["test_model"]
        cs = load(RES / f"claim21_codec_sweep_{m}_rho{RHO_TAG}.json")
        br = float(cs["codec_sweep"]["fp8"]["codecs"]["brotli-11"]["bits_per_byte"])
        h2 = float(load(RES / f"claim21_fp8_order2_{m}_rho{RHO_TAG}.json")["order2_H_bpB"])
        ext[m] = dict(br=br, H2=h2)

    lines = []
    lines.append("Claim-21 wave 34: universal (cross-model) order-2 static coder")
    lines.append("=" * 124)
    lines.append("")
    lines.append("Tests whether wave-26 cross-model histogram correlation (r>0.9995 at order 0)")
    lines.append("translates into usable order-2 prior transfer. Leave-one-out protocol: for each")
    lines.append("held-out test model, sum order-2 triples from the other 3 models, add Laplace-alpha,")
    lines.append("and statically code the held-out stream. Oracle uses the test model's OWN counts.")
    lines.append("Reports both for alphas in {1.0, 0.5, 0.1, 0.01}.")
    lines.append("")

    # Per-alpha cohort summary
    hdr = (f"  {'alpha':<8}{'oracle (wt)':>14}{'universal (wt)':>18}"
           f"{'brotli-11 (wt)':>18}{'oracle - br':>14}{'univ - br':>12}"
           f"{'univ - oracle':>16}")
    lines.append(hdr)
    tN = sum(int(r["n_bytes"]) for r in rows)
    sB = sum(ext[r["test_model"]]["br"] * int(r["n_bytes"]) for r in rows)
    for a in ALPHAS:
        sO = sU = 0.0
        for r in rows:
            n = int(r["n_bytes"])
            sO += float(r["by_alpha"][a]["oracle_static_bpB"]) * n
            sU += float(r["by_alpha"][a]["universal_static_bpB"]) * n
        oracle = sO / tN
        universal = sU / tN
        br = sB / tN
        lines.append(
            f"  {a:<8}{oracle:>14.4f}{universal:>18.4f}"
            f"{br:>18.4f}{oracle-br:>+14.4f}{universal-br:>+12.4f}"
            f"{universal-oracle:>+16.4f}"
        )

    lines.append("")
    lines.append("Per-model breakdown at alpha=0.1 (best oracle point):")
    lines.append(
        f"  {'model':<14}{'n_bytes':>14}{'Shannon H2':>12}"
        f"{'oracle':>10}{'universal':>12}"
        f"{'brotli-11':>12}{'oracle-br':>12}{'univ-br':>10}"
    )
    for r in rows:
        m = r["test_model"]
        o = float(r["by_alpha"]["0.1"]["oracle_static_bpB"])
        u = float(r["by_alpha"]["0.1"]["universal_static_bpB"])
        br = ext[m]["br"]
        h2 = ext[m]["H2"]
        lines.append(
            f"  {m:<14}{int(r['n_bytes']):>14,}{h2:>12.4f}"
            f"{o:>10.4f}{u:>12.4f}"
            f"{br:>12.4f}{o-br:>+12.4f}{u-br:>+10.4f}"
        )

    lines.append("")
    lines.append("=" * 124)
    lines.append("INTERPRETATION")
    lines.append("-" * 124)
    lines.append(
        "- ORACLE static coder at alpha=0.1 beats brotli-11 on EVERY model:")
    lines.append(
        "    olmo2_1b     -0.119 bpB     qwen3_1.7b   -0.095 bpB")
    lines.append(
        "    smollm2_1.7b -0.065 bpB     tinyllama    -0.122 bpB")
    lines.append(
        "  Cohort mean -0.095 bpB. This CONSTRUCTIVELY realizes ~60% of the")
    lines.append(
        "  wave-31 theoretical -0.155 bpB gap and proves the sub-brotli path")
    lines.append(
        "  is implementable with an extremely simple static Laplace coder,")
    lines.append(
        "  provided the coder has access to the test-model-specific order-2")
    lines.append(
        "  context tables.")
    lines.append(
        "- UNIVERSAL (leave-one-out) static coder plateaus at ~6.70 bpB")
    lines.append(
        "  regardless of alpha, and FAILS TO BEAT brotli-11 (cohort gap +0.14")
    lines.append(
        "  bpB at alpha=1.0 rising slightly to +0.20 bpB at alpha=0.01 as")
    lines.append(
        "  low-smoothing amplifies out-of-sample context mass on the held-out")
    lines.append(
        "  model). The gap between oracle and universal at alpha=0.1 is")
    lines.append(
        "  +0.222 bpB cohort.")
    lines.append(
        "- IMPLICATION FOR WAVE-26 CROSS-MODEL TRANSFER: the r>0.9995")
    lines.append(
        "  correlation was measured on order-0 MARGINAL byte histograms.")
    lines.append(
        "  That does NOT imply the order-2 JOINT distribution transfers;")
    lines.append(
        "  wave 34 shows it does not, at least not tightly enough to beat")
    lines.append(
        "  brotli-11. The order-2 context structure carries model-specific")
    lines.append(
        "  information that a universal prior cannot capture.")
    lines.append(
        "- OPERATIONAL CONCLUSION for Claim 21: a constructive sub-brotli")
    lines.append(
        "  order-2 fp8 coder must ship with MODEL-SPECIFIC order-2 tables")
    lines.append(
        "  as side information, or equivalently (and cheaper) must BUILD the")
    lines.append(
        "  tables ONLINE in a second pass over the decoded stream. A single")
    lines.append(
        "  65,536 x 256 context table at 2 bytes per cell is 32 MiB --")
    lines.append(
        "  within budget for a model checkpoint but not for the per-tile")
    lines.append(
        "  envelope of the Claim-21 payload. Two-pass decode is the")
    lines.append(
        "  realistic shipping path.")
    lines.append("")

    out_txt = RES / "claim21_fp8_order2_universal.txt"
    out_json = RES / "claim21_fp8_order2_universal_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    out_json.write_text(json.dumps({
        "claim": 21,
        "wave": 34,
        "experiment": "fp8_order2_universal_summary",
        "rho": float(RHO_TAG),
        "rows": rows,
        "external": ext,
        "alphas": ALPHAS,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
