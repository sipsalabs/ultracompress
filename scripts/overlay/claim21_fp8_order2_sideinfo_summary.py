"""Aggregator for wave-36 side-info cost analysis."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
RHO_TAG = "0.01"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    d = load(RES / f"claim21_fp8_order2_sideinfo_rho{RHO_TAG}.json")
    rows = d["models"]

    lines = []
    lines.append("Claim-21 wave 36: one-shot net cost of shipping order-2 priors as side info")
    lines.append("=" * 124)
    lines.append("")
    lines.append("Wave 35 concluded that the sub-brotli-11 order-2 advantage requires shipping")
    lines.append("the context-count table as side information. Wave 36 quantifies that cost")
    lines.append("exactly by serializing the 65,536 x 256 count table with several general-")
    lines.append("purpose encoders and computing the NET rate (payload-coded + side-info-")
    lines.append("amortized) vs brotli-11.")
    lines.append("")
    lines.append("Table serialization (bytes; best encoder in bold per row):")
    hdr = (f"  {'model':<14}{'raw int32':>12}{'varint':>12}{'zlib9 int32':>14}"
           f"{'br11 int32':>14}{'br11 varint':>14}{'best':>18}")
    lines.append(hdr)
    for r in rows:
        e = r["encodings"]
        lines.append(
            f"  {r['model']:<14}{e['raw_int32']:>12,}{e['varint']:>12,}"
            f"{e['zlib9_int32']:>14,}{e['brotli11_int32']:>14,}"
            f"{e['brotli11_varint']:>14,}"
            f"{r['best_encoder']+' '+str(r['best_side_info_bytes']):>18}"
        )

    lines.append("")
    lines.append("Net one-shot rate = oracle alpha=0.01 + side_info * 8 / N_bytes:")
    hdr = (f"  {'model':<14}{'n_bytes':>14}{'oracle':>10}{'side (MB)':>12}"
           f"{'side bpB':>12}{'net':>10}{'brotli-11':>12}{'net - br':>12}")
    lines.append(hdr)
    tN = 0
    sOR = sSI = sNET = sBR = 0.0
    for r in rows:
        n = int(r["n_bytes"])
        tN += n
        sOR += r["oracle_alpha001_bpB"] * n
        sSI += r["side_info_bits_per_payload_byte"] * n
        sNET += r["net_rate_bpB"] * n
        sBR += r["brotli11_bpB"] * n
        sz_mb = r["best_side_info_bytes"] / (1024 * 1024)
        lines.append(
            f"  {r['model']:<14}{n:>14,}"
            f"{r['oracle_alpha001_bpB']:>10.4f}{sz_mb:>12.3f}"
            f"{r['side_info_bits_per_payload_byte']:>12.4f}"
            f"{r['net_rate_bpB']:>10.4f}{r['brotli11_bpB']:>12.4f}"
            f"{r['net_rate_bpB']-r['brotli11_bpB']:>+12.4f}"
        )
    if tN:
        lines.append("")
        lines.append(
            f"  {'COHORT':<14}{tN:>14,}"
            f"{sOR/tN:>10.4f}{'':>12}"
            f"{sSI/tN:>12.4f}"
            f"{sNET/tN:>10.4f}{sBR/tN:>12.4f}"
            f"{(sNET-sBR)/tN:>+12.4f}"
        )

    lines.append("")
    lines.append("=" * 124)
    lines.append("INTERPRETATION")
    lines.append("-" * 124)
    lines.append(
        "- Best serialization: brotli-11 applied to LEB128 varint count arrays,")
    lines.append(
        "  producing 1.09-1.38 MB per model (2-3% of the raw 67 MB int32 table).")
    lines.append(
        "- That cheapest side-info encoding still costs 0.69-0.97 bpB when amortized")
    lines.append(
        "  over the payload -- i.e., 4-6x the 0.155 bpB theoretical advantage from")
    lines.append(
        "  wave 31. The ONE-SHOT net rate (oracle payload + side info) is 7.15-7.35")
    lines.append(
        "  bpB, which is +0.58 to +0.78 bpB WORSE than brotli-11 at 6.53-6.57 bpB.")
    lines.append(
        "- Cohort net: 7.1829 bpB vs brotli-11 6.5583 bpB = +0.625 bpB WORSE.")
    lines.append(
        "- CONCLUSION: one-shot (single-payload) order-2 context coding on fp8 is")
    lines.append(
        "  NET WORSE than brotli-11 at this payload volume. Side-info overhead swamps")
    lines.append(
        "  the theoretical order-2 advantage by a factor of 4-6x.")
    lines.append(
        "- AMORTIZATION CAVEAT: in real deployment a model's priors are shipped ONCE")
    lines.append(
        "  and reused across every inference host / download / redeployment. If the")
    lines.append(
        "  same priors amortize over K >= 5 payload transfers, the effective side-info")
    lines.append(
        "  cost drops to 0.14-0.19 bpB, below the 0.155 bpB advantage. At K >= 10 the")
    lines.append(
        "  net is a clean win of roughly 0.07-0.10 bpB vs brotli-11. The practical")
    lines.append(
        "  deployment crossover is at K ~= 4-7 reuses per prior shipment.")
    lines.append(
        "- CLAIM-21 CONCLUDING STATEMENT ON ORDER-2 CONTEXT CODING:")
    lines.append(
        "    * The theoretical order-2 fp8 floor sits 0.155 bpB below brotli-11.")
    lines.append(
        "    * An oracle static Laplace-0.01 coder realizes 94% of that gap.")
    lines.append(
        "    * No payload-only coder (adaptive, self-bootstrap, universal cross-")
    lines.append(
        "      model) can beat brotli-11.")
    lines.append(
        "    * Shipping priors as side info costs 0.7-1.0 bpB one-shot -- a 4-6x")
    lines.append(
        "      overhead -- making order-2 context coding NET WORSE than brotli-11")
    lines.append(
        "      for single-deployment use.")
    lines.append(
        "    * Order-2 context coding is NET POSITIVE only under multi-deployment")
    lines.append(
        "      amortization (K >= 5 reuses), a legitimate but limited shipping")
    lines.append(
        "      regime.")
    lines.append("")

    out_txt = RES / "claim21_fp8_order2_sideinfo.txt"
    out_json = RES / "claim21_fp8_order2_sideinfo_summary.json"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    out_json.write_text(json.dumps({
        "claim": 21,
        "wave": 36,
        "experiment": "fp8_order2_sideinfo_summary",
        "rho": float(RHO_TAG),
        "cohort": {
            "n_bytes": tN,
            "oracle_alpha001_bpB": sOR / tN,
            "side_info_bpB": sSI / tN,
            "net_rate_bpB": sNET / tN,
            "brotli11_bpB": sBR / tN,
            "net_minus_brotli_bpB": (sNET - sBR) / tN,
        },
        "per_model": rows,
    }, indent=2), encoding="utf-8")
    print(f"[wrote] {out_txt}")
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
