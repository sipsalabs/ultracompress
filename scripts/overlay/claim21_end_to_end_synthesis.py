"""Claim-21 wave 27: end-to-end bit-level payload synthesis.

Waves 24 and 26 each computed a tighter-than-brotli per-stream bound:
  - wave 24: Rice-best idx_delta emitter reaches 2.022 bpB (cohort)
  - wave 26: joint fp16 scale H reaches 4.377 bpB (cohort)
  - wave 19: fp8 order-0 Shannon H is 6.691 bpB (cohort)

Wave 27 combines these into a single end-to-end payload lower bound
per model and compares against:
  (a) raw payload = 8 bpB everywhere (the literal byte count)
  (b) shipping v17 payload under brotli-11 (wave 23 codec_sweep)
  (c) the per-stream order-0 Shannon floor (wave 19)

For each model at rho=0.010 we assemble the total payload bits under
four assumptions:
  - RAW:        sum(n_bytes) * 8
  - BROTLI-11:  sum(bytes_reported_by_wave23_codec_sweep) * 8
  - ORDER-0:    sum(n_bytes * H_order0_per_byte)
  - BITLEVEL:   fp8_order0 + rice_idx_delta (wave24) + joint_scale (wave26)

Output: results/claim21_end_to_end_synthesis.{txt,json}
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
MODELS = ["olmo2_1b", "qwen3_1.7b", "smollm2_1.7b", "tinyllama"]
RHO = 0.010


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    rows = []
    for m in MODELS:
        w19 = load(RES / f"claim21_fp8_histogram_{m}_rho0.01.json")
        w15 = load(RES / f"claim21_codec_sweep_{m}_rho0.01.json")
        w24 = load(RES / f"claim21_varint_emitter_{m}_rho0.01.json")
        w26 = load(RES / f"claim21_scale_pair_{m}_rho0.01.json")

        # --- per-stream n_bytes, H from wave 19
        fp8_n = int(w19["streams"]["fp8"]["n_bytes"])
        fp8_H = float(w19["streams"]["fp8"]["shannon_bits_per_byte"])
        idx_n = int(w19["streams"]["idx_delta"]["n_bytes"])
        idx_H = float(w19["streams"]["idx_delta"]["shannon_bits_per_byte"])
        sc_n  = int(w19["streams"]["scale"]["n_bytes"])
        sc_H  = float(w19["streams"]["scale"]["shannon_bits_per_byte"])

        # --- brotli-11 bytes from wave 15 codec_sweep
        cs = w15["codec_sweep"]
        fp8_br = int(cs["fp8"]["codecs"]["brotli-11"]["bytes"])
        idx_br = int(cs["idx_delta"]["codecs"]["brotli-11"]["bytes"])
        sc_br  = int(cs["scale"]["codecs"]["brotli-11"]["bytes"])

        # --- wave 24 rice-best idx_delta bits
        idx_rice = int(w24["rice_best_bits"])

        # --- wave 26 joint scale bits
        sc_joint_bpB = float(w26["joint_H_bpB"])  # bpB per byte
        sc_joint_bits = sc_joint_bpB * sc_n  # bits

        # --- fp8 order-0 bits = fp8_H * fp8_n
        fp8_ord_bits = fp8_H * fp8_n

        # --- totals
        raw_bytes   = fp8_n + idx_n + sc_n
        raw_bits    = raw_bytes * 8
        brotli_bits = (fp8_br + idx_br + sc_br) * 8
        order0_bits = fp8_H*fp8_n + idx_H*idx_n + sc_H*sc_n
        bitlevel_bits = fp8_ord_bits + idx_rice + sc_joint_bits

        rows.append(dict(
            model=m,
            n_bytes_total=raw_bytes,
            fp8_bytes=fp8_n, idx_bytes=idx_n, scale_bytes=sc_n,
            raw_bits=raw_bits,
            brotli_bits=brotli_bits,
            order0_bits=order0_bits,
            bitlevel_bits=bitlevel_bits,
            brotli_bpB=brotli_bits/raw_bytes,
            order0_bpB=order0_bits/raw_bytes,
            bitlevel_bpB=bitlevel_bits/raw_bytes,
            reduction_vs_brotli_pct=(brotli_bits-bitlevel_bits)/brotli_bits*100,
            reduction_vs_raw_pct=(raw_bits-bitlevel_bits)/raw_bits*100,
            approach_order0_pct=(
                (brotli_bits-bitlevel_bits) /
                max(1.0, (brotli_bits-order0_bits)) * 100.0
            ),
        ))

    # cohort totals
    def S(key): return sum(r[key] for r in rows)
    raw_T = S("raw_bits"); br_T = S("brotli_bits")
    o0_T  = S("order0_bits"); bl_T = S("bitlevel_bits")
    nB_T  = S("n_bytes_total")

    lines = []
    lines.append("Claim-21 wave 27: end-to-end bit-level payload synthesis")
    lines.append("=" * 98)
    lines.append("")
    lines.append("Combines wave-24 (Rice idx_delta 2.022 bpB cohort) +")
    lines.append("wave-26 (joint fp16 scale 4.377 bpB cohort) +")
    lines.append("wave-19 (fp8 order-0 H 6.691 bpB cohort) into a single")
    lines.append("end-to-end payload lower bound per model, compared against")
    lines.append("raw (8 bpB), shipping brotli-11, and per-stream order-0.")
    lines.append("")
    hdr = (f"  {'model':<14}{'n_bytes':>12}"
           f"{'raw bpB':>10}{'brotli bpB':>12}{'order0 bpB':>12}"
           f"{'bitlvl bpB':>12}{'vs brotli %':>13}{'vs raw %':>11}")
    lines.append(hdr)
    for r in rows:
        lines.append(
            f"  {r['model']:<14}{r['n_bytes_total']:>12,}"
            f"{8.0:>10.4f}"
            f"{r['brotli_bpB']:>12.4f}"
            f"{r['order0_bpB']:>12.4f}"
            f"{r['bitlevel_bpB']:>12.4f}"
            f"{r['reduction_vs_brotli_pct']:>12.3f}%"
            f"{r['reduction_vs_raw_pct']:>10.3f}%"
        )
    # cohort row
    lines.append("")
    lines.append(f"  {'COHORT':<14}{nB_T:>12,}"
                 f"{8.0:>10.4f}"
                 f"{br_T/nB_T:>12.4f}"
                 f"{o0_T/nB_T:>12.4f}"
                 f"{bl_T/nB_T:>12.4f}"
                 f"{(br_T-bl_T)/br_T*100:>12.3f}%"
                 f"{(raw_T-bl_T)/raw_T*100:>10.3f}%")
    lines.append("")
    lines.append("Additional cohort figures:")
    lines.append(
        f"  absolute cohort payload: raw = {raw_T/8:,.0f} B  "
        f"brotli-11 = {br_T/8:,.0f} B  bitlevel = {bl_T/8:,.0f} B")
    lines.append(
        f"  bitlevel saves {(br_T-bl_T)/8:,.0f} B over brotli-11 shipping "
        f"payload (cohort)")
    lines.append(
        f"  bitlevel closes "
        f"{(br_T-bl_T)/max(1.0,(br_T-o0_T))*100:.1f}% of the "
        f"brotli-to-order0 gap")
    lines.append("")
    lines.append(
        "Bit-level construction used:  fp8 = order-0 Shannon H (wave 19),")
    lines.append(
        "idx_delta = Rice-best (wave 24, k=6 on all 4 models),  scale =")
    lines.append(
        "joint 16-bit pair H (wave 26). These are per-stream information-")
    lines.append(
        "theoretic bounds reachable by an arithmetic coder on each stream's")
    lines.append(
        "own distribution (and in wave 24's case, a concrete bit-packed")
    lines.append(
        "Rice emitter also achieves the bound within 0.06 bpB).")
    lines.append("")
    lines.append("=" * 98)
    txt = "\n".join(lines) + "\n"
    (RES/"claim21_end_to_end_synthesis.txt").write_text(txt, encoding="utf-8")
    out = dict(
        claim=21, experiment="end_to_end_synthesis", rho=RHO,
        models=MODELS, per_model=rows,
        cohort=dict(
            n_bytes_total=nB_T,
            raw_bits=raw_T, brotli_bits=br_T,
            order0_bits=o0_T, bitlevel_bits=bl_T,
            raw_bpB=8.0,
            brotli_bpB=br_T/nB_T,
            order0_bpB=o0_T/nB_T,
            bitlevel_bpB=bl_T/nB_T,
            reduction_vs_brotli_pct=(br_T-bl_T)/br_T*100,
            reduction_vs_raw_pct=(raw_T-bl_T)/raw_T*100,
            close_gap_to_order0_pct=(br_T-bl_T)/max(1.0,(br_T-o0_T))*100,
        ),
    )
    (RES/"claim21_end_to_end_synthesis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(txt)
    print("[wrote] results/claim21_end_to_end_synthesis.txt")
    print("[wrote] results/claim21_end_to_end_synthesis.json")


if __name__ == "__main__":
    main()
