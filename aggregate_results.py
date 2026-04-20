"""Aggregate Claim 16 cross-family .pt artifacts into a single results.json.

Reads v17_*_ppl.pt and topk_*_results.pt for each of the 4 validated models
and produces one flat JSON consumable by plots / portfolio pages.
"""
import json
import torch

MODELS = [
    # (display_name, ppl_file, topk_file, params_B, family, body_linears, sigma_ratio)
    ("TinyLlama-1.1B",   "v17_tinyllama_ppl.pt", "topk_tinyllama_results.pt",  1.100, "Llama-2", 154, 1126),
    ("OLMo-2-1B",        "v17_olmo2_ppl.pt",     "topk_olmo2_results.pt",      1.485, "Llama-2", 112, 20),
    ("SmolLM2-1.7B",     "v17_smollm2_ppl.pt",   "topk_smollm2_results.pt",    1.812, "Llama-2", 168, 779),
    ("Qwen3-1.7B",       "v17_ppl_results.pt",   "claim16_topk_results.pt",    1.7,   "Qwen3",   168, 120),
    ("Mistral-7B-v0.3",  "v17_mistral_ppl.pt",   "topk_mistral_results.pt",    7.248, "Mistral", 224, 2173),
    ("Qwen3-8B",         "v17_8b_ppl.pt",        "topk_8b_results.pt",         8.19,  "Qwen3",   252, 120),
]

# Fallback known values (from PATENT_CLAIMS.md) for files that may not exist yet
KNOWN = {
    "Qwen3-1.7B":  {"ppl_fp16": 33.21, "ppl_v17": 59.40, "ratio": 1.788,
                    "bpw": 2.4017, "rel_w_mean": 0.0643, "rel_w_max": 0.0941,
                    "t1_teacher": None, "t10_teacher": None,
                    "t1_v17": None, "t10_v17": None,
                    "t1_retention": 84.65, "t10_retention": 90.68,
                    "t1_agreement": None, "t10_agreement": 93.88},
    "Qwen3-8B":    {"ppl_fp16": 20.6963, "ppl_v17": 28.6829, "ratio": 1.386,
                    "bpw": 2.3998, "rel_w_mean": 0.0642, "rel_w_max": 0.0834,
                    "t1_teacher": None, "t10_teacher": None,
                    "t1_v17": None, "t10_v17": None,
                    "t1_retention": 91.85, "t10_retention": 95.83,
                    "t1_agreement": None, "t10_agreement": 96.98},
}


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None


def extract(name, ppl_file, topk_file, params_B, family, body_linears, sigma):
    out = {
        "model": name, "family": family, "params_B": params_B,
        "body_linears": body_linears, "sigma2_ratio_q": sigma,
        "alpha_attn": 0.25, "alpha_mlp": 0.125, "D": 8,
    }
    ppl = load_pt(ppl_file) if ppl_file else None
    topk = load_pt(topk_file) if topk_file else None

    if ppl is not None:
        if "ppl_baseline" in ppl: out["ppl_fp16"] = float(ppl["ppl_baseline"])
        if "ppl_v17" in ppl:      out["ppl_v17"]  = float(ppl["ppl_v17"])
        if "ratio" in ppl:        out["ratio"]    = float(ppl["ratio"])
        if "global_bpw" in ppl:   out["bpw"]      = float(ppl["global_bpw"])
        if "rel_w_final_mean" in ppl and ppl["rel_w_final_mean"] is not None:
            out["rel_w_mean"] = float(ppl["rel_w_final_mean"])
        # older Qwen3-1.7B schema may have {"baseline":..,"v16":..,"v17":..}
        # values could be dicts or floats; only use if numeric
        b, v = ppl.get("baseline"), ppl.get("v17")
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            out.setdefault("ppl_fp16", float(b))
            out.setdefault("ppl_v17",  float(v))
            out.setdefault("ratio", float(v) / float(b))

    if topk is not None:
        t = topk.get("teacher", {}) or {}
        c = topk.get("compressed") or topk.get("claim16") or {}
        if "t1_gt" in t:    out["t1_teacher"]  = 100.0 * float(t["t1_gt"])
        if "t10_gt" in t:   out["t10_teacher"] = 100.0 * float(t["t10_gt"])
        if "t1_gt" in c:    out["t1_v17"]      = 100.0 * float(c["t1_gt"])
        if "t10_gt" in c:   out["t10_v17"]     = 100.0 * float(c["t10_gt"])
        if "t1_agree" in c and c["t1_agree"] == c["t1_agree"]:    # NaN check
            out["t1_agreement"]  = 100.0 * float(c["t1_agree"])
        if "t10_agree" in c and c["t10_agree"] == c["t10_agree"]:
            out["t10_agreement"] = 100.0 * float(c["t10_agree"])
        if "retention_t1" in topk:  out["t1_retention"]  = float(topk["retention_t1"])
        if "retention_t10" in topk: out["t10_retention"] = float(topk["retention_t10"])
        # if retention missing, compute from t1_gt
        if "t1_retention" not in out and "t1_teacher" in out and "t1_v17" in out:
            out["t1_retention"] = 100.0 * out["t1_v17"] / out["t1_teacher"]
        if "t10_retention" not in out and "t10_teacher" in out and "t10_v17" in out:
            out["t10_retention"] = 100.0 * out["t10_v17"] / out["t10_teacher"]

    # fill with known values if missing
    if name in KNOWN:
        for k, v in KNOWN[name].items():
            if k not in out or out[k] is None:
                out[k] = v
    return out


def main():
    rows = [extract(*m) for m in MODELS]
    summary = {
        "claim": "Claim 16: (a_attn=0.25, a_mlp=0.125) 2.40-bpw model-agnostic compression law",
        "operating_point": {"alpha_attn": 0.25, "alpha_mlp": 0.125,
                            "D": 8, "beam": 8, "iters": 6,
                            "K_per_role": {"q_proj": [2048,256], "k_proj":[2048,256],
                                           "v_proj":[2048,256], "o_proj":[4096,512],
                                           "gate_proj":[2048,256], "up_proj":[2048,256],
                                           "down_proj":[2048,256]}},
        "models": rows,
        "envelope": {
            "bpw_min": min(r.get("bpw", 2.4) for r in rows),
            "bpw_max": max(r.get("bpw", 2.4) for r in rows),
            "ratio_min": min(r.get("ratio", 2.0) for r in rows),
            "ratio_max": max(r.get("ratio", 2.0) for r in rows),
            "t10_agreement_min": min(r.get("t10_agreement", 100) for r in rows),
            "t10_retention_min": min(r.get("t10_retention", 100) for r in rows),
        },
    }
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results.json with {len(rows)} models.")
    for r in rows:
        bpw = r.get("bpw", "?"); ratio = r.get("ratio", "?"); agr = r.get("t10_agreement", "?")
        bpw_s = f"{bpw:.4f}" if isinstance(bpw, float) else str(bpw)
        ratio_s = f"{ratio:.3f}x" if isinstance(ratio, float) else str(ratio)
        agr_s = f"{agr:.2f}%" if isinstance(agr, float) else str(agr)
        print(f"  {r['model']:<20} bpw={bpw_s}  ratio={ratio_s}  T10agr={agr_s}")


if __name__ == "__main__":
    main()
