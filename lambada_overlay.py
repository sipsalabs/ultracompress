"""lambada_overlay.py -- Claim 17: sparse fp16 outlier-row overlay.

For every body Linear of every model, after the Claim-16 codebook decode
produces Wq, we compute a per-row activation-weighted residual score
    score[o] = sum_i ( s_col[i] * (W_fp16[o,i] - Wq[o,i]) )^2
and restore the top rho*O rows to their fp16 ground-truth values. The
restored rows carry a uint32 row index + a row of fp16 values, for a
bit-cost per restored row of (16*I + 32) bits vs the Claim-16 cost
of (global_bpw * I). Net overhead is ~= rho * (16 - global_bpw) bpw
(plus 32*rho/I for indices, which is negligible for I>=1024).

The overlay composes on top of *any* Claim-16 fit (base 2.40, hifi 2.78,
or ultra tier) without refitting. We evaluate on the six hifi fits at
rho in {0.002, 0.005} and write all rows to lambada_overlay_results.json.
"""
from __future__ import annotations
import argparse, gc, json, os, time, traceback
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign


def _reconstruct_v17_with_overlay(W_fp16: torch.Tensor, role: str, bank: dict,
                                  s_col: torch.Tensor, D: int,
                                  rot: torch.Tensor, device: str,
                                  rho: float) -> tuple[torch.Tensor, int]:
    """v17 decode + outlier-row overlay. Returns (Wq_with_overlay_fp16_cpu, n_restored)."""
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    W_scaled = W * s.unsqueeze(0)
    Wrot = W_scaled @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    chunk = max(25_000, (200_000 * 2048) // max(cb1.shape[0], 1))
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=chunk)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)                        # fp32 [O, I], model space

    # per-row activation-weighted residual
    diff = (W - Wq) * s.unsqueeze(0)                       # [O, I]
    score = (diff * diff).sum(1)                           # [O]
    K = max(1, int(round(rho * O))) if rho > 0 else 0
    n_restored = 0
    if K > 0:
        idx = score.topk(K).indices
        # restore top-K rows to fp16 ground truth
        Wq[idx] = W[idx]
        n_restored = K

    del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, s, diff, score
    out = Wq.to(dtype=torch.float16, device="cpu")
    del Wq
    return out, n_restored


def substitute_v17_overlay(model, state_dict: dict, v17: dict, device: str,
                           D: int, rho: float):
    rots = {}
    dims = sorted({v.shape[1] for k, v in state_dict.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims:
        rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v17["banks"]
    s_col = v17["s_col"]
    hf_keys = [k for k in state_dict.keys()
               if "layers." in k and any(p in k for p in ROLE_PATTERNS)
               and k.endswith(".weight") and state_dict[k].ndim == 2
               and state_dict[k].shape[1] % D == 0]
    print(f"  substituting {len(hf_keys)} body Linears (v17+overlay, "
          f"rho={rho}, alpha={v17.get('alpha','?')})")
    total_restored = 0
    total_rows = 0
    total_params_restored = 0
    total_params = 0
    missing = 0
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        W = state_dict[k]
        if k not in s_col:
            missing += 1
            s = torch.ones(W.shape[1])
        else:
            s = s_col[k]
        W_new, nr = _reconstruct_v17_with_overlay(
            W, role, bank, s, D, rots[W.shape[1]], device, rho)
        total_restored += nr
        total_rows += W.shape[0]
        total_params_restored += nr * W.shape[1]
        total_params += W.numel()
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_new.to(mod.weight.device, dtype=mod.weight.dtype))
        del W_new
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    if missing:
        print(f"  WARNING: {missing} tensors missing s_col; used identity")
    torch.cuda.empty_cache(); gc.collect()
    print(f"  overlay: restored {total_restored}/{total_rows} rows "
          f"({100*total_restored/max(total_rows,1):.3f}%), "
          f"{total_params_restored}/{total_params} params "
          f"({100*total_params_restored/max(total_params,1):.3f}%)")
    # bpw overhead = extra bits per param vs baseline fit
    #   per restored row: 16*I + 32 bits, baseline was global_bpw*I bits
    #   excess = (16 - global_bpw)*I + 32  (per row)
    gbpw = float(v17.get("global_bpw", 2.78))
    excess_bits = 0.0
    if total_restored > 0:
        # approximate I as params_restored / rows_restored
        avg_I = total_params_restored / total_restored
        excess_bits = total_restored * ((16 - gbpw) * avg_I + 32)
    overlay_bpw = excess_bits / max(total_params, 1)
    print(f"  overlay bpw overhead: +{overlay_bpw:.4f} "
          f"(base {gbpw:.4f} -> effective {gbpw+overlay_bpw:.4f})")
    return {
        "n_restored_rows": total_restored,
        "n_total_rows": total_rows,
        "rows_pct": 100*total_restored/max(total_rows,1),
        "params_pct": 100*total_params_restored/max(total_params,1),
        "overlay_bpw": float(overlay_bpw),
        "base_bpw": gbpw,
        "effective_bpw": float(gbpw + overlay_bpw),
    }


MODELS = [
    # (display, hf_id, teacher, v17hi_fit, lambada_tokens)
    ("OLMo-2-1B",      "allenai/OLMo-2-0425-1B",             "olmo2_1b_cache.pt",        "v17hi_fit_olmo2.pt",      "lambada_test_olmo2.pt"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_1.1b_cache.pt",  "v17hi_fit_tinyllama.pt",  "lambada_test_tinyllama.pt"),
    ("Qwen3-1.7B",     "Qwen/Qwen3-1.7B",                    "qwen3_1.7b_cache.pt",      "v17hi_fit_qwen3_1.7b.pt", "lambada_test_qwen3.pt"),
    ("SmolLM2-1.7B",   "HuggingFaceTB/SmolLM2-1.7B",         "smollm2_1.7b_cache.pt",    "v17hi_fit_smollm2.pt",    "lambada_test_smollm2.pt"),
    ("Mistral-7B",     "mistralai/Mistral-7B-v0.3",          "mistral_7b_v0.3_cache.pt", "v17hi_fit_mistral.pt",    "lambada_test_mistral.pt"),
    ("Qwen3-8B",       "Qwen/Qwen3-8B",                      "qwen3_8b_cache.pt",        "v17hi_fit_8b.pt",         "lambada_test_qwen3.pt"),
]


def run_one(name: str, model_id: str, teacher_path: str, v17_path: str,
            tokens_path: str, n: int, seq_len: int, device: str,
            rho: float) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    from eval_v16_ppl import measure_ppl
    from eval_claim16_topk import measure_topk

    print(f"\n[{name}] teacher -> {teacher_path}", flush=True)
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    toks = torch.load(tokens_path, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, toks.numel() - seq_len - 1, (n,), generator=g)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    t0 = time.time()
    tch_topk, tch_cache = measure_topk(model, toks, starts, seq_len, device, teacher_topk=None)
    tch_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] teacher  PPL={tch_ppl:.4f}  T1={tch_topk['t1_gt']*100:.2f}%  "
          f"T10={tch_topk['t10_gt']*100:.2f}%   ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    D = v17.get("D", 8)
    ov_stats = substitute_v17_overlay(model, sd, v17, device, D, rho)
    v17_topk, _ = measure_topk(model, toks, starts, seq_len, device, teacher_topk=tch_cache)
    v17_ppl, _ = measure_ppl(model, toks, starts, seq_len, device)
    print(f"[{name}] v17+ov rho={rho}  PPL={v17_ppl:.4f}  T1={v17_topk['t1_gt']*100:.2f}%  "
          f"T10={v17_topk['t10_gt']*100:.2f}%  T1_vs_teacher={v17_topk['t1_agree']*100:.2f}%  "
          f"({time.time()-t1:.0f}s)", flush=True)

    ppl_ratio = v17_ppl / tch_ppl
    t1_ret = v17_topk['t1_gt'] / tch_topk['t1_gt'] if tch_topk['t1_gt'] > 0 else 0.0
    t10_ret = v17_topk['t10_gt'] / tch_topk['t10_gt'] if tch_topk['t10_gt'] > 0 else 0.0

    rec = {
        "name": name, "model_id": model_id,
        "fit": v17_path, "rho": rho,
        "n": n, "seq_len": seq_len,
        "teacher_ppl": float(tch_ppl),
        "teacher_t1": float(tch_topk['t1_gt']),
        "teacher_t10": float(tch_topk['t10_gt']),
        "v17_ppl": float(v17_ppl),
        "v17_t1": float(v17_topk['t1_gt']),
        "v17_t10": float(v17_topk['t10_gt']),
        "v17_t1_vs_teacher": float(v17_topk['t1_agree']),
        "ppl_ratio": float(ppl_ratio),
        "t1_ret": float(t1_ret),
        "t10_ret": float(t10_ret),
        "tier": "hifi+overlay",
    }
    rec.update(ov_stats)
    del model, sd, toks
    torch.cuda.empty_cache(); gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rhos", default="0.002,0.005",
                    help="comma-separated overlay row fractions")
    ap.add_argument("--only", default="", help="comma-separated model short tokens")
    ap.add_argument("--out", default="lambada_overlay_results.json")
    args = ap.parse_args()

    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]
    wanted = set(s.strip().lower() for s in args.only.split(",") if s.strip()) or None

    results = []
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                results = json.load(f)
        except Exception:
            results = []

    for rho in rhos:
        for name, mid, teacher, fit, tokens in MODELS:
            if wanted and name.lower() not in wanted and name.lower().split("-")[0] not in wanted:
                continue
            if not os.path.exists(fit):
                print(f"[skip] {name}: {fit} not present")
                continue
            key = (name, rho)
            if any((r.get("name"), r.get("rho")) == key and "error" not in r for r in results):
                print(f"[skip] {name} rho={rho}: already done")
                continue
            try:
                r = run_one(name, mid, teacher, fit, tokens,
                            args.n, args.seq_len, args.device, rho)
            except Exception as e:
                traceback.print_exc()
                r = {"name": name, "rho": rho, "fit": fit, "error": repr(e)}
            results = [x for x in results if (x.get("name"), x.get("rho")) != key] + [r]
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[lambada_overlay] wrote {args.out} ({len(results)})", flush=True)

    # summary
    print("\n================ overlay summary ================")
    print(f"{'model':<18} {'rho':>7} {'t1_ret':>8} {'ppl_r':>7} {'eff_bpw':>8}")
    for r in sorted(results, key=lambda x: (str(x.get('rho')), x.get('name',''))):
        if "error" in r:
            print(f"  {r.get('name','?'):<16}  rho={r.get('rho','?')}  ERROR")
            continue
        print(f"{r['name']:<18} {r['rho']:>7.4f} {r['t1_ret']*100:>7.2f}% "
              f"{r['ppl_ratio']:>7.3f} {r['effective_bpw']:>7.4f}")


if __name__ == "__main__":
    main()
