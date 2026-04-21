"""
eval_v17_ppl.py — PPL validation of Claim 14 (activation-aware input-column
rescaling). Same protocol as eval_v16_ppl.py; reports v17 vs v16 vs fp16
baseline on identical WikiText-103 windows so the activation-awareness
benefit is isolated.
"""
from __future__ import annotations
import argparse, gc, math, time
import torch
import torch.nn.functional as F

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign
from eval_v16_ppl import (
    measure_ppl, substitute_v16, reset_teacher, _reconstruct_v16,
)


def _reconstruct_v17(W_fp16: torch.Tensor, role: str, bank: dict,
                     s_col: torch.Tensor, D: int, rot: torch.Tensor,
                     device: str) -> torch.Tensor:
    """v17 decode: W' = W*s, rotate, assign, reconstruct, unrotate, unscale."""
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    W_scaled = W * s.unsqueeze(0)
    Wrot = W_scaled @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    # Scale beam_assign chunk inversely with K1 so hifi codebooks don't OOM on 7B/8B.
    chunk = max(25_000, (200_000 * 2048) // max(cb1.shape[0], 1))
    idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8, chunk=chunk)
    gh = cb1[idx1] + cb2[idx2]
    Wq_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.expand(O, I)
    Wq_scaled = Wq_rot_scaled @ rot.T
    Wq = Wq_scaled / s.unsqueeze(0)
    del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, gh, Wq_rot_scaled, Wq_scaled, s
    return Wq.to(dtype=torch.float16, device="cpu")


def substitute_v17(model, state_dict: dict, v17: dict, device: str, D: int):
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
    print(f"  substituting {len(hf_keys)} body Linears (v17 stack, "
          f"alpha={v17.get('alpha','?')})")
    missing = 0
    for n, k in enumerate(hf_keys):
        role = _role_of(k)
        bank = banks[role]
        if k not in s_col:
            missing += 1
            s = torch.ones(state_dict[k].shape[1])
        else:
            s = s_col[k]
        W_new = _reconstruct_v17(state_dict[k], role, bank, s, D,
                                 rots[state_dict[k].shape[1]], device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16", default="v16_result.pt")
    ap.add_argument("--v17", default="v17_result.pt")
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_v16", action="store_true")
    ap.add_argument("--out", default="v17_ppl_results.pt")
    args = ap.parse_args()
    device = args.device

    print(f"[eval] loading teacher from {args.teacher}")
    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd: teacher_sd = teacher_sd["state_dict"]

    print(f"[eval] loading tokens from {args.tokens}")
    all_tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, all_tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    print(f"[eval] building Qwen3-1.7B on {device}")
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    results = {}

    print("\n[eval] === BASELINE (fp16) ===")
    t = time.time()
    ppl_b, nll_b = measure_ppl(model, all_tokens, starts, args.seq_len, device)
    results["baseline"] = {"ppl": ppl_b, "nll": nll_b}
    print(f"[eval] baseline PPL = {ppl_b:.4f}  ({time.time()-t:.0f}s)")

    if not args.skip_v16:
        print(f"\n[eval] === v16 (loading {args.v16}) ===")
        v16 = torch.load(args.v16, map_location="cpu", weights_only=False)
        substitute_v16(model, teacher_sd, v16, device, args.D)
        t = time.time()
        ppl_16, nll_16 = measure_ppl(model, all_tokens, starts, args.seq_len, device)
        results["v16"] = {"ppl": ppl_16, "nll": nll_16,
                          "global_bpw": v16.get("global_bpw")}
        print(f"[eval] v16 PPL = {ppl_16:.4f}  (ratio {ppl_16/ppl_b:.3f}×)  "
              f"({time.time()-t:.0f}s)")
        reset_teacher(model, teacher_sd)
        del v16; torch.cuda.empty_cache(); gc.collect()

    print(f"\n[eval] === v17 (loading {args.v17}) ===")
    v17 = torch.load(args.v17, map_location="cpu", weights_only=False)
    print(f"[eval]   alpha={v17.get('alpha')}  "
          f"rel-W mean={v17.get('rel_w_final_mean'):.4f}")
    substitute_v17(model, teacher_sd, v17, device, args.D)
    t = time.time()
    ppl_17, nll_17 = measure_ppl(model, all_tokens, starts, args.seq_len, device)
    results["v17"] = {"ppl": ppl_17, "nll": nll_17,
                      "alpha": v17.get("alpha"),
                      "global_bpw": v17.get("global_bpw"),
                      "overhead_bpw": v17.get("overhead_bpw")}
    print(f"[eval] v17 PPL = {ppl_17:.4f}  (ratio {ppl_17/ppl_b:.3f}×)  "
          f"({time.time()-t:.0f}s)")

    print("\n" + "=" * 60)
    print(f"SUMMARY  (WikiText-103 test, n={args.n}, seq_len={args.seq_len})")
    print("=" * 60)
    print(f"  baseline fp16 : PPL {ppl_b:.4f}  (ratio 1.000×)")
    if "v16" in results:
        r = results["v16"]["ppl"] / ppl_b
        print(f"  v16  2.396bpw : PPL {results['v16']['ppl']:.4f}  "
              f"(ratio {r:.2f}×)")
    r17 = ppl_17 / ppl_b
    bpw17 = (v17.get("global_bpw") or 0) + (v17.get("overhead_bpw") or 0)
    print(f"  v17  {bpw17:.3f}bpw : PPL {ppl_17:.4f}  (ratio {r17:.2f}×, "
          f"alpha={v17.get('alpha')})")
    if "v16" in results:
        print(f"  v17/v16 PPL ratio: {ppl_17/results['v16']['ppl']:.3f}× "
              f"({(1-ppl_17/results['v16']['ppl'])*100:+.1f}% improvement)")

    torch.save(results, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
