"""eval_v18_ppl.py — PPL for v18 (two-sided scaling, Claim 15)."""
from __future__ import annotations
import argparse, gc, time
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign
from eval_v16_ppl import measure_ppl, reset_teacher, substitute_v16
from eval_v17_ppl import substitute_v17


def _reconstruct_v18(W_fp16, role, bank, s_col, u_row, D, rot, device):
    W = W_fp16.to(device=device, dtype=torch.float32)
    s = s_col.to(device=device, dtype=torch.float32)
    u = u_row.to(device=device, dtype=torch.float32)
    W_tilde = u.unsqueeze(1) * W * s.unsqueeze(0)
    Wrot = W_tilde @ rot
    O, I = Wrot.shape
    rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)
    g = (Wrot / rs).view(O, I//D, D).reshape(-1, D)
    cb1 = bank["cb1"].to(device); cb2 = bank["cb2"].to(device)
    i1, i2, _ = beam_assign(g, cb1, cb2, beam=8)
    gh = cb1[i1] + cb2[i2]
    Wq_rot_t = gh.view(O, I//D, D).reshape(O, I) * rs.expand(O, I)
    Wq_t = Wq_rot_t @ rot.T
    Wq = Wq_t / u.unsqueeze(1) / s.unsqueeze(0)
    del W, s, u, W_tilde, Wrot, g, cb1, cb2, i1, i2, gh, Wq_rot_t, Wq_t
    return Wq.to(dtype=torch.float16, device="cpu")


def substitute_v18(model, state_dict, v18, device, D):
    rots = {}
    dims = sorted({v.shape[1] for k, v in state_dict.items()
                   if "layers." in k and any(p in k for p in ROLE_PATTERNS)
                   and v.ndim == 2 and v.shape[1] % D == 0})
    for I in dims: rots[I] = build_rotation(I, device, seed=42 + I)
    banks = v18["banks"]; s_all = v18["s_col"]; u_all = v18["u_row"]
    hf = [k for k in state_dict if "layers." in k
          and any(p in k for p in ROLE_PATTERNS)
          and k.endswith(".weight") and state_dict[k].ndim == 2
          and state_dict[k].shape[1] % D == 0]
    print(f"  substituting {len(hf)} Linears (v18 two-sided, "
          f"alpha={v18.get('alpha')} beta={v18.get('beta')})")
    miss = 0
    for n, k in enumerate(hf):
        role = _role_of(k); W = state_dict[k]
        s = s_all.get(k, torch.ones(W.shape[1]))
        u = u_all.get(k, torch.ones(W.shape[0]))
        if k not in s_all or k not in u_all: miss += 1
        Wn = _reconstruct_v18(W, role, banks[role], s, u, D,
                              rots[W.shape[1]], device)
        mod = model
        for p in k.replace(".weight", "").split("."): mod = getattr(mod, p)
        mod.weight.data.copy_(Wn.to(mod.weight.device, dtype=mod.weight.dtype))
        del Wn
        if (n+1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    if miss: print(f"  WARNING: {miss} missing scale entries")
    torch.cuda.empty_cache(); gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16", default="v16_result.pt")
    ap.add_argument("--v17", default="v17_result.pt")
    ap.add_argument("--v18", default="v18_result.pt")
    ap.add_argument("--teacher", default="qwen3_1.7b_cache.pt")
    ap.add_argument("--tokens", default="wikitext103_test_qwen3.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_v16", action="store_true")
    ap.add_argument("--skip_v17", action="store_true")
    ap.add_argument("--out", default="v18_ppl_results.pt")
    args = ap.parse_args()
    device = args.device

    teacher_sd = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd: teacher_sd = teacher_sd["state_dict"]
    tokens = torch.load(args.tokens, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, tokens.numel() - args.seq_len - 1,
                           (args.n,), generator=g)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    res = {}
    print("\n[eval] baseline")
    ppl_b, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
    res["baseline"] = {"ppl": ppl_b}
    print(f"  PPL {ppl_b:.4f}")

    if not args.skip_v16:
        v16 = torch.load(args.v16, map_location="cpu", weights_only=False)
        substitute_v16(model, teacher_sd, v16, device, args.D)
        ppl, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
        res["v16"] = {"ppl": ppl, "bpw": v16.get("global_bpw")}
        print(f"  v16 PPL {ppl:.4f} ({ppl/ppl_b:.3f}x)")
        reset_teacher(model, teacher_sd)
        del v16; torch.cuda.empty_cache(); gc.collect()

    if not args.skip_v17:
        v17 = torch.load(args.v17, map_location="cpu", weights_only=False)
        substitute_v17(model, teacher_sd, v17, device, args.D)
        ppl, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
        res["v17"] = {"ppl": ppl, "alpha": v17.get("alpha"),
                      "bpw": v17.get("global_bpw"),
                      "overhead_bpw": v17.get("overhead_bpw")}
        print(f"  v17 PPL {ppl:.4f} ({ppl/ppl_b:.3f}x)")
        reset_teacher(model, teacher_sd)
        del v17; torch.cuda.empty_cache(); gc.collect()

    v18 = torch.load(args.v18, map_location="cpu", weights_only=False)
    substitute_v18(model, teacher_sd, v18, device, args.D)
    ppl18, _ = measure_ppl(model, tokens, starts, args.seq_len, device)
    res["v18"] = {"ppl": ppl18, "alpha": v18.get("alpha"),
                  "beta": v18.get("beta"),
                  "bpw": v18.get("global_bpw"),
                  "overhead_bpw": v18.get("overhead_bpw")}
    print(f"  v18 PPL {ppl18:.4f} ({ppl18/ppl_b:.3f}x)")

    print("\n" + "="*70)
    print(f"SUMMARY (n={args.n}, seq_len={args.seq_len})")
    print("="*70)
    print(f"  baseline        PPL {ppl_b:.4f}  1.00x")
    for k in ("v16", "v17", "v18"):
        if k in res:
            r = res[k]; b = (r.get('bpw') or 0) + (r.get('overhead_bpw') or 0)
            print(f"  {k:<15s} PPL {r['ppl']:.4f}  {r['ppl']/ppl_b:.3f}x  "
                  f"bpw {b:.4f}")
    if "v17" in res:
        print(f"  v18/v17 PPL ratio: {ppl18/res['v17']['ppl']:.3f}x")
    torch.save(res, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
