"""Quick sanity: does the decoder reproduce v16's reported rel-W (0.0577)?"""
import torch, sys, math
from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from eval_v16_ppl import _reconstruct_v16

device = "cuda:0"
teacher_sd = torch.load("qwen3_1.7b_cache.pt", map_location="cpu", weights_only=False)
v16 = torch.load("v16_result.pt", map_location="cpu", weights_only=False)
D = 8

keys = [k for k in teacher_sd if "layers." in k and any(p in k for p in ROLE_PATTERNS)
        and k.endswith(".weight") and teacher_sd[k].ndim == 2
        and teacher_sd[k].shape[1] % D == 0]
print(f"num body linears: {len(keys)}")

dims = sorted({teacher_sd[k].shape[1] for k in keys})
rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

rws_by_role: dict[str, list[float]] = {}
for n, k in enumerate(keys):
    role = _role_of(k)
    bank = v16["banks"][role]
    W_orig = teacher_sd[k]
    W_q = _reconstruct_v16(W_orig, role, bank, D, rots[W_orig.shape[1]], device)
    W_o = W_orig.to(torch.float32)
    W_q32 = W_q.to(torch.float32)
    rw = ((W_o - W_q32).pow(2).mean() / W_o.pow(2).mean()).item()
    rws_by_role.setdefault(role, []).append(rw)
    if n < 5 or (n + 1) % 50 == 0:
        print(f"  [{n+1:3d}] {k:70s} role={role:10s} rel-W={rw:.4f}")

print("\nper-role:")
all_rw = []
for role, rs in rws_by_role.items():
    print(f"  {role:<12s} mean {sum(rs)/len(rs):.4f}  max {max(rs):.4f}  (n={len(rs)})")
    all_rw.extend(rs)
print(f"  global       mean {sum(all_rw)/len(all_rw):.4f}  max {max(all_rw):.4f}")
print(f"\nv16 report:    mean 0.0577  max 0.0600")
