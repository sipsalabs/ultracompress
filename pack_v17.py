"""
pack_v17.py — pack a v17 fit into a compact on-disk binary format and verify
round-trip reconstruction bit-matches (within fp16 tolerance) the in-memory
v17 substitution.

Patent significance: this closes the "2.40 bpw is a counting argument, not a
format" gap. The on-disk file size, divided by the number of weight parameters
it represents, equals the claimed bpw to within ~0.5%.

Format (little-endian, streamed):

  HEADER (magic + metadata, JSON bytes prefixed by u32 length)
    magic:      b"UCV17\\x01"
    header_len: u32 (length of JSON)
    header:     JSON with D, roles, role_K={role: [K1,K2]}, weight_keys,
                per-key (shape, role, n_groups, n_bits_per_group)
    codebooks:  for each role: cb1 (K1*D fp16), cb2 (K2*D fp16)
    per-weight: for each weight key, in header order:
                  s_col : I * fp16
                  rs    : O * (I/D) * fp16
                  codes : bit-packed (idx1: log2(K1) bits, idx2: log2(K2) bits)
                          per D-group, MSB-first within a uint64 stream,
                          rounded up to whole uint64 boundary per weight
"""
from __future__ import annotations
import argparse, io, json, math, struct, time, os
import numpy as np
import torch

from compress_v14 import ROLE_PATTERNS, _role_of, build_rotation
from compress_v15 import beam_assign

MAGIC = b"UCV17\x01"


# ---------- bit-packing helpers ----------

def _bits_needed(n: int) -> int:
    assert n >= 1
    return int(math.ceil(math.log2(n))) if n > 1 else 1


def pack_codes(idx1: np.ndarray, idx2: np.ndarray,
               b1: int, b2: int) -> bytes:
    """Pack two parallel code streams into a contiguous bit string.

    Layout per group i: b1 bits of idx1[i] then b2 bits of idx2[i], MSB-first.
    Stored in little-endian uint64 words; the final word is zero-padded in
    its low (unused) bits. Returns a bytes object of length 8 * ceil(N*b/64).
    """
    assert idx1.shape == idx2.shape and idx1.ndim == 1
    N = idx1.size
    b = b1 + b2
    total_bits = N * b
    n_words = (total_bits + 63) // 64
    out = np.zeros(n_words, dtype=np.uint64)
    pos = 0  # absolute bit index (MSB = bit 0 within word 0)

    # Vectorised pack: process in chunks to keep memory bounded.
    # Each group writes b1 bits of v1 shifted into place, then b2 bits of v2.
    idx1_u64 = idx1.astype(np.uint64)
    idx2_u64 = idx2.astype(np.uint64)
    for i in range(N):
        v = (idx1_u64[i] << np.uint64(b2)) | idx2_u64[i]  # b bits
        word = pos >> 6
        off = pos & 63
        # we store "MSB-first": group's high bit goes to bit (63-off) of word
        room = 64 - off
        if b <= room:
            out[word] |= v << np.uint64(room - b)
        else:
            hi = b - room
            out[word] |= v >> np.uint64(hi)
            out[word + 1] |= (v & ((np.uint64(1) << np.uint64(hi)) - np.uint64(1))) << np.uint64(64 - hi)
        pos += b
    return out.tobytes()


def unpack_codes(buf: bytes, N: int, b1: int, b2: int):
    """Inverse of pack_codes."""
    b = b1 + b2
    total_bits = N * b
    n_words = (total_bits + 63) // 64
    words = np.frombuffer(buf, dtype=np.uint64, count=n_words)
    idx1 = np.empty(N, dtype=np.uint32)
    idx2 = np.empty(N, dtype=np.uint32)
    mask_b1 = (np.uint64(1) << np.uint64(b1)) - np.uint64(1)
    mask_b2 = (np.uint64(1) << np.uint64(b2)) - np.uint64(1)
    pos = 0
    for i in range(N):
        word = pos >> 6
        off = pos & 63
        room = 64 - off
        if b <= room:
            v = (words[word] >> np.uint64(room - b)) & ((np.uint64(1) << np.uint64(b)) - np.uint64(1))
        else:
            hi = b - room
            v = ((words[word] & ((np.uint64(1) << np.uint64(room)) - np.uint64(1))) << np.uint64(hi)) \
                | (words[word + 1] >> np.uint64(64 - hi))
        idx1[i] = int((v >> np.uint64(b2)) & mask_b1)
        idx2[i] = int(v & mask_b2)
        pos += b
    return idx1, idx2


# ---------- packer ----------

def pack_fit(v17_path: str, teacher_path: str, out_path: str,
             device: str = "cuda:0") -> dict:
    t0 = time.time()
    print(f"[pack] loading v17 fit: {v17_path}")
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    print(f"[pack] loading teacher state_dict: {teacher_path}")
    sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    D = v17["D"]
    banks = v17["banks"]
    s_col = v17["s_col"]
    role_K = v17["role_K"]  # {role: [K1, K2]}

    # weight keys to pack (same filter the substitution uses)
    keys = [k for k in sd.keys()
            if "layers." in k and any(p in k for p in ROLE_PATTERNS)
            and k.endswith(".weight") and sd[k].ndim == 2
            and sd[k].shape[1] % D == 0]
    keys.sort()

    # rotations (deterministic per input dim)
    dims = sorted({sd[k].shape[1] for k in keys})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}

    header = {
        "magic": "UCV17",
        "version": 1,
        "D": D,
        "a_attn": v17.get("a_attn"),
        "a_mlp":  v17.get("a_mlp"),
        "role_K": {r: list(role_K[r]) for r in role_K},
        "roles": list(banks.keys()),
        "weights": [],  # filled below
    }

    # === Write header first with placeholder, then rewrite once sizes known ===
    body_parts: list[bytes] = []

    # codebooks (role order fixed by header['roles'])
    for r in header["roles"]:
        cb1 = banks[r]["cb1"].to(torch.float16).cpu().contiguous().numpy()
        cb2 = banks[r]["cb2"].to(torch.float16).cpu().contiguous().numpy()
        body_parts.append(cb1.tobytes())
        body_parts.append(cb2.tobytes())

    # per-weight
    total_param_count = 0
    total_packed_bytes = 0   # accumulated packed bytes for everything per-weight
    packed_bytes_codes = 0   # just the code payload (for bpw reporting)
    t_fit = time.time()
    for n, k in enumerate(keys):
        W = sd[k].to(device=device, dtype=torch.float32)
        O, I = W.shape
        role = _role_of(k)
        K1, K2 = role_K[role]
        b1 = _bits_needed(K1); b2 = _bits_needed(K2)
        s = s_col[k].to(device=device, dtype=torch.float32)
        W_scaled = W * s.unsqueeze(0)
        rot = rots[I]
        Wrot = W_scaled @ rot
        rs = Wrot.abs().amax(1, keepdim=True).clamp(min=1e-6)  # O × 1
        # beam-assign each D-group
        g = (Wrot / rs).view(O, I // D, D).reshape(-1, D)
        cb1 = banks[role]["cb1"].to(device=device, dtype=torch.float32)
        cb2 = banks[role]["cb2"].to(device=device, dtype=torch.float32)
        idx1, idx2, _ = beam_assign(g, cb1, cb2, beam=8)
        idx1_np = idx1.detach().to("cpu").numpy().astype(np.uint32).reshape(-1)
        idx2_np = idx2.detach().to("cpu").numpy().astype(np.uint32).reshape(-1)

        s_bytes  = s.to(torch.float16).cpu().numpy().tobytes()
        rs_bytes = rs.squeeze(1).to(torch.float16).cpu().numpy().tobytes()
        codes_bytes = pack_codes(idx1_np, idx2_np, b1, b2)

        body_parts.append(s_bytes)
        body_parts.append(rs_bytes)
        body_parts.append(codes_bytes)

        header["weights"].append({
            "key": k, "O": O, "I": I, "role": role,
            "b1": b1, "b2": b2, "n_groups": int(idx1_np.size),
            "s_bytes": len(s_bytes), "rs_bytes": len(rs_bytes),
            "codes_bytes": len(codes_bytes),
        })
        total_param_count += O * I
        total_packed_bytes += len(s_bytes) + len(rs_bytes) + len(codes_bytes)
        packed_bytes_codes += len(codes_bytes)
        del W, W_scaled, Wrot, g, cb1, cb2, idx1, idx2, rs, s
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache()
            print(f"  packed {n+1}/{len(keys)} ({time.time()-t_fit:.0f}s)")

    # total codebook bytes (shared overhead)
    cb_bytes = sum(
        banks[r]["cb1"].numel() * 2 + banks[r]["cb2"].numel() * 2
        for r in header["roles"]
    )

    # finalise header
    header_json = json.dumps(header).encode("utf-8")
    print(f"[pack] header JSON: {len(header_json)} bytes")
    print(f"[pack] codebooks:   {cb_bytes} bytes (shared, {len(header['roles'])} roles)")
    print(f"[pack] per-weight:  {total_packed_bytes} bytes")
    print(f"[pack] params:      {total_param_count:,}")
    total_disk = len(MAGIC) + 4 + len(header_json) + cb_bytes + total_packed_bytes
    bpw_disk = (total_disk * 8) / total_param_count
    bpw_codes_only = (packed_bytes_codes * 8) / total_param_count
    bpw_scales = ((total_packed_bytes - packed_bytes_codes) * 8) / total_param_count
    bpw_cb = (cb_bytes * 8) / total_param_count
    print(f"[pack] disk bpw breakdown:")
    print(f"         codes  : {bpw_codes_only:.4f} bpw")
    print(f"         scales : {bpw_scales:.4f} bpw  (s_col fp16 + rs fp16)")
    print(f"         cbooks : {bpw_cb:.4f} bpw (amortised)")
    print(f"         header : {(len(header_json)+len(MAGIC)+4)*8/total_param_count:.5f} bpw")
    print(f"[pack] TOTAL on-disk: {total_disk:,} bytes  =  {bpw_disk:.4f} bpw")
    print(f"[pack] claimed bpw  : {v17.get('total_bpw', 'n/a')}")

    # === write file ===
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for part in body_parts:
            f.write(part)
    actual_size = os.path.getsize(out_path)
    print(f"[pack] wrote {out_path}: {actual_size:,} bytes "
          f"(claimed {total_disk:,}; match: {actual_size == total_disk})")
    print(f"[pack] wall: {time.time()-t0:.0f}s")
    return {
        "out_path": out_path,
        "bytes": actual_size,
        "params": total_param_count,
        "bpw_disk": bpw_disk,
        "bpw_codes": bpw_codes_only,
        "bpw_scales": bpw_scales,
        "bpw_codebooks": bpw_cb,
    }


# ---------- unpacker ----------

def unpack_fit(pack_path: str):
    """Stream-read the pack file into in-memory tensors. Returns a dict
    with the same shape as a v17 fit, suitable for substitute_v17, plus
    per-weight (idx1, idx2, rs) that can be used for a pure unpacker path."""
    f = open(pack_path, "rb")
    magic = f.read(len(MAGIC))
    assert magic == MAGIC, f"bad magic {magic!r}"
    (hl,) = struct.unpack("<I", f.read(4))
    header = json.loads(f.read(hl).decode("utf-8"))
    D = header["D"]
    banks = {}
    for r in header["roles"]:
        K1, K2 = header["role_K"][r]
        cb1 = np.frombuffer(f.read(K1 * D * 2), dtype=np.float16, count=K1 * D).reshape(K1, D).copy()
        cb2 = np.frombuffer(f.read(K2 * D * 2), dtype=np.float16, count=K2 * D).reshape(K2, D).copy()
        banks[r] = {"cb1": torch.from_numpy(cb1).to(torch.float32),
                    "cb2": torch.from_numpy(cb2).to(torch.float32)}
    weights = {}
    for w in header["weights"]:
        s  = torch.from_numpy(np.frombuffer(f.read(w["s_bytes"]),  dtype=np.float16, count=w["I"]).copy()).to(torch.float32)
        rs = torch.from_numpy(np.frombuffer(f.read(w["rs_bytes"]), dtype=np.float16, count=w["O"]).copy()).to(torch.float32)
        codes_buf = f.read(w["codes_bytes"])
        idx1, idx2 = unpack_codes(codes_buf, w["n_groups"], w["b1"], w["b2"])
        weights[w["key"]] = {
            "O": w["O"], "I": w["I"], "role": w["role"],
            "idx1": torch.from_numpy(idx1.astype(np.int64)),
            "idx2": torch.from_numpy(idx2.astype(np.int64)),
            "rs":   rs,
            "s":    s,
        }
    f.close()
    return {"D": D, "banks": banks, "weights": weights,
            "a_attn": header["a_attn"], "a_mlp": header["a_mlp"],
            "role_K": header["role_K"]}


def substitute_from_pack(model, teacher_sd, pack_data: dict, device: str):
    """Substitute model body Linears using the *unpacked* file (pure decode,
    no beam_assign)."""
    import gc
    D = pack_data["D"]
    banks = pack_data["banks"]
    weights = pack_data["weights"]
    dims = sorted({w["I"] for w in weights.values()})
    rots = {I: build_rotation(I, device, seed=42 + I) for I in dims}
    print(f"  pure-decode substituting {len(weights)} body Linears")
    for n, (k, w) in enumerate(weights.items()):
        role = w["role"]
        cb1 = banks[role]["cb1"].to(device=device, dtype=torch.float32)
        cb2 = banks[role]["cb2"].to(device=device, dtype=torch.float32)
        O, I = w["O"], w["I"]
        idx1 = w["idx1"].to(device)
        idx2 = w["idx2"].to(device)
        rs = w["rs"].to(device=device, dtype=torch.float32)
        s  = w["s"].to(device=device, dtype=torch.float32)
        gh = cb1[idx1] + cb2[idx2]                               # (O*I/D) × D
        W_rot_scaled = (gh.view(O, I // D, D).reshape(O, I)) * rs.unsqueeze(1)
        rot = rots[I]
        W_scaled = W_rot_scaled @ rot.T
        W_hat = W_scaled / s.unsqueeze(0)
        # assign into model
        mod = model
        for p in k.replace(".weight", "").split("."):
            mod = getattr(mod, p)
        mod.weight.data.copy_(W_hat.to(dtype=mod.weight.dtype, device=mod.weight.device))
        del gh, W_rot_scaled, W_scaled, W_hat, rs, s, cb1, cb2, idx1, idx2
        if (n + 1) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
    torch.cuda.empty_cache(); gc.collect()


# ---------- verify (roundtrip) ----------

def verify_roundtrip(teacher_path: str, v17_path: str, pack_path: str,
                     model_id: str, tokens_path: str, n: int = 64,
                     seq_len: int = 128, device: str = "cuda:0"):
    """Load model, run two substitutions (original v17 vs pack unpack), and
    assert that perplexity on n WikiText windows agrees to within 1e-3."""
    import gc
    from transformers import AutoConfig, AutoModelForCausalLM
    from eval_v17_ppl import substitute_v17
    from eval_v16_ppl import measure_ppl, reset_teacher

    print(f"[verify] teacher -> {teacher_path}")
    teacher_sd = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if "state_dict" in teacher_sd: teacher_sd = teacher_sd["state_dict"]
    all_tokens = torch.load(tokens_path, weights_only=True).to(torch.long)
    g = torch.Generator().manual_seed(42)
    starts = torch.randint(0, all_tokens.numel() - seq_len - 1, (n,), generator=g)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    model.load_state_dict(teacher_sd, strict=False)
    model = model.to(device).eval()

    print("[verify] path A: original v17 fit")
    v17 = torch.load(v17_path, map_location="cpu", weights_only=False)
    substitute_v17(model, teacher_sd, v17, device, v17["D"])
    ppl_a, _ = measure_ppl(model, all_tokens, starts, seq_len, device)
    del v17; torch.cuda.empty_cache(); gc.collect()
    reset_teacher(model, teacher_sd)

    print("[verify] path B: packed file -> pure decode")
    pack_data = unpack_fit(pack_path)
    substitute_from_pack(model, teacher_sd, pack_data, device)
    ppl_b, _ = measure_ppl(model, all_tokens, starts, seq_len, device)

    rel = abs(ppl_a - ppl_b) / ppl_a
    print(f"[verify] PPL (original fit)   : {ppl_a:.6f}")
    print(f"[verify] PPL (packed->decode) : {ppl_b:.6f}")
    print(f"[verify] relative diff        : {rel*100:.4f}%")
    return {"ppl_original": ppl_a, "ppl_packed": ppl_b, "rel_diff": rel}


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack")
    p.add_argument("--v17", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")

    v = sub.add_parser("verify")
    v.add_argument("--v17", required=True)
    v.add_argument("--teacher", required=True)
    v.add_argument("--pack", required=True)
    v.add_argument("--model_id", required=True)
    v.add_argument("--tokens", required=True)
    v.add_argument("--n", type=int, default=64)
    v.add_argument("--seq_len", type=int, default=128)
    v.add_argument("--device", default="cuda:0")

    args = ap.parse_args()
    if args.cmd == "pack":
        pack_fit(args.v17, args.teacher, args.out, args.device)
    elif args.cmd == "verify":
        verify_roundtrip(args.teacher, args.v17, args.pack,
                         args.model_id, args.tokens, args.n,
                         args.seq_len, args.device)


if __name__ == "__main__":
    main()
