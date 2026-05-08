# 02 — Albert Gu (Cartesia AI cofounder, Mamba co-creator)

**To:** `albert@cartesia.ai` (founder address; fallback `hello@cartesia.ai`, `agu@andrew.cmu.edu`)
**From:** `sipsalabs@gmail.com`
**Send window:** today/tomorrow, US morning

---

**Subject:** Lossless 5-bit Mamba-2.8B pack — PPL_r 1.0119x, useful for Cartesia edge?

Hi Albert,

We compressed `state-spaces/mamba-2.8b-hf` end-to-end at 5 bpw today: 256 SSM Linears (`in_proj` / `x_proj` / `dt_proj` / `out_proj`), bit-identical reconstruction per Linear, end-to-end PPL ratio **1.0119x** on FineWeb-edu held-out — no correction overlay needed. To my knowledge it is the first published ultra-low-bit lossless compression on a state-space LLM.

The Cartesia angle: same codec runs on transformers and SSMs alike, and a single 32 GB consumer GPU does the compression. A Cartesia voice/SSM stack shipping to on-device or thin-edge targets gets ~3x weight footprint reduction with bit-identical reconstruction — no "approximate quality" caveat for the audio team to argue about.

Run it yourself in 30 seconds:

```
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./pack
uc verify ./pack
```

Mamba pack uploads tonight to `SipsaLabs/mamba-2.8b-uc-v3-bpw5`. Happy to point Cartesia at any specific Mamba-2 variant for a paid 1-week Phase 0 if useful.

30-min Zoom this week or next?

Best,
Missipssa Ounnar
Founder, Sipsa Labs Inc.
sipsalabs.com / github.com/sipsalabs/ultracompress
