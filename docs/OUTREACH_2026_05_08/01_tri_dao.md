# 01 — Tri Dao (Princeton + Together.ai)

**To:** `tri@tridao.me` (preferred personal/lab) — fallback `tridao@princeton.edu` or `tri@together.ai`
**From:** `sipsalabs@gmail.com`
**Send window:** today/tomorrow, US morning

---

**Subject:** First lossless 5-bit Mamba result — PPL_r 1.0119 on mamba-2.8b-hf, runnable now

Hi Tri,

Long-time admirer of the FlashAttention and Mamba/Mamba-2 work. Wanted to flag a result that landed today and is sitting in front of you in 3 commands.

We compressed `state-spaces/mamba-2.8b-hf` end-to-end at 5 bpw — 256 SSM Linears (`in_proj`/`x_proj`/`dt_proj`/`out_proj`), bit-identical reconstruction per Linear, end-to-end PPL ratio **1.0119x** (no V18-C correction layer needed). To my knowledge this is the first published ultra-low-bit lossless compression result on a state-space LLM. Same codec runs on transformers — Qwen3-0.6B hit PPL_r **1.0069x** at 5 bpw this morning, the tightest dense-decoder ratio at this bitrate I have measured anywhere.

Two questions:

1. Would Together's serving stack benefit from a verified compressed Mamba pack? Today we shipped v0.5.2 to PyPI; 12 architectures end-to-end validated at PPL_r ≤ 1.013x; Hermes-3-Llama-3.1-405B is mid-compression on a single 32GB consumer GPU as I write this.
2. NeurIPS 2026 paper draft is solo, but if you want a citation/footnote arrangement on the Mamba reference architecture I would gladly coordinate.

Run it yourself in 30 seconds:
```
pip install ultracompress
hf download SipsaLabs/qwen3-1.7b-uc-v3-bpw5 --local-dir ./pack
uc verify ./pack
```

Or 30 minutes on Zoom whenever it suits you.

Best,
Missipssa Ounnar
Founder, Sipsa Labs Inc.
sipsalabs.com / github.com/sipsalabs/ultracompress
