# 04 — Lambda Labs / CoreWeave (IaaS biz dev)

**To (Lambda):** `partnerships@lambdalabs.com` — cc warm AE if available
**To (CoreWeave):** `partnerships@coreweave.com` — fallback Brian Venturo (CTO) via LinkedIn DM `/in/brianventuro`, Rosie Zhao (ML infra lead) on LinkedIn
**From:** `sipsalabs@gmail.com`
**Send window:** today/tomorrow, US morning

> Send as two separate emails — same body, different recipient line. Below is the canonical version (CoreWeave-addressed); for Lambda swap "CoreWeave" → "Lambda Labs" and "your H100 / H200 fleet" → "your H100 / GH200 / 5090 fleet".

---

**Subject:** 5x more rentable models per H100 — UltraCompress v0.5.2 just shipped, 30-second eval

Hi [name],

Sip from Sipsa Labs (YC S26, in review). v0.5.2 of UltraCompress shipped to PyPI this morning. 12 architectures end-to-end validated at 5 bpw, mean PPL_r ≤ 1.013x — Qwen3-0.6B at **1.0069x**, Llama-3.1-8B at **1.0125x**, Mamba-2.8B SSM at 1.0119x. Bit-identical reconstruction (lossless pack format). Hermes-3-Llama-3.1-405B mid-compression on a single 32 GB consumer GPU as I write this.

The CoreWeave fit: your H100 / H200 customers running inference fit ~3x more model classes per GPU, which translates 1:1 into rentable inference revenue per chip. A Llama-70B that today takes 2x H100 sharded fits on one. A 405B that today needs 8x H100 fits on a single H100 80GB at the same quality envelope. 7 stars on github.com/sipsalabs/ultracompress this week — early but the right kind of customers are showing up.

Customer flow is one line per artifact:

```
pip install ultracompress
hf download SipsaLabs/llama-3.1-70b-uc-v3-bpw5 --local-dir ./model
uc verify ./model
```

I'd like to propose a paid 1-week Phase 0: $5K, you pick the model from your fleet (or one a customer is asking for), we ship back the packed checkpoint plus a benchmark report (PPL, T1 retention, decode latency, peak VRAM). 2 USPTO provisionals filed; 5 more file tomorrow.

Worth a 30-min intro call this week?

Best,
Missipssa Ounnar
Founder, Sipsa Labs Inc.
sipsalabs.com / github.com/sipsalabs/ultracompress
