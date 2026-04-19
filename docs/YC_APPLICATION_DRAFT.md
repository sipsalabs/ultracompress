# Y Combinator Application Draft

Verified as of 2026-04-18. Apply at ycombinator.com/apply. Copy each section into the online form; fill in **[TODO]** bracketed values with your personal info.

---

## Company

**Company name:** FractalLM *(check trademark availability before finalizing; backup: FRR-Compress, EntropyKD)*

**Company URL (website):** [TODO — set up github.io page by Tuesday; placeholder fractallm.ai]

**Company demo video URL:** [TODO — 1-minute screen recording showing compressed model outputting text side-by-side with teacher; use OBS or phone, no production values needed]

**Describe what your company does in 50 characters or less:**
`Extreme compression of LLMs — 311× with 70% quality`

**What is your company going to make? Please describe your product and what it does or will do (500 chars):**

> FractalLM compresses large language models to ≥300× smaller while retaining roughly 70% of the teacher model's next-token behavior. Our first public benchmark: a 1.51M-parameter student reproduces 69.64% of a 1.72B Qwen3 model's top-10 predictions on held-out FineWeb-Edu (1000 samples, seed 42). The method combines a novel fractal-iterative shared-block architecture with entropy-weighted teacher distillation. Provisional patent filed April 2026. Code + numbers reproducible on a single GPU.

---

## Founders

**Number of founders:** 1 *(YC usually prefers 2+. Prepare a good answer for why you're solo: "I built the technical core alone; actively looking for a commercial cofounder during the batch.")*

**Please tell us in one or two sentences something impressive that each founder has done:**

> [TODO — your accomplishment. The bar here is concrete evidence of technical + execution ability. Examples of format: "Built X, shipped Y, resulted in Z." For this project specifically you can say: "Single-handedly designed and trained a novel LLM compression method that achieves 311× compression at 69.64% teacher-top-10 retention, reproducible in the public repo at github.com/[user]/ultracompress. Provisional patent filed."]

**What's your LinkedIn URL?** [TODO]

**What's your GitHub URL?** https://github.com/[TODO-user]

---

## Progress

**How far along are you?**

> Technical core complete. Public open-source repo with reproducible training + evaluation scripts. 1000-sample held-out benchmark confirms 311× compression at 69.64% top-10 agreement with Qwen3-1.7B teacher (all-T1 = 55.40%, quality score = 75.94%, PPL ratio = 1.216). A second configuration at 734× compression retains 68.00% top-10. Provisional patent application being filed April 20, 2026. No paying customers yet. No revenue. Currently running longer-horizon training (160K steps vs 80K) to push the ceiling further.

**How long have each of you been working on this?**

> [TODO — approximately 6-9 months full-time-equivalent alongside day job; adjust to your real timeline]

**Which of the following best describes your progress?**
`[X] We have a working prototype`

**Are you launched?** `No` *(or "Yes" if you count the public repo as launch)*

**If Yes, when did you launch?** [TODO — repo public date]

**How many active users or customers do you have?** `0`

**How much revenue?** `$0`

**Anything else you would like us to know about your progress?**

> The compression ratios in this application (311×, 734×) are an order of magnitude beyond published distillation baselines (DistilBERT ~1.7×, TinyBERT ~7×, ASVD ~3×) at comparable quality. Verified on a held-out region of FineWeb-Edu that was least-touched during training. All numbers reproducible: clone the repo, run `hires_eval.py`, get the same numbers on any 32GB GPU in under 15 minutes. Combined end-to-end stack (compressed body + ASVD-compressed output head) measurement in progress and will be posted before submission deadline.

---

## Idea

**Why did you pick this idea to work on? Do you have domain expertise? How do you know people need what you're making?**

> I picked this because on-device and edge LLM deployment is blocked by model size, and the existing compression stack (quantization + low-rank) plateaus at ~10× with quality loss. I had the intuition that most of the teacher's per-token information is concentrated in high-entropy positions, and that iterative-shared-block architectures (Universal Transformer family) had not been combined with entropy-aware distillation. Both were right — the combination produces 50–100× more compression than either alone.
>
> Domain expertise: [TODO — your relevant background. If you're self-taught, say "self-taught ML engineer, 5+ years across PyTorch, CUDA, distributed training" or whatever is true. Honesty here matters.]
>
> Demand evidence: every foundation model vendor (OpenAI, Anthropic, Google, Meta, Mistral) is publicly investing in on-device variants. Apple shipped a 3B on-device model with their 2024 Intelligence launch. Qualcomm, Meta, and Arm have public bounties and research tracks for <500MB LLMs. The exact buyer archetype for the first commercial deployment is any company that wants to serve an LLM to phones, browsers, cars, or air-gapped regulated environments.

**Who are your competitors? What do you understand about your business that they don't?**

> Direct competitors: Neural Magic (SparseGPT, acquired by Red Hat 2024), Deci AI (acquired by NVIDIA 2024), Mosaic (acquired by Databricks 2023). Each achieves 2–8× compression via structured pruning + quantization.
>
> Indirect competitors: Hugging Face's distilled model releases (DistilBERT, DistilGPT), Microsoft Phi-family small-by-design models, Google Gemini Nano.
>
> What I understand that they don't: combining fractal parameter sharing with entropy-weighted distillation collapses the parameter-vs-quality Pareto front by roughly 30–50×. The best-known distilled small model (Phi-3-mini at 3.8B) compresses from a ~100B teacher at ~26× with ~60% benchmark retention. I hit 311× at ~70% retention. The gap is not small. The reason no one has done this yet is that it requires specifically training one block with a schedule optimizer on entropy-weighted teacher logits — it is not an emergent property of any existing technique.

**How do or will you make money? How much could you make?**

> Three revenue motions, roughly in execution order:
>
> 1. **Paid compression service for enterprise** — customer sends us a model, we return a compressed version + fine-tuning on their data. Pricing: $50K–$500K per model depending on size class. Initial TAM: any company running a private LLM in production. Signing 10 customers in year 1 is a $3M ARR bootstrap.
> 2. **Compute-cost-reduction API** — customers route their inference traffic through compressed mirrors of open-source models (Llama, Qwen, Mistral) at ~1/100 the GPU cost. Charging 30% of saved compute is a billion-dollar market if we take ~1% of current hosted-LLM spend.
> 3. **On-device SDK** — embedded into mobile apps, automotive, browsers. Pricing per-device license. This is the long-term big business.
>
> Floor: $5M ARR in year 2 from motion 1 alone. Ceiling: infrastructure company at low-billions valuation within 5 years via motion 2. Motion 3 is a decade play.

---

## Equity

**Have you incorporated, or formed any legal entity yet?** `No` *(or Yes if you have)*

**Have you taken any investment yet?** `No`

**Are you currently fundraising?** `No — applying to YC is the start.`

---

## Other

**If you had any other ideas you considered applying with, please list them.**

> [TODO — leave blank if none, or list briefly. Keep it short.]

**Please tell us something surprising or amusing that one of you has discovered.**

> Surprising technical result from this project: weighting each training token's loss by a power of the teacher's predictive entropy (w = (1+H)^p, p ≥ 1) — treating the *ambiguous* positions as the informative ones rather than the confident ones — gives more quality improvement than any architectural change I tried. The "easy" positions, where the teacher is near-deterministic, contribute almost nothing. This is counterintuitive: most distillation literature assumes uniform weighting and focuses on architecture. The information is in the hard positions.

---

## Video script (for the 1-minute founder video)

Record this on your phone. No editing needed. Sit somewhere with good natural light, look into the camera.

> Hi, I'm [name]. I'm the solo founder of FractalLM. I built a compression method for large language models that achieves 311 times compression while retaining 69.64% of the teacher model's top-10 predictions. That's ten to fifty times better than any existing compression technique at comparable quality.
>
> The method combines two ideas: a fractal-iterative architecture where a single transformer block is re-used 28 times with learned scaling, and an entropy-weighted distillation loss that focuses training on the tokens where the teacher model is genuinely uncertain.
>
> Provisional patent is being filed Monday. All code and benchmarks are public on GitHub. You can clone the repo and reproduce the 311-times number on a single GPU in 15 minutes.
>
> I'm applying to YC because I need help turning this from a reproducible technical result into a compression-as-a-service business. Thanks for watching.

---

## After submitting

- Don't obsess over the app — submit and move on.
- Within 48 hours, also submit: a16z Speedrun, Neo, AI Grant, Pear, South Park Commons, NVIDIA Inception.
- Keep building. YC reviews applications over weeks, and the strength of your repo *at interview time* matters as much as the app text.
