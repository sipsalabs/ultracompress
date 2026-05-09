# Sipsa Labs Phase 0 ICP Ranking — IaaS / Inference Platforms

**Date:** 2026-05-09
**Author:** deep-research agent (commissioned by SuperClaude)
**Scope:** Rank 7 inference / IaaS platforms by likelihood of converting on a $5,000 / 1-week paid POC for 5-bpw lossless compression of 3 customer-requested open-weight models with sub-1% PPL drift, verified UC v3 packs + benchmark JSONs, customer keeps artifacts and deployment rights.
**Strategic context:** Sipsa needs $7-10K/month replacement income. Two POCs in 30 days hits the threshold. Two USPTO provisionals (64/049,511 and 64/049,517) filed April 25 2026 give defensible posture. 20 architectures publicly validated including Mamba SSM. Pre-funding cash mode — no paid attorney, no expensive marketing. Cold-email bandwidth: 5/day.

---

## Executive ranking (summary up front)

| Rank | Company | Score | Why |
|------|---------|-------|-----|
| **1** | **Together AI** | **9.2/10** | Co-built Mamba-3 with CMU/Princeton/Cartesia (Mar 17 2026). Quality-first FP8/INT4 marketing posture creates an opening for "we go further with provable bound." Active model library expansion (150+ models). SVP Eng Albert Meixner runs infra; Dan Fu (former Hazy Research, Mamba/LoLCATs co-author) is SVP Eng — he literally signed the Mamba-3 paper. Mamba SSM validation = unique unlock. |
| **2** | **Fireworks AI** | **8.4/10** | Already runs FP8 quantization in production via FireOptimizer, has an entire blog series defending their methodology — they have *opinions* on quantization quality, which means they will engage on a quality benchmark. Hiring Performance Optimization engineer (Mar 2026). $315M ARR, $4B valuation, very POC-friendly stage. CTO Dmytro Dzhulgakov (PyTorch core ex-Meta) is the right technical contact. |
| **3** | **Replicate** | **7.6/10** | Now part of Cloudflare. 50K+ models in catalog being migrated to Workers AI. Cog containerization is their core IP — UC v3 pack output integrates directly into Cog workflows. Lower deal size threshold than the others. Ben Firshman now Head of AI Platform at Cloudflare; Andreas Jansson Co-founder still active. Risk: post-acquisition decision authority diluted. |
| **4** | **Anyscale** | **6.8/10** | Microsoft Azure first-party deal (Nov 2025) means they need every bit of throughput improvement to look good vs Foundry-native options. Quantization is in their batch-inference docs. Wide-EP launch creates active R&D appetite. Risk: Ray/vLLM-centric culture may prefer to do compression in-house. CEO Keerti Melkote, Co-founder/CTO Philipp Moritz. |
| **5** | **Lambda Labs** | **5.4/10** | New CEO Michel Combes (May 5 2026 — 4 days ago), full leadership reshuffle, "$1.5B Series E, 3GW by 2030" capital story. Inference API ships only FP8 today on Llama/Qwen/Hermes. They're capacity-constrained, not model-quality-constrained — POC pitch lands weaker. CTO Stephen Balaban just shifted from CEO role; busy. |
| **6** | **CoreWeave** | **3.9/10** | Public company, $21B Meta deal, $36B Perplexity-class commitments. Inference is run by hyperscaler customers (Meta, Perplexity, Mistral) on rented GPU — CoreWeave doesn't own the model layer. POC gets rerouted to a customer success motion that doesn't exist for $5K deals. Skip for Phase 0; revisit at $50K+. |
| **7** | **Groq** | **3.2/10** | LPU is custom silicon — TruePoint numerics is hardware-level mixed-precision, not weight quantization. Sipsa's UC v3 pack format (designed for GPU inference engines) doesn't drop into LPU pipelines without architectural rework. POC mismatch: high-effort port, low-confidence delivery. |

**Top 3 to email FIRST tomorrow morning:** Together AI, Fireworks AI, Replicate.

---

## 1. Together AI — Score 9.2 / 10 (LEAD)

### Decision-maker
- **Primary target:** **Dan Fu — SVP Engineering, Together AI**
  - LinkedIn: linkedin.com/in/realdanfu (verify) — formerly Hazy Research / Stanford, **co-author on Mamba and LoLCATs** (the linearization paper Together published October 2024) and on the **Mamba-3 ICLR 2026 paper** announced March 17 2026.
  - Why him: He decides on technical POCs that touch the inference engine and is the person at Together who understands SSM compression at the math layer. He will read a 5-bpw Mamba pack benchmark in 30 seconds and either reject it or escalate it. No one else at Together has higher signal-to-noise for this pitch.
- **Secondary target:** **Albert Meixner — SVP Engineering Infrastructure** (linkedin.com/in/albert-meixner-74679a1b) — owns the inference serving stack; routes serious POCs to engineering.
- **Tertiary:** **Ce Zhang — Co-founder & CTO** — only after Dan Fu has engaged. Going to Ce first is wasted because he'll just forward to Dan.
- **Avoid:** CEO Vipul Ved Prakash. Wrong altitude for $5K POC.

### Current quantization story
- **Together Turbo (FP8):** "quality-preserving quantization" using "incoherence processing." Marketed as "closely matching the quality of full-precision FP16 models" — no public PPL number disclosed. Source: [together.ai/blog/together-inference-engine-2](https://www.together.ai/blog/together-inference-engine-2)
- **Together Lite (INT4):** "excellent quality relative to full precision reference implementations." Same blog. No quantitative bound published.
- **Llama 4 Maverick:** ships as `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` on Together's API — they took Meta's FP8 as-is. Source: [together.ai/models/llama-4-maverick](https://www.together.ai/models/llama-4-maverick)
- **The opening this creates:** Together publicly claims quality preservation but *will not publish numbers*. Sipsa shows up with "we hit sub-1% PPL drift at 5 bpw on 20 architectures including Mamba — here are the JSONs, signed by us, repeatable on your hardware." That's a credibility delta they cannot beat with marketing copy.

### Recent quality complaints
- No specific Together-attributed quality complaints surface in reddit search. Cerebras's competitive blog ([cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks](https://www.cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks)) implies Together quantization underperforms Cerebras's full-precision endpoints on coding benchmarks but stops short of naming specific deltas — Together has air cover but it's thin.
- The community sentiment around quantization broadly: FP8 is "-0.3 to -0.5 MMLU-Pro points," AWQ INT4 is "1.4-1.8 points." Sipsa's 5-bpw at sub-1% PPL drift is *measurably better* and provable on customer hardware.

### POC budget signal
- **Strong.** Together AI **Startup Accelerator** (ongoing) explicitly offers "platform credits, engineering expertise and GTM support" — they pay engineers to work on customer integrations, which means they have a P&L line for partner work. Source: [together.ai/blog/announcing-together-ai-startup-accelerator](https://www.together.ai/blog/announcing-together-ai-startup-accelerator)
- They co-publish papers with academia (Mamba-3 with Princeton/CMU/Cartesia). Co-author / co-blog with Sipsa is a plausible deliverable.
- $50M Series A → Series B+ funded; recent NVIDIA partnership at GTC 2026; not capital constrained.

### Best opening line
> "Hi Dan — saw you co-signed Mamba-3 last month. We just hit sub-1% PPL drift at 5 bpw on a Mamba SSM (one of 20 architectures we've publicly validated, USPTO provisionals filed Apr 25). Quick question: if we compressed three of your most-served models in a week and shipped you the verified UC v3 packs + benchmark JSONs, would $5K cover the engineering cost of evaluating them on your inference stack?"

### Likelihood: 9.2/10
- Mamba SSM validation hooks directly into the work he just shipped. Dan Fu is the perfect ICP for this exact pitch. Even if the answer is "not now," he is highly likely to refer or counter-propose.

### Tightest objection-handling
- **"We already do FP8 in-house, why would we need 5-bpw?"** → "Your FP8 saves ~50% memory; ours saves ~37% on top of that with verifiable PPL drift bound. If FP8 is enough, you don't need us — but you'll want the cached benchmark for when a customer asks 'why not 4-bit.' We'll deliver the comparison numbers on three of your models so you have ammunition either way."
- **"Quantization quality is hard to verify."** → "Agreed. That's why the deliverable is JSON benchmarks across MMLU/GSM8K/HumanEval/PPL on a calibration set you choose, run on your hardware not ours. You can also invalidate the artifacts — we eat the loss."
- **"Why $5K for ~$30K of work?"** → "It's a Phase 0 POC. We need three reference customers fast, and you need to evaluate before we charge what this is worth. The asymmetry is intentional and only available pre-funding."

---

## 2. Fireworks AI — Score 8.4 / 10 (LEAD)

### Decision-maker
- **Primary target:** **Dmytro Dzhulgakov — Co-founder & VP Engineering, Fireworks AI**
  - Background: PyTorch core maintainer at Meta. He owns the inference engine layer. He will instantly understand the POC math and the patent significance.
  - Find via: linkedin.com/in/dzhulgakov — verify exact title; he is one of the 5 co-founders listed on fireworks.ai/team.
- **Secondary target:** **Lin Qiao — CEO** (lqiao@fireworks.ai per RocketReach; verify before sending). She has just been doing keynotes at HumanX (April 6-9 2026) and PyCon US 2026 — she is *publicly framing 2026 as the year of token-consumption growth*, which means cost-per-token reduction is a top-of-mind strategic theme. But for $5K POC she'll forward to Dmytro anyway. Address Dmytro first.
- **Tertiary:** **Benny Chen** (PyTorch infra) and **Chenyu Zhao** (founding CTO) — backup names if Dmytro doesn't respond in 5 days.

### Current quantization story
- **FP8 quantization shipping in production via FireOptimizer.** Available only on H100. Source: [docs.fireworks.ai/models/quantization](https://docs.fireworks.ai/models/quantization)
- **FireAttention:** their custom kernel; claims 4× faster than vLLM "by quantizing with ~no tradeoffs." Source: [fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs](https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs)
- **Quantization-Aware Training (QAT) for DeepSeek R1 / V3:** they've productized a fine-tuning pipeline that does QAT. Source: [fireworks.ai/blog/fine-tuning-deepseek-models](https://fireworks.ai/blog/fine-tuning-deepseek-models)
- **Public quality methodology blog:** [fireworks.ai/blog/fireworks-quantization](https://fireworks.ai/blog/fireworks-quantization) — they write at length defending their numerics. **This is the unlock**: they think about quantization quality publicly. They will engage on a benchmark.

### Recent quality complaints
- **GLM-5 via Fireworks degraded ~April 2026** (GitHub issue pollinations/pollinations#9936): "coherence loss and poor instruction following… community reports on Reddit and Fireworks Discord confirmed similar GLM-5 degradation symptoms… speculation about unannounced quantization or model weight changes." Source: [github.com/pollinations/pollinations/issues/9936](https://github.com/pollinations/pollinations/issues/9936)
- This is *exactly* the pain Sipsa solves: provable, repeatable quantization with PPL bound that customers can audit. They have a recent, public, embarrassing incident.

### POC budget signal
- **Very strong.** $315M ARR (Feb 2026), $4B valuation post-Series C (Oct 2025), 416% YoY growth. Source: [sacra.com/c/fireworks-ai](https://sacra.com/c/fireworks-ai)
- **Hiring:** "Software Engineer, Performance Optimization" — explicitly working on "low-level GPU kernels to large-scale distributed systems" and "implementing cutting-edge optimizations." That role is being recruited *because* they need throughput wins. Source: [job-boards.greenhouse.io/fireworksai](https://job-boards.greenhouse.io/fireworksai)
- **Customer logos:** Cursor, Perplexity, Notion, Sourcegraph, Uber, DoorDash, Shopify, Upwork — they have a procurement motion that handles small contracts.
- **Microsoft Foundry public preview** (Mar 11 2026) — Fireworks now competes inside Azure Foundry against Microsoft-native inference. They need every quality differentiator they can publicize.

### Best opening line
> "Hi Dmytro — read your Fireworks quantization methodology blog last week. Question: if we compressed three of your top-served open models to 5 bpw with sub-1% PPL drift in a week, and shipped you the verified packs + benchmark JSONs you can re-run on your H100s, would $5K cover the engineering eval cost? We just shipped 20 architectures publicly validated, USPTO provisionals filed Apr 25. (And re: the GLM-5 incident last month — our packs come with a signed PPL bound, exactly so end-users have something to audit.)"

### Likelihood: 8.4/10
- Real engineering culture, real budget, recent quality incident creating urgency. Slight downside vs Together: their FireOptimizer pipeline overlaps more directly with Sipsa's value prop, so they may treat us as a competitor not a vendor. Frame as *complementary measurement* not *replacement*.

### Tightest objection-handling
- **"FireOptimizer already does QAT for our customers."** → "Right — and you've published that methodology, which is rare and credible. The question is whether 5 bpw with a sub-1% PPL bound makes that pipeline cheaper or its outputs auditable to your enterprise customers. The POC produces JSON evidence either way. You walk away with leverage on the next customer pricing conversation."
- **"We don't license third-party compression."** → "We're not asking you to ship our format to end-users. The $5K POC delivers UC v3 packs + benchmarks; you decide what to do with them. Customer keeps the artifacts and the deployment rights — same terms we offer everyone."
- **"5 bpw vs FP8 — what's the actual win?"** → "FP8 is 50% memory vs FP16. 5 bpw is ~37% memory vs FP8 with sub-1% PPL drift. On a 70B model that's 17.5GB extra headroom for KV cache. We'll prove the bound on three of your models in a week. If we miss, you don't pay."

---

## 3. Replicate — Score 7.6 / 10 (LEAD)

### Decision-maker
- **Primary target:** **Andreas Jansson — Co-founder, Replicate**
  - Why him not Ben: Ben Firshman is now "Head of AI Platform at Cloudflare" (per his LinkedIn) — post-acquisition his scope is much broader; Andreas is still operating Replicate brand-side and owns the model deployment / Cog roadmap.
- **Secondary target:** **Ben Firshman — Head of AI Platform, Cloudflare** (linkedin.com/in/bfirsh, x.com/bfirsh — actively posting)
- **Tertiary:** A Cloudflare Workers AI PM (need to identify) — for the broader integration play (50K+ models migrating to Workers AI catalog needs compression).

### Current quantization story
- Replicate historically does *not* aggressively quantize — model authors upload via Cog and decide their own precision. Most popular Replicate-hosted LLMs ship in FP16 / BF16. Source: general Replicate hosting model knowledge + [github.com/replicate/cog](https://github.com/replicate/cog)
- **Now part of Cloudflare Workers AI:** Cloudflare is bringing "all 50,000+ models and fine-tunes" into Workers AI, with Cog as the containerization standard. Source: [blog.cloudflare.com/replicate-joins-cloudflare](https://blog.cloudflare.com/replicate-joins-cloudflare)
- **The opening this creates:** 50K+ models × edge inference at Cloudflare's network requires every model to be small. Sipsa's UC v3 pack format dropped into a Cog container is the natural unit of work. There's no Cloudflare/Replicate person whose job is "get 5-bpw lossless on 50K models" — but there should be.

### Recent quality complaints
- No specific Replicate-attributed quality complaints in search results. Their model hosting pattern (author uploads precision they choose) means quality complaints route to model authors, not Replicate itself.
- Cog has GitHub stars and active community; tooling is well-regarded. The pain isn't *complaints*, it's *missing capability*.

### POC budget signal
- **Mixed.** Pre-acquisition Replicate was a 19-person YC-backed company, modest budgets. **Post-Cloudflare** ($65B+ market cap parent), Cloudflare has unlimited POC budget but acquires through procurement, not handshake.
- Cog is open source — they have a culture of working with external contributors. A "we packaged your top 50 LLMs as Cog containers with 5-bpw weights, here are the JSONs" pitch lands.
- **Risk:** post-acquisition decision authority is fluid for ~6 months. Andreas may say "love it, but I need to talk to Cloudflare." That stalls. Mitigate by structuring the pitch as a *Cog community contribution* with a paid scope, not a vendor proposal.

### Best opening line
> "Hi Andreas — congrats on the Cloudflare integration; the 50K+ models migrating to Workers AI is going to bottleneck on memory at the edge. We just hit sub-1% PPL drift at 5 bpw on 20 architectures (USPTO provisionals filed Apr 25). Question: $5K for a 1-week POC where we compress three of your top-traffic models, ship them as Cog-compatible UC v3 packs + benchmark JSONs, and you keep deployment rights — does that fit your roadmap budget, or should I pitch this through Workers AI directly?"

### Likelihood: 7.6/10
- Strong product-market fit (Cog + 50K models + edge inference). Decision-authority risk drops the score. The Cloudflare alternative routing in the email is intentional — gives them an out and signals we know the org chart.

### Tightest objection-handling
- **"We let model authors decide precision."** → "Agreed — that's the right model. We're proposing to do the precision work *for the top-traffic authors* you choose, deliver UC v3 packs back to them, and you keep the catalog migration unblocked. No policy change required."
- **"Cloudflare procurement won't move on $5K."** → "Then bill it as a Cog contribution under Andreas's discretionary engineering budget. The deliverable is open-source-compatible artifacts plus a benchmark methodology. If we deliver well, the next contract is the one Cloudflare procurement processes."
- **"50K models is a lot of work."** → "We're proposing 3 models in week 1. If the JSONs check out you can scope the next 50, the next 500, etc. Phase 0 is just to verify the bound on your stack."

---

## 4. Anyscale — Score 6.8 / 10 (DO NOT EMAIL DAY 1; HOLD FOR DAY 3-4)

### Decision-maker
- **Primary target:** **Philipp Moritz — Co-founder & CTO, Anyscale**
  - He still owns the technical roadmap. Ray/vLLM integration calls go to him.
- **Secondary target:** A Ray Serve LLM PM — they recently launched Wide-EP and Disaggregated Serving with vLLM ([anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm)). Find them on LinkedIn or via Ray docs.
- **Avoid:** CEO Keerti Melkote — hired in July 2024 to run sales motion; not the right altitude for technical POC.

### Current quantization story
- Per [docs.anyscale.com/llm/batch-inference/throughput-optimization/quantization](https://docs.anyscale.com/llm/batch-inference/throughput-optimization/quantization), Anyscale's batch-inference docs cover quantization for memory reduction, but the *implementation* is delegated to vLLM's quantization stack (FP8, AWQ, GPTQ, Compressed Tensors).
- Their position is: **"Use whatever vLLM supports."** They don't have a proprietary quantization method.
- **The opening this creates:** Sipsa's UC v3 pack can ship as a vLLM-compatible loader. The POC deliverable becomes "we got these three of your customer models down to 5 bpw, with verified PPL drift, in a vLLM-loadable format." This makes Anyscale's Ray Serve LLM endpoints faster on Azure (where they're competing with Foundry-native models).

### Recent quality complaints
- No specific Anyscale-attributed quality complaints surface. Their position is platform/infra; quality is "vLLM's job" in the customer's mind.

### POC budget signal
- **Moderate.** Anyscale-on-Azure first-party deal (Nov 2025, Source: [anyscale.com/press/anyscale-collaborates-with-microsoft-to-deliver-ai-native-computing-on-azure](https://www.anyscale.com/press/anyscale-collaborates-with-microsoft-to-deliver-ai-native-computing-on-azure)) creates pressure to differentiate from Microsoft's own offerings — quality + cost wins matter.
- They host conferences (Ray Summit), pay academic researchers, sponsor open source. Discretionary engineering spend exists.
- Risk: their culture is "we contribute to open source vLLM, anyone benefits." They may want this work upstream-PR'd, not vendored.

### Best opening line
> "Hi Philipp — congrats on the Wide-EP launch; the disaggregated serving piece is the right play for the Azure competition. We just hit sub-1% PPL drift at 5 bpw on 20 architectures, vLLM-loadable, USPTO provisionals filed Apr 25. Question: $5K for a 1-week POC where we compress three of your top customer models on the Ray Serve LLM stack, ship UC v3 packs + benchmark JSONs verifiable on your nodes — would that fit your engineering budget, or is there a vLLM-upstream PR path that works better for Anyscale?"

### Likelihood: 6.8/10
- Real fit, real budget, but their open-source-first culture and vLLM dependency mean the buy-vs-build decision skews against us. Hold for Day 3 — let Together / Fireworks responses sharpen the pitch first.

### Tightest objection-handling
- **"Send a vLLM PR."** → "Happy to discuss upstreaming the loader after the POC validates. The $5K isn't for the loader — it's for the engineering work to compress three of *your* specific customer-served models with verified bounds. Loader is a deliverable; the labor and the bound are the value."
- **"Why pay if vLLM has AWQ for free?"** → "AWQ is 4-bit with 1.4-1.8 point MMLU regression. We're 5 bpw with sub-1% PPL drift. You can run both, but only one has a published bound your enterprise customers can audit."

---

## 5. Lambda Labs — Score 5.4 / 10 (DO NOT EMAIL THIS WEEK)

### Decision-maker
- **Primary target:** **Stephen Balaban — Co-founder & CTO, Lambda** (just shifted from CEO to CTO, May 5 2026)
  - linkedin.com/in/sbalaban
- **Secondary target:** Michael Balaban — Chief Product Officer (his brother / co-founder)
- **Avoid for now:** Michel Combes (incoming CEO, May 2026, ex-Sprint/Alcatel-Lucent) — wrong technical context, in onboarding mode.

### Current quantization story
- Lambda Inference API ships **FP8 by default** on Llama 3.3 70B Instruct ($0.20 / M tokens), Llama 3.1, Hermes 3, Qwen 2.5. Source: [lambdalabs.com/blog/inference-release](https://lambdalabs.com/blog/inference-release) and [venturebeat.com/ai/lambda-launches-inference-as-a-service-api-claiming-lowest-costs](https://venturebeat.com/ai/lambda-launches-inference-as-a-service-api-claiming-lowest-costs)
- They market "lowest-cost inference anywhere" — they compete on price, not quality differentiation. Cost-per-token already low without aggressive compression.

### Recent quality complaints
- None surfaced specific to Lambda inference. Their API is recent (Q4 2024 launch); customer base is smaller; complaints would be in private channels.

### POC budget signal
- **Just raised $1.5B Series E, $1B credit facility, scaling to 3GW by 2030.** Source: [businesswire.com/news/home/20260505895594](https://www.businesswire.com/news/home/20260505895594/en/Lambda-Assembles-Leadership-Team-to-Power-Gigawatt-Scale-AI-Infrastructure-for-the-Superintelligence-Era)
- Capital available, but the strategic narrative is *gigawatts of physical infrastructure*, not *bits per weight*. Sipsa's value prop is orthogonal to their public capital story. POC pitch lands as a small distraction.
- Leadership reshuffle (May 5, 2026 — 4 days ago) means org chart is re-stabilizing. Lousy timing.

### Best opening line
> "Hi Stephen — saw the leadership announcement Monday and the 3GW target. Quick question while you're focused on physical scale: if we could deliver verified 5-bpw lossless compression (sub-1% PPL drift, 20 architectures publicly validated) on your top Inference API models, would your team take a $5K / 1-week POC to evaluate? You'd keep all artifacts. USPTO provisionals filed Apr 25."

### Likelihood: 5.4/10
- Wrong moment. Wrong narrative match. Email Lambda only after one of the top-3 converts and you have a logo to lead with.

### Tightest objection-handling
- (Skip — defer this account.)

---

## 6. CoreWeave — Score 3.9 / 10 (SKIP PHASE 0)

### Decision-maker
- **Peter Salanki — Co-founder & CTO, CoreWeave**
  - Public-company ($CRWV) CTO; covered by IR; not accessible to a $5K POC pitch through cold email.
- Other named execs: Michael Intrator (CEO), Brian Venturo (CSO), Brannin McBee (CDO), Nitin Agrawal (CFO).

### Current quantization story
- CoreWeave is a **GPU infrastructure provider**, not a model serving company. Their inference customers (Meta, Perplexity, Mistral, etc.) make the quantization decisions on their own model weights running on CoreWeave-rented hardware.
- Their MLPerf 6.0 wins ([coreweave.com/news/coreweave-delivers-leading-inference-performance-in-mlperf-r-benchmark](https://www.coreweave.com/news/coreweave-delivers-leading-inference-performance-in-mlperf-r-benchmark)) demonstrate hardware throughput, not model quality.

### Recent quality complaints
- Not applicable — they don't own the model layer.

### POC budget signal
- **Massive capital, but routed through enterprise procurement.** Meta committed $21B+ ([thenextweb.com/news/meta-coreweave-21-billion-ai-cloud-deal](https://thenextweb.com/news/meta-coreweave-21-billion-ai-cloud-deal)). Perplexity multi-year deal Mar 4 2026. Not the buyer for a $5K POC.
- They could be a *partner* (referrer) once Sipsa has Meta-tier customer logos, but for Phase 0 they are wrong.

### Best opening line
- (Skip — wrong sales motion.)

### Likelihood: 3.9/10
- Wrong layer of the stack. Revisit at $50K-$500K deal size when you have a Meta/Perplexity/Mistral-tier customer asking CoreWeave to host 5-bpw packs at scale.

---

## 7. Groq — Score 3.2 / 10 (SKIP PHASE 0)

### Decision-maker
- Gavin Sherry — VP Engineering, Groq.
- Founder/CEO Jonathan Ross is on the conference / capital-raising circuit.

### Current quantization story
- **Groq does NOT do weight quantization in the GPU sense.** Their LPU uses **TruePoint numerics**: weights stored at INT8/FP8 but matmul accumulators are 100-bit, with selective post-quantization. Source: [groq.com/blog/inside-the-lpu-deconstructing-groq-speed](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- Their inference engine is hardware-pipelined (deterministic compiler, no caches in the GPU sense). UC v3 packs designed for vLLM/SGLang/TGI on GPU don't drop into the LPU pipeline.

### Recent quality complaints
- Cerebras blog ([cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks](https://www.cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks)) implies Groq quantization underperforms Cerebras full-precision on coding benchmarks — but this is a SambaNova / Cerebras / Groq squabble where each picks favorable benchmarks. No clean quality complaint.
- Community: "Please bring back DeepSeek models to the Groq API" feature request thread. Source: [community.groq.com/t/please-bring-back-deepseek-models-to-the-groq-api/663](https://community.groq.com/t/please-bring-back-deepseek-models-to-the-groq-api/663)

### POC budget signal
- $750M raise at $6.9B valuation (Oct 2025). $1.5B from Saudi Arabia for HUMAIN. ALLaM Arabic model partnership. *Massive* capex on data centers. Source: [winbuzzer.com/2025/10/03/groq-charts-major-data-center-expansion](https://winbuzzer.com/2025/10/03/groq-charts-major-data-center-expansion-after-raising-750m-at-6-9b-valuation-xcxwbn/)
- They have *hardware* budget, not software-vendor budget. POC pitch lands as out-of-scope.

### Best opening line
- (Skip — architectural mismatch.)

### Likelihood: 3.2/10
- The hardware/software boundary is wrong. UC v3 needs a GPU inference engine to be useful; Groq doesn't have one. Revisit only if Sipsa develops an LPU-compatible pack format (months of work, not a POC).

---

## Final Day-1 Email Plan (5/day budget)

**Tomorrow morning (Saturday May 10, 2026):**

| Slot | Target | Subject Line | Notes |
|------|--------|--------------|-------|
| 1 | **Dan Fu — SVP Engineering, Together AI** | "5-bpw Mamba SSM benchmark — POC offer for 3 of your models" | Lead. Highest signal; Mamba-3 hook. |
| 2 | **Dmytro Dzhulgakov — VP Engineering, Fireworks AI** | "5-bpw / sub-1% PPL drift — POC for FireOptimizer comparison" | Frame as complementary measurement, not competition. |
| 3 | **Andreas Jansson — Co-founder, Replicate** | "5-bpw Cog packs for Workers AI top 3 models" | Use the Cloudflare-routing line; signal org-chart awareness. |
| 4 | (Hold) | — | Reserve; wait for response signal. |
| 5 | (Hold) | — | Reserve. |

**Day 2 (Sunday May 11) — only if no responses by EOD Saturday:**
- Slot 4: Lin Qiao (Fireworks CEO) as a backstop to Dmytro
- Slot 5: Albert Meixner (Together SVP Eng Infra) as a backstop to Dan Fu

**Day 3 (Monday May 12) — only if still no responses:**
- Add: Philipp Moritz (Anyscale CTO) — full-court press on inference platforms
- Add: Ben Firshman at Cloudflare directly

**Skip entirely this week:** Lambda, CoreWeave, Groq.

---

## Cross-cutting strategic observations

1. **The Mamba-3 angle is the single strongest hook in the whole pitch deck.** Together AI literally co-built Mamba-3 (Mar 17 2026, ICLR 2026). Sipsa has *publicly validated 5-bpw on Mamba SSM*. This is a once-in-a-quarter alignment. Lead with it. Do not waste the hook on a non-Together email.

2. **The GLM-5 / Fireworks Apr 2026 quality incident is the second-strongest hook.** Don't be cute with it — frame it as "your customers want auditable PPL bounds, here's a way to ship them."

3. **The Cloudflare/Replicate 50K-model migration is the most under-priced opportunity.** The *strategic* size is enormous (50K models × edge bandwidth × Cloudflare's 320-city network); the *deal* size is small (one Cog contribution). It might be the highest-leverage Phase 0 win even if it doesn't pay $5K — trial as a paid Cog contribution if cash needed.

4. **Don't pitch CEOs.** None of the top-3 targets is a CEO. CEOs forward $5K POCs to engineering anyway, and the forward-loop kills 2-3 days. Hit engineering directly with technical fluency.

5. **The signed PPL bound is the actual product.** Every prospect is sophisticated enough to know FP8 is "good enough for most." What they cannot get from Meta / NVIDIA / vLLM is a *vendor who signs the bound and re-runs the benchmark on their hardware*. The deliverable is *trust*, not bytes.

6. **Two POCs in 30 days, not three.** $5K × 2 = $10K = above the $7-10K floor. Do not expand the funnel beyond the top-3 until at least one converts. Cold-email batch fatigue is real and the founder is solo.

---

## Source URLs (for follow-up)

- Together AI Mamba-3 announcement: [together.ai/blog/mamba-3](https://www.together.ai/blog/mamba-3)
- Together Inference Engine 2: [together.ai/blog/together-inference-engine-2](https://www.together.ai/blog/together-inference-engine-2)
- Together LoLCATs: [together.ai/blog/linearizing-llms-with-lolcats](https://www.together.ai/blog/linearizing-llms-with-lolcats)
- Together at GTC 2026: [together.ai/blog/together-ai-at-nvidia-gtc-2026](https://www.together.ai/blog/together-ai-at-nvidia-gtc-2026)
- Together Startup Accelerator: [together.ai/blog/announcing-together-ai-startup-accelerator](https://www.together.ai/blog/announcing-together-ai-startup-accelerator)
- Fireworks quantization docs: [docs.fireworks.ai/models/quantization](https://docs.fireworks.ai/models/quantization)
- Fireworks quantization methodology: [fireworks.ai/blog/fireworks-quantization](https://fireworks.ai/blog/fireworks-quantization)
- Fireworks FireAttention: [fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs](https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs)
- Fireworks Llama 4 Maverick optimization: [fireworks.ai/blog/llama4-maverick](https://fireworks.ai/blog/llama4-maverick)
- Fireworks Microsoft Foundry launch: [azure.microsoft.com/en-us/blog/introducing-fireworks-ai-on-microsoft-foundry-bringing-high-performance-low-latency-open-model-inference-to-azure/](https://azure.microsoft.com/en-us/blog/introducing-fireworks-ai-on-microsoft-foundry-bringing-high-performance-low-latency-open-model-inference-to-azure/)
- GLM-5 / Fireworks degradation: [github.com/pollinations/pollinations/issues/9936](https://github.com/pollinations/pollinations/issues/9936)
- Replicate joins Cloudflare: [blog.cloudflare.com/replicate-joins-cloudflare](https://blog.cloudflare.com/replicate-joins-cloudflare)
- Cloudflare AI Platform announcement: [blog.cloudflare.com/ai-platform/](https://blog.cloudflare.com/ai-platform/)
- Anyscale Wide-EP launch: [anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm)
- Anyscale on Azure: [anyscale.com/press/anyscale-collaborates-with-microsoft-to-deliver-ai-native-computing-on-azure](https://www.anyscale.com/press/anyscale-collaborates-with-microsoft-to-deliver-ai-native-computing-on-azure)
- Lambda new CEO: [businesswire.com/news/home/20260505895594/en/](https://www.businesswire.com/news/home/20260505895594/en/Lambda-Assembles-Leadership-Team-to-Power-Gigawatt-Scale-AI-Infrastructure-for-the-Superintelligence-Era)
- Lambda Inference API launch: [lambdalabs.com/blog/inference-release](https://lambdalabs.com/blog/inference-release)
- CoreWeave Perplexity deal: [investors.coreweave.com/news/news-details/2026/CoreWeave-Announces-Agreement-to-Power-Perplexitys-AI-Inference-Workloads](https://investors.coreweave.com/news/news-details/2026/CoreWeave-Announces-Agreement-to-Power-Perplexitys-AI-Inference-Workloads/default.aspx)
- CoreWeave MLPerf 6.0 win: [coreweave.com/news/coreweave-delivers-leading-inference-performance-in-mlperf-r-benchmark](https://www.coreweave.com/news/coreweave-delivers-leading-inference-performance-in-mlperf-r-benchmark)
- Groq LPU TruePoint: [groq.com/blog/inside-the-lpu-deconstructing-groq-speed](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- Groq DeepSeek R1 launch: [groq.com/blog/groqcloud-makes-deepseek-r1-distill-llama-70b-available](https://groq.com/blog/groqcloud-makes-deepseek-r1-distill-llama-70b-available)
- Cerebras competitive comparison: [cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks](https://www.cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks)

---

## Open questions / further research needed

1. **Verify Dan Fu's current title at Together AI** (LinkedIn or together.ai/team page) — assumed SVP Engineering based on search; confirm before sending.
2. **Find Dmytro Dzhulgakov's Fireworks email** — RocketReach has Lin Qiao's; may not have Dmytro's. Try `dmytro@fireworks.ai`, `dzhulgakov@fireworks.ai`, or LinkedIn DM.
3. **Confirm Andreas Jansson's current Replicate role post-Cloudflare** — last data point: Co-founder, still operating Replicate brand. Cross-check his LinkedIn.
4. **Identify the Workers AI PM at Cloudflare** for Day-3 backup outreach — try `careers.cloudflare.com` for recently-announced PM hires + LinkedIn search.
5. **Get Together AI's actual PPL/MMLU numbers on Llama 4 Maverick FP8** — they have not published; if we get them via inquiry, that becomes the calibration baseline for the POC pitch.
6. **Cog GitHub stars / contributor count** — quick check on github.com/replicate/cog to size the community before pitching Andreas as "let me contribute to your ecosystem."
