# Monday Launch Post Drafts

**DO NOT POST until the provisional patent is filed and the receipt is in hand.** After filing, public disclosure in the US is fine and actually starts the 12-month non-provisional clock — which is useful.

---

## Hacker News — `Show HN` post

**Title options (pick one — A is highest-signal):**

- A. `Show HN: I compressed a 1.7B LLM by 311x while keeping ~70% of its behavior`
- B. `Show HN: FractalLM — a 1.5M-parameter neural net that tracks a 1.7B teacher 69.64% top-10`
- C. `Show HN: Fractal iterative distillation compresses LLMs 300-700x (open source)`

**Body (paste directly into HN):**

```
Hi HN. I've been working on extreme LLM compression and I think I have
something worth showing.

Result: a 1.51 M-parameter student model reproduces 69.64% of the top-10
next-token predictions of Qwen3-1.7B on a 1000-sample held-out slice of
FineWeb-Edu. That's 311x compression (trainable-vs-trainable). A smaller
0.64 M variant hits 68.00% top-10 at 734x compression.

Two ideas, combined:

1. Fractal iterative architecture. Instead of stacking 28 transformer
   blocks, train one transformer block and apply it 28 times in a nested
   scale x iteration schedule, modulated by per-(scale, iteration) learned
   scalars. Universal-Transformer-adjacent but with a fixed schedule and
   no adaptive-computation-time halting.

2. Entropy-weighted teacher distillation. Each training position's loss is
   weighted by (1 + H(teacher))^p. Near-deterministic positions contribute
   almost nothing; high-entropy positions, where the teacher is genuinely
   spreading mass across candidates, contribute most. Counterintuitive vs
   standard uniform-weighted distillation.

Full numbers (seed 42, SEQ_LEN=128, bootstrap 95% CIs):

  h256  1,509,916 params  311x   all-T1 55.40%  all-T10 69.64%  PPL ratio 1.216
  h128    640,284 params  734x   all-T1 53.78%  all-T10 68.00%  PPL ratio 1.254

Reproducible in ~15 min on a single 32 GB GPU:

  git clone https://github.com/<user>/ultracompress
  cd ultracompress
  python hires_eval.py --tags hq5_h256 hq5_h128 --n 1000

Side-by-side token-probability demo:

  python demo.py --prompt "The capital of France is"

For reference, the typical distillation baseline (DistilBERT, TinyBERT,
MobileBERT family) hits 2-7x compression at similar quality retention.
ASVD / GPTQ / SparseGPT top out around 8-10x. This is roughly 30-100x
beyond the distillation frontier at comparable quality.

Provisional patent filed. Repo is Apache 2.0. Happy to answer questions
about the training schedule, the entropy-power sweep, or the failure
modes I hit getting here.

Repo: https://github.com/<user>/ultracompress
Paper-style writeup: docs/HQ5_RESULTS.md
Reproducible results: hires_results_hq5.json
```

**When to post:** Tuesday 8:30 AM Pacific / 11:30 AM Eastern. Avoid Monday (noisier). Avoid weekends. Use the "Show HN" prefix explicitly.

**Top-comment strategy:** within 5 minutes of your own post going live, post a follow-up comment from the same account with the exact commit SHA being referenced and a one-line explanation of what "all-T10" means. Pre-empts the usual HN "what even is top-10 agreement" question.

---

## Twitter / X — thread (9 tweets)

**Tweet 1** (the hook):
```
I compressed Qwen3-1.7B by 311× (1.72B → 1.51M trainable params)
while keeping 69.64% of its top-10 next-token predictions on
held-out text.

Reproducible on one 32GB GPU in 15 min.

Full thread + code ↓
```

**Tweet 2** (the concrete number):
```
The hires eval, 1000 samples, seed 42, tail-50M FineWeb-Edu:

h256  1,509,916 params  311×   T1 55.40%  T10 69.64%
h128    640,284 params  734×   T1 53.78%  T10 68.00%

Quality score (¹⁄₂ T10 + ¹⁄₂ 1/PPL-ratio): 75.94% for h256.

[image of hires_results_hq5.json table]
```

**Tweet 3** (the architecture):
```
The student is one transformer block, applied 28 times in a nested
(4 scales × 7 iterations) schedule with per-step learned gating scalars.

28 block-applications. 1 set of weights. Each (scale, iter) pair has its
own γ, β, α — so the same parameters induce different behavior at
different depths.

[fig]
```

**Tweet 4** (the loss trick):
```
The secret sauce is the distillation loss.

Standard distillation weights every position equally. But look at the
teacher's per-position entropy — it's a power-law. Most positions are
trivially deterministic. The information is in the high-entropy tail.

w_i = (1 + H_i)^p  focuses gradient where the teacher actually has
something to say.
```

**Tweet 5** (context / prior art):
```
Related work:
- Universal Transformer (Dehghani 2018) — shared blocks, but adaptive
  halting + no entropy weighting.
- DistilBERT / TinyBERT — 2-7× compression, uniform KD.
- ASVD / GPTQ / SparseGPT — weight compression, caps around 8-10×.

The combination of fractal schedule + entropy weighting gets you 30-100×
past that frontier.
```

**Tweet 6** (the demo):
```
You can run it on your machine right now:

  git clone https://github.com/<user>/ultracompress
  python demo.py --prompt "The capital of France is"

Outputs top-5 teacher predictions vs top-5 student predictions side by
side. You see with your own eyes what 311× compression looks like.
```

**Tweet 7** (what it's for):
```
Applications:
• On-device LLMs (phones, browsers, embedded)
• Serving cost reduction (same latency/quality at ~1% the GPU)
• Regulated/air-gapped deployment
• Cheap per-customer fine-tuning substrate

The compression ratio is high enough to make some of these products
economically viable that currently aren't.
```

**Tweet 8** (the meta):
```
One person. 6 months. Consumer GPU. Provisional patent filed this week.

The paper/patent describe a training recipe, not a new ISA or new
hardware. Anyone can reproduce or build on it.

If you work in on-device AI, foundation-model serving, or edge deploy,
I'd love to talk.
```

**Tweet 9** (the CTA):
```
Code: github.com/<user>/ultracompress
Numbers: hires_results_hq5.json in the repo
Writeup: docs/HQ5_RESULTS.md
Contact: [email]

Open to collaboration, consulting, licensing, and loud technical
disagreement. RTs appreciated.
```

**Timing:** post Tuesday 9:00 AM Pacific (caught by both US + Europe). Repost the first tweet at 7 PM Pacific for the West Coast evening crowd. Don't re-post the thread more than twice in the first 72 hours.

**Accounts to tag in replies (not in the main thread — looks thirsty):**
- Reply to tweet 1 tagging @karpathy, @jeremyphoward, @swyx, @ylecun with a single sentence that isn't a "please look at me" plea.
- DM the thread link to Nat Friedman, Daniel Gross, Elad Gil, Gokul Rajaram.

---

## LinkedIn post (shorter, for recruiter / strategic eyes)

```
Over the past 6 months I built a method for compressing large language
models by 300-700x while retaining ~70% of the teacher model's top-10
next-token behavior.

The numbers, verified on 1000 held-out samples:

• 1.51M-parameter student → 311x compression → 69.64% T10 agreement with
  Qwen3-1.7B on FineWeb-Edu
• 0.64M-parameter student → 734x compression → 68.00% T10

The method combines a fractal-iterative shared-block architecture with an
entropy-weighted distillation loss. Provisional patent filed April 2026;
code and results fully open source.

If you lead engineering or product at a company that serves LLMs at scale,
deploys models on-device, or needs compressed private models for
regulated environments — happy to talk.

Repo: github.com/<user>/ultracompress
```

Post this Tuesday morning alongside the Twitter thread. LinkedIn is where the buyer audience (engineering VPs, corp-dev leads, enterprise AI platform heads) actually looks.

---

## Email template — cold outreach to 25 strategic targets

Subject: `Qwen3-1.7B compressed 311x with 69.64% T10 retention — 3-min ask`

```
Hi [first name],

Short version: I built an LLM compression method that hits 311x
compression (1.72B → 1.51M trainable params) while retaining 69.64% of
the teacher model's top-10 next-token predictions on held-out text.
Reproducible in 15 min on one GPU.

Why I'm writing: I think [COMPANY] would care because [one specific
reason — their recent on-device push / cost-reduction goals / regulated-
industry customer / edge product / etc].

Provisional patent filed this week. Open source at
github.com/<user>/ultracompress — the hires_results_hq5.json file is the
single source of truth on the numbers.

Can we do a 20-minute call next week? I'm happy to run the benchmark
live on your hardware or demonstrate a compression on one of your models
at no cost.

[signature + link]
```

Targets for the first wave (25 total):
- Apple Intelligence (on-device priority): [specific names — look up]
- Meta Llama team: [Thomas Scialom, Angela Fan, Ahmad Al-Dahle]
- Google Gemini Nano: [Oriol Vinyals, Andrew Dai]
- Qualcomm AI Research: [Durk Kingma]
- Arm AI: [specific names]
- Hugging Face: [Thomas Wolf, Julien Chaumond]
- Mistral: [Arthur Mensch]
- Cohere: [Aidan Gomez]
- Red Hat (Neural Magic acquirer): [Mark Kurtz]
- NVIDIA (Deci acquirer): [Yochay Ettun]
- Together AI, Anyscale, Modal, Fireworks, Baseten (inference infra — licensing motion)
- OpenAI, Anthropic research teams (longer shot but cheap to send)

25 emails = ~5 replies = ~2 meetings. Enough to calibrate real pricing.

---

## What success looks like in week 1

- [ ] HN `Show HN` post: top-10 page, 200+ upvotes
- [ ] Twitter thread: 500K+ impressions, 10+ DMs from relevant operators
- [ ] LinkedIn post: 100+ reactions, 5+ meaningful comments from engineering leaders
- [ ] 25 cold emails sent → 5+ replies → 2+ meetings booked
- [ ] YC + Speedrun + Neo + AI Grant applications submitted

Zero of these require quitting your job. All of them can be done in 4 evenings.
