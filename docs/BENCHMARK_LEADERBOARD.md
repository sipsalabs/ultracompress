# UltraCompress Open Compression Leaderboard -- Design Document

**Status:** Draft v1 -- April 2026
**Goal:** Become THE standard public benchmark for LLM compression methods. First-mover advantage means our compression techniques get maximum visibility and our metrics become the industry standard.

---

## 1. Strategic Context

### Why This Matters

There is no single, authoritative, community-adopted leaderboard for LLM compression. The closest thing is Intel's Low-Bit Quantized Open LLM Leaderboard on HuggingFace, but it only covers quantization (not pruning, distillation, factorization, or hybrid methods) and is Intel-branded, which limits community ownership.

The HuggingFace Open LLM Leaderboard proved that whoever defines the benchmark defines the narrative. MMLU, HellaSwag, ARC became the standard metrics *because the leaderboard tracked them*. We do the same for compression.

### Competitive Landscape (Existing Benchmarks to Study and Differentiate From)

| Existing Resource | What It Covers | Gap We Fill |
|---|---|---|
| **Intel Low-Bit Quantized LLM Leaderboard** (HF Space) | Quantization only (AutoRound, GPTQ, AWQ, BnB, GGUF). 10 benchmarks. Intel-centric. | We cover ALL compression methods. Vendor-neutral. Track compression ratio + inference speed, not just accuracy. |
| **LLMC Toolkit** (ModelTC/llmc, EMNLP 2024) | Benchmarking toolkit for quantization. ~600 experiments. | Toolkit, not a live leaderboard. No community submissions. No pruning/distillation/hybrid. |
| **LLM-KICK** (Jaiswal et al., 2023) | Evaluation framework beyond perplexity for compressed LLMs. | Paper, not a live resource. We operationalize their insight that perplexity alone is insufficient. |
| **"When Reasoning Meets Compression"** (Zhang et al., April 2025) | Benchmarked compressed reasoning models on complex tasks. | One-off study. We make it continuous and community-driven. |
| **"Benchmarking PTQ in LLMs"** (Zhao et al., Feb 2025) | Comprehensive taxonomy of PTQ methods with unified eval. | Paper-only. No live leaderboard. |

**Our differentiator:** Method-agnostic (quantization + pruning + distillation + factorization + hybrid), continuously updated, community-submitted, tracking both quality AND efficiency metrics, with standardized reproducible evaluation.

---

## 2. Metrics to Track

### Primary Metrics (Displayed in Main Table)

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Compression Ratio** | Size reduction (original / compressed) | `original_bytes / compressed_bytes` -- includes all artifacts (codebooks, indices, adapters, etc.) |
| **Bits Per Weight (BPW)** | Effective precision after compression | `(compressed_bytes * 8) / num_parameters` |
| **Quality Score** | Composite accuracy across eval tasks (see Section 4) | Weighted average of benchmark scores, normalized to FP16 baseline |
| **Quality Retention (%)** | How much of the original model's capability is preserved | `compressed_score / baseline_score * 100` |
| **Inference Throughput (tok/s)** | Generation speed on reference hardware | Measured on standardized hardware configs (see Section 6) |
| **Peak Memory (GB)** | Runtime memory footprint | Measured during inference with batch_size=1, seq_len=2048 |
| **Time-to-Compress (minutes)** | How long the compression pipeline takes | Wall-clock time on reference hardware |

### Derived / Secondary Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Quality-Compression Pareto Score** | Combined ranking on quality vs. compression frontier | Identifies methods that are strictly dominated by others |
| **Efficiency Score** | `quality_retention * throughput / baseline_throughput` | Single number: "how good AND how fast" |
| **Memory Efficiency** | `quality_retention / peak_memory_gb` | Quality per GB of memory used |
| **Compression Speed** | `original_gb / time_to_compress_minutes` | How fast can you compress |

### What Counts Toward "Compressed Size"

This is where cheating happens. Strict rules:

- Total compressed size = model weights + codebooks + index tables + lookup tables + any auxiliary data needed for inference (LoRA adapters, scale factors, zero points, etc.)
- Huffman/entropy coding: the decoder tables count toward size
- Distillation: only the student model size counts, but must declare teacher
- External knowledge (tokenizer, config files): excluded (same across all entries)

---

## 3. Benchmark Target Models

Models chosen to span size ranges, architectures, and popularity. All must be freely downloadable from HuggingFace.

### Tier 1: Required (must submit results on ALL of these)

| Model | HuggingFace ID | Parameters | Why |
|---|---|---|---|
| **Llama 3.1 8B** | `meta-llama/Llama-3.1-8B` | 8B | Industry standard, most-compressed model in the wild |
| **Qwen 2.5 7B** | `Qwen/Qwen2.5-7B` | 7B | Strong non-Llama architecture, popular in Asia |
| **Mistral 7B v0.3** | `mistralai/Mistral-7B-v0.3` | 7B | Different architecture choices (sliding window attention) |
| **Phi-3 Mini 3.8B** | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Small model, tests compression at lower parameter counts |

### Tier 2: Optional (bonus points, broader picture)

| Model | HuggingFace ID | Parameters | Why |
|---|---|---|---|
| **Llama 3.1 70B** | `meta-llama/Llama-3.1-70B` | 70B | Tests scaling -- does compression method work on large models? |
| **Qwen 2.5 72B** | `Qwen/Qwen2.5-72B` | 72B | Second large model, different architecture |
| **Gemma 2 9B** | `google/gemma-2-9b` | 9B | Google architecture, tests generalization |
| **DeepSeek-R1-Distill-Llama-8B** | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | Reasoning model -- tests if compression kills chain-of-thought |

### Tier 3: Edge / Extreme (for bragging rights)

| Model | HuggingFace ID | Parameters | Why |
|---|---|---|---|
| **SmolLM2 1.7B** | `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | Can you compress a tiny model even further? |
| **Llama 3.1 405B** | `meta-llama/Llama-3.1-405B` | 405B | Ultimate stress test |

### Baseline Entries (Auto-Generated)

For each target model, we auto-generate baseline entries:
- FP16 (uncompressed baseline)
- GPTQ INT4 (standard quantization)
- AWQ INT4 (activation-aware quantization)
- GGUF Q4_K_M (llama.cpp standard)
- llm-compressor / SparseGPT 50% sparsity + INT4

These give every visitor an immediate reference point and make the leaderboard useful even before external submissions arrive.

---

## 4. Evaluation Tasks

### Design Principle

Perplexity alone is insufficient for evaluating compressed models (confirmed by LLM-KICK paper). Compression can preserve perplexity while destroying specific capabilities. We need a multi-faceted evaluation.

### Core Eval Suite (Required)

All evaluations run through **EleutherAI lm-evaluation-harness** for reproducibility.

| Task | What It Tests | Shots | Metric | Weight in Composite |
|---|---|---|---|---|
| **WikiText-2 Perplexity** | Language modeling quality | 0 | PPL (lower = better) | 15% |
| **MMLU** | Broad knowledge across 57 subjects | 5 | Accuracy | 20% |
| **HellaSwag** | Commonsense reasoning | 10 | Accuracy (norm) | 15% |
| **ARC-Challenge** | Science reasoning | 25 | Accuracy (norm) | 15% |
| **TruthfulQA (MC2)** | Factual accuracy / hallucination resistance | 0 | MC2 score | 10% |
| **GSM8K** | Math reasoning (chain of thought) | 5 | Accuracy (strict) | 15% |
| **WinoGrande** | Coreference resolution | 5 | Accuracy | 10% |

**Why these specific tasks:**
- **WikiText-2 PPL**: Still the universal quick-check for compression quality. Required for backwards compatibility with existing literature.
- **MMLU**: Tests if compression destroys stored knowledge. Critical -- quantization often kills rare knowledge first.
- **HellaSwag + ARC**: Reasoning capabilities. If compression method preserves surface statistics but damages reasoning, these catch it.
- **TruthfulQA**: Compressed models can become more prone to hallucination. Must track this.
- **GSM8K**: Math reasoning is fragile under compression. A sensitive canary metric.
- **WinoGrande**: Nuanced language understanding. Catches subtle quality degradation.

### Extended Eval Suite (Optional, for deeper analysis)

| Task | What It Tests | Why Include |
|---|---|---|
| **IFEval** | Instruction following | Critical for chat/instruct models |
| **GPQA Diamond** | PhD-level science | Tests extreme knowledge retention |
| **BBH (BIG-Bench Hard)** | Hard reasoning tasks | Stress-test for reasoning |
| **HumanEval / MBPP** | Code generation | Tests code capability preservation |
| **MT-Bench** | Multi-turn conversation quality | For chat model compression |

### Quality Score Formula

```
quality_score = (
    0.15 * normalize(wikitext2_ppl, lower_is_better=True) +
    0.20 * mmlu_acc +
    0.15 * hellaswag_acc_norm +
    0.15 * arc_challenge_acc_norm +
    0.10 * truthfulqa_mc2 +
    0.15 * gsm8k_acc +
    0.10 * winogrande_acc
)
```

For the composite ranking displayed on the leaderboard, we compute:

```
leaderboard_rank = quality_retention_pct * log2(compression_ratio)
```

This rewards BOTH quality and compression. A method that preserves 95% quality at 4x compression scores higher than one preserving 90% at 4x, but a method preserving 90% at 16x scores even higher.

---

## 5. Submission System

### Platform: HuggingFace Spaces (Gradio)

**Why HuggingFace Spaces:**
- Built-in community, discoverability, and trust
- Gradio has a native Leaderboard component (`gradio-leaderboard`)
- Free hosting for the frontend
- OAuth integration for user identity
- Datasets hub for storing results
- The Open LLM Leaderboard already proved this architecture works at scale

### Architecture

```
[Submitter] --> [Leaderboard Space (Gradio UI)]
                        |
                        v
              [Submissions Dataset (HF Hub)]
                        |
                        v
              [Evaluator Space (Private, GPU)]
                  - Downloads compressed model
                  - Runs lm-evaluation-harness
                  - Measures throughput + memory
                  - Writes results to Results Dataset
                        |
                        v
              [Results Dataset (HF Hub)]
                        |
                        v
              [Leaderboard Space reads & displays]
```

### Submission Requirements

Each submission must include:

1. **Compressed model** uploaded to HuggingFace Hub (public repo)
2. **Submission metadata** (YAML):
   ```yaml
   method_name: "UltraCompress-Genome-v2"
   method_type: "hybrid"  # quantization | pruning | distillation | factorization | hybrid | other
   base_model: "meta-llama/Llama-3.1-8B"
   compressed_model_id: "ultracompress/Llama-3.1-8B-genome-v2"
   bits_per_weight: 2.1
   compression_ratio: 7.6
   requires_gpu: true
   inference_framework: "transformers"  # transformers | vllm | llama.cpp | custom
   compression_time_minutes: 45
   hardware_used: "1x A100 80GB"
   paper_url: ""  # optional
   code_url: "https://github.com/ultracompress/ultracompress"
   description: "Genome-based compression with MoE codebooks"
   ```
3. **Inference script** (standardized template we provide, or custom with justification)
4. **Compression script** (for reproducibility verification)

### Submission Flow

1. User fills out form on Leaderboard Space
2. System validates: model exists on Hub, metadata is complete, required target models are present
3. Submission enters queue
4. Evaluator Space pulls model, runs full eval suite on standardized hardware
5. Results appear on leaderboard within 24-48 hours
6. User is notified via HuggingFace notification

---

## 6. Anti-Cheating and Reproducibility

### The Cheating Problem

Compression benchmarks are uniquely vulnerable to gaming:
- **Size inflation:** Claiming smaller size by not counting codebooks/indices
- **Eval overfitting:** Tuning compression to specific benchmark prompts
- **Cherry-picking:** Only reporting results on easy models/tasks
- **Unreproducible methods:** "Our method gets X" but no one can verify

### Countermeasures

#### 1. Standardized Size Measurement
```python
def measure_compressed_size(model_repo_id: str) -> int:
    """Official size = total bytes of ALL files in the HF repo
    excluding: config.json, tokenizer*, special_tokens_map*,
    README.md, .gitattributes
    Including: *.safetensors, *.bin, *.pt, *.gguf, *.codebook,
    *.index, *.scales, *.zeros, ANY other file needed for inference
    """
    total = 0
    for file in list_repo_files(model_repo_id):
        if not is_excluded(file):
            total += get_file_size(model_repo_id, file)
    return total
```

We measure size ourselves from the uploaded repo. Submitters cannot self-report size.

#### 2. Server-Side Evaluation Only
- All eval runs happen on OUR hardware (or HF donated compute)
- Submitters upload the model; we run the eval
- No self-reported benchmark scores accepted
- This is exactly how the Open LLM Leaderboard works and it is proven

#### 3. Reproducibility Requirements
- Compression code must be public (GitHub link required)
- We randomly select 10% of submissions for full reproducibility audit
- Community can flag suspicious results for review
- Reproduce from scratch: download base model -> run compression script -> verify output matches submitted model (within tolerance)

#### 4. Eval Contamination Detection
- We use the same lm-evaluation-harness version and prompts as the Open LLM Leaderboard (well-studied, hard to game)
- Periodically rotate a small "canary" eval set that is not publicly disclosed
- Monitor for suspiciously high scores on specific tasks relative to base model performance

#### 5. Hardware Standardization for Speed Benchmarks
- **Tier A (Primary):** 1x NVIDIA A100 80GB, PyTorch 2.x, CUDA 12.x
- **Tier B (Consumer):** 1x NVIDIA RTX 4090 24GB
- **Tier C (Edge):** 1x Apple M2 Pro, Metal (via llama.cpp)
- Throughput numbers always labeled with hardware tier
- We run the inference benchmark, not the submitter

#### 6. Mandatory Baseline Comparison
- Every submission is displayed alongside the same-bit-width GPTQ/AWQ/GGUF baseline
- Makes it immediately obvious if a method is actually better or just marketing

---

## 7. Technical Implementation Plan

### Phase 1: MVP (Weeks 1-3)

**Goal:** Live leaderboard with auto-generated baselines. No external submissions yet.

1. Create HuggingFace organization: `ultracompress-benchmark`
2. Build Gradio Space with `gradio-leaderboard` component
3. Generate baseline results for Tier 1 models:
   - FP16, GPTQ-INT4, AWQ-INT4, GGUF-Q4_K_M
   - Run lm-evaluation-harness on all core tasks
   - Measure throughput on A100
4. Add UltraCompress results (our own methods)
5. Store results in HF Dataset: `ultracompress-benchmark/results`
6. Launch blog post + social media

**Tech stack:**
- Frontend: Gradio (Python) on HF Spaces
- Data: HuggingFace Datasets (parquet backend)
- Eval: EleutherAI lm-evaluation-harness v0.4+
- Infra: HF Spaces (free tier for frontend, paid GPU for eval)

### Phase 2: Open Submissions (Weeks 4-6)

1. Build submission form (Gradio tab)
2. Build private evaluator Space (GPU-enabled)
3. Implement queue system (HF Dataset as job queue)
4. Add email/HF notification on completion
5. Write submission guide and FAQ
6. Seed with invitations to 10-15 known compression researchers

### Phase 3: Community Growth (Months 2-3)

1. Add Tier 2 and Tier 3 models
2. Add extended eval suite
3. Add consumer hardware (RTX 4090) benchmarks
4. Build "Compare" feature (side-by-side method comparison)
5. Add historical tracking (method performance over time)
6. Publish quarterly analysis reports
7. Submit leaderboard as a workshop paper (NeurIPS / ICML)

### Phase 4: Ecosystem (Months 4-6)

1. API for programmatic result access
2. Badges system ("Certified X BPW on Llama-3.1-8B")
3. Integration with popular compression toolkits (LLMC, AutoGPTQ, llama.cpp)
4. Corporate sponsor tier (compute donations for faster eval)
5. Annual "Compression Challenge" with prizes

---

## 8. Leaderboard UI Design

### Main View

```
===========================================================
  ULTRACOMPRESS OPEN COMPRESSION LEADERBOARD
  The standard benchmark for LLM compression methods
===========================================================

[Filter: Model] [Filter: Method Type] [Filter: BPW Range] [Sort By: v]

 #  Method               Model           BPW   Ratio  Quality%  MMLU  HellaSwag  Throughput  Memory
--- -------------------- --------------- ----- ------ --------- ----- ---------- ----------- ------
 1  UltraCompress-G2     Llama-3.1-8B    2.1   7.6x    94.2%   63.1    78.2      245 tok/s   4.1GB
 2  AWQ-INT4             Llama-3.1-8B    4.0   4.0x    96.8%   65.2    80.1      741 tok/s   5.2GB
 3  GPTQ-INT4            Llama-3.1-8B    4.0   4.0x    95.1%   64.0    79.3      712 tok/s   5.2GB
 4  GGUF-Q4_K_M          Llama-3.1-8B    4.5   3.6x    96.1%   64.8    79.8      320 tok/s   5.5GB
 5  SparseGPT-50%+INT4   Llama-3.1-8B    2.0   8.0x    88.3%   58.2    72.1      410 tok/s   3.8GB
 ...

[Tabs: Llama-3.1-8B | Qwen-2.5-7B | Mistral-7B | Phi-3-Mini | All Models]
```

### Detail View (Click on Entry)

- Full benchmark breakdown (all 7+ tasks)
- Perplexity curve (by sequence position)
- Memory usage over time
- Link to model on HF Hub
- Link to compression code
- Reproducibility status badge
- Comparison chart vs. FP16 baseline

### Pareto Frontier Visualization

Interactive scatter plot: X-axis = compression ratio, Y-axis = quality retention. Pareto frontier highlighted. Every dot is clickable to see details.

---

## 9. Marketing and Launch Strategy

### Pre-Launch (2 weeks before)

- Seed the leaderboard with 20+ entries (baselines + our methods)
- Write blog post: "Why LLM Compression Needs a Standard Benchmark"
- Contact 5-10 compression paper authors, invite them to submit
- Prepare Twitter/X thread with key visualizations

### Launch

- Post on HuggingFace blog (request community blog feature)
- Show HN post
- Reddit: r/LocalLLaMA, r/MachineLearning
- Twitter/X thread with Pareto frontier chart
- Direct outreach to: TheBloke (prolific quantizer), Intel Neural Compressor team, MIT Han Lab (SparseGPT, AWQ creators), IST Austria (GPTQ creators), llama.cpp community

### Ongoing

- Weekly leaderboard digest (top new submissions)
- Monthly analysis post comparing method families
- Quarterly "State of LLM Compression" report
- Conference workshop proposal (NeurIPS 2026 or ICML 2027)

---

## 10. Naming Options

| Name | Pros | Cons |
|---|---|---|
| **Open Compression Leaderboard (OCL)** | Neutral, descriptive, mirrors "Open LLM Leaderboard" | Generic |
| **CompressBench** | Clear, memorable | Sounds like a one-off benchmark |
| **UltraBoard** | Tied to our brand | May discourage competing methods from submitting |
| **The Squeeze Leaderboard** | Fun, memorable | Possibly too informal |

**Recommendation:** Launch as **"Open Compression Leaderboard"** under the `ultracompress-benchmark` HuggingFace org. The neutral name maximizes adoption. Our brand gets visibility as the creator/maintainer without being the name itself.

---

## 11. Resource Requirements

| Item | Cost | Notes |
|---|---|---|
| HF Space (frontend) | Free | Gradio on HF free tier |
| HF Space (evaluator, GPU) | ~$200-500/month | A100 hours for running evals. Could seek HF compute grant. |
| A100 time for baselines | ~$100-200 one-time | Generate initial 20+ baseline entries |
| Engineering time | 2-3 weeks | Build frontend + evaluator + submission pipeline |
| Domain (optional) | $15/year | compressionleaderboard.com or similar, redirect to HF Space |

**Total MVP cost: Under $500 + 3 weeks of work.**

HuggingFace offers compute grants for community projects. The Open LLM Leaderboard itself runs on donated compute. We should apply.

---

## 12. Risk Analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| No one submits | Medium | Seed heavily with baselines + our methods. Invite known researchers personally. Make submission trivially easy. |
| Intel leaderboard expands to cover our scope | Low-Medium | Move fast. First-mover with broader scope. Their branding limits community ownership. |
| Compute costs spiral | Medium | Rate-limit submissions. Seek HF compute grant. Require submitters to run optional self-eval first. |
| Eval gaming / cheating | Medium | Server-side eval only. Random reproducibility audits. Community flagging. |
| Models get gated/removed | Low | Pin specific model revisions. Keep local copies of weights. |
| Benchmark saturation (everyone scores 95%+) | Long-term | Continuously add harder tasks. Move to Tier 2/3 models. Add edge-device constraints. |

---

## 13. Success Criteria

| Timeframe | Metric | Target |
|---|---|---|
| Month 1 | Submissions | 30+ entries (including baselines) |
| Month 1 | HF Space likes | 100+ |
| Month 3 | External submissions | 20+ from non-ultracompress teams |
| Month 3 | Citation in papers | 5+ papers reference the leaderboard |
| Month 6 | Community recognition | Referenced in compression toolkit READMEs |
| Year 1 | Standard status | "Evaluated on OCL" becomes standard in compression papers |

---

## Appendix A: Quick-Start for Internal Use

Before public launch, use this to generate our own baseline entries:

```bash
# Install eval harness
pip install lm-eval

# Run eval on a compressed model
lm_eval --model hf \
  --model_args pretrained=ultracompress/Llama-3.1-8B-genome-v2 \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,gsm8k,winogrande \
  --batch_size auto \
  --output_path results/

# Measure perplexity separately
lm_eval --model hf \
  --model_args pretrained=ultracompress/Llama-3.1-8B-genome-v2 \
  --tasks wikitext \
  --batch_size auto \
  --output_path results/
```

## Appendix B: Submission Metadata Schema (Full)

```json
{
  "method_name": "string (required)",
  "method_type": "enum: quantization|pruning|distillation|factorization|hybrid|other (required)",
  "method_subtype": "string (optional, e.g., 'GPTQ', 'SparseGPT', 'SVD+quant')",
  "base_model": "string, HF model ID (required)",
  "compressed_model_id": "string, HF model ID (required)",
  "bits_per_weight": "float (required)",
  "requires_gpu": "bool (required)",
  "requires_custom_kernel": "bool (required)",
  "inference_framework": "enum: transformers|vllm|llama.cpp|custom (required)",
  "compression_time_minutes": "float (required)",
  "compression_hardware": "string (required)",
  "calibration_dataset": "string (optional)",
  "calibration_samples": "int (optional)",
  "training_required": "bool (required)",
  "training_time_hours": "float (optional)",
  "paper_url": "string (optional)",
  "code_url": "string (required)",
  "description": "string, max 500 chars (required)",
  "contact_email": "string (optional)",
  "license": "string (required)"
}
```
