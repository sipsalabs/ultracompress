# Contributing to UltraCompress

UltraCompress is active research exploring the limits of model compression. Contributions welcome!

## High-Impact Areas

### 1. Scaling Validation (Most Needed)
Run FRR on models we haven't tested yet:
```bash
python compress_frr.py --model meta-llama/Llama-3.1-8B --steps 15000
python compress_frr.py --model Qwen/Qwen3-4B --steps 15000
```
Report your results as a GitHub Issue with: model name, steps, T10 agreement, GPU used, training time.

### 2. Benchmark Results
Run standard evals on compressed models:
```bash
pip install lm-eval
python run_standard_eval.py
```
We need WikiText-2 perplexity and HellaSwag accuracy numbers.

### 3. Quality Improvements
Test new training techniques:
- `run_controller_test.py` — Input-dependent modulation (Ouroboros V2 style)
- `run_mol_test.py` — Mixture of LoRA experts
- `run_born_again.py` — Self-distillation (2-3 generations)
- `run_optimized_train.py` — Better training hyperparameters

### 4. Speculative Decoding
Build and benchmark the speculative decoding pipeline:
- `ultracompress/speculative.py` has the core engine
- Need: wall-clock speedup measurements vs baseline inference
- Need: acceptance rate at different temperatures

### 5. llama.cpp Integration
Fork llama.cpp and add FRR as a new architecture (~500-2000 lines C++).
The GGUF file would store one block; inference loops over it.

## How to Contribute

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Run your experiment
4. Open a PR with results

## Code Style

- Python scripts in root directory: `run_*.py` for experiments, `*.py` for tools
- Library modules in `ultracompress/`
- Use `import lib.unbuffered` at top of scripts (fixes Windows buffering)
- Include timing and eval metrics in all experiment scripts

## What We Need Most

**Compute.** FRR quality improves with more training steps (63% at 50K, still climbing at 100K). If you have GPUs and can run extended training on 4B/8B/70B models, that's the most valuable contribution.

## License

Apache 2.0. All contributions are licensed under the same terms.
