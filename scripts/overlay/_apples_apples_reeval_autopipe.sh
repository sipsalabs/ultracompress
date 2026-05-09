#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8

# Wait for Yi-9B compression to finish (look for "[e2e] DONE." in log)
LOG=scripts/overlay/_e2e_yi_1_5_9b.log
echo "[reeval] waiting for Yi-1.5-9B compression to finish..."
while ! grep -q "\[e2e\] DONE\." "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[reeval] Yi-9B compression DONE. Firing apples-to-apples re-eval (uniform 5bpw at n_eval=50)..."

# Re-run PPL eval on the ORIGINAL Qwen3-1.7B-Base uniform 5bpw e2e dir at n_eval=50
# (matching v1's n_eval=50 for honest direct comparison)
python scripts/overlay/eval_compressed_only.py \
    --model qwen3-1.7b-base \
    --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base \
    --device cuda:1 \
    --seq_len 1024 \
    --n_eval 50 \
    > scripts/overlay/_eval_qwen3_1_7b_base_uniform_n50.log 2>&1

cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
   docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json 2>/dev/null

echo "[reeval] DONE — apples-to-apples uniform 5bpw at n_eval=50:"
grep -E "ppl_ratio|baseline_ppl|compressed_ppl" docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json 2>/dev/null
echo "[reeval] Compare to v1 (same n_eval=50): docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json"
