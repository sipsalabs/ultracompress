#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8

# Wait for v1 compression to finish (look for "[e2e] DONE." in log)
LOG=scripts/overlay/_e2e_qwen3_1_7b_base_adaptive.log
echo "[autopipe] waiting for v1 compression to finish..."
while ! grep -q "\[e2e\] DONE\." "$LOG" 2>/dev/null; do
    sleep 30
done
echo "[autopipe] v1 compression DONE. Firing PPL eval @ seq_len=1024..."

# Fire PPL eval at seq_len=1024 for honest comparison
python scripts/overlay/eval_compressed_only.py \
    --model qwen3-1.7b-base \
    --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base_adaptive \
    --device cuda:1 \
    --seq_len 1024 \
    > scripts/overlay/_eval_qwen3_1_7b_base_adaptive_v1.log 2>&1

echo "[autopipe] PPL eval done. Result:"
grep -E "ppl_ratio|baseline_ppl|compressed_ppl" scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json 2>/dev/null | tail -5

# Copy result to docs/
cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
   docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json 2>/dev/null

echo "[autopipe] DONE — result at docs/PPL_EVAL_qwen3-1.7b-base-adaptive-bpw-v1_2026_05_08.json"
