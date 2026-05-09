#!/bin/bash
# V4-A: AWQ-style channel rescaling experiment
# Fires after V4-D Multi-Pass eval done on cuda:1 (queued behind V4-D)
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UC_AWQ_SCALING=1
export UC_AWQ_ALPHA=0.5

LOG=docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json
echo "[v4a] waiting for V4-D Multi-Pass result to land..."
while [ ! -f "$LOG" ]; do
    sleep 60
done
echo "[v4a] V4-D done. Firing V4-A AWQ-style scaling on Qwen3-1.7B-Base..."

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
mkdir -p scripts/overlay/_e2e_qwen3_1_7b_base_v4a_awq/_teacher_hidden_cache
if [ -f scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache/manifest.json ]; then
    cp -r scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache scripts/overlay/_e2e_qwen3_1_7b_base_v4a_awq/
fi

UC_AWQ_SCALING=1 UC_AWQ_ALPHA=0.5 python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-1.7B-Base --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_qwen3_1_7b_base_v4a_awq \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 --skip-cache > scripts/overlay/_e2e_qwen3_1_7b_base_v4a_awq.log 2>&1

if [ $? -eq 0 ]; then
    python scripts/overlay/eval_compressed_only.py \
        --model qwen3-1.7b-base --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base_v4a_awq \
        --device cuda:1 --seq_len 1024 --n_eval 50 \
        > scripts/overlay/_eval_qwen3_1_7b_base_v4a_awq.log 2>&1
    cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
       docs/PPL_EVAL_qwen3-1.7b-base-v4a-awq_2026_05_09.json 2>/dev/null
    echo "[v4a] DONE. Result:"
    grep "ppl_ratio" docs/PPL_EVAL_qwen3-1.7b-base-v4a-awq_2026_05_09.json
fi
