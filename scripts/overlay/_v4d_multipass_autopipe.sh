#!/bin/bash
# V4-D: Multi-Pass Cascade Correction experiment
# Fires after Hermes-405B PPL eval done on cuda:1
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UC_MULTI_PASS=2

LOG=scripts/overlay/_eval_hermes_3_405b_compressed.log
echo "[v4d] waiting for Hermes-405B PPL eval to finish..."
while ! grep -qE "Output JSON:|FAILED|ERROR|Traceback" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[v4d] Hermes eval done. Firing V4-D Multi-Pass on Qwen3-1.7B-Base..."

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1

mkdir -p scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass/_teacher_hidden_cache
if [ -f scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache/manifest.json ]; then
    cp -r scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass/
fi

UC_MULTI_PASS=2 python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-1.7B-Base --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 --skip-cache > scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass.log 2>&1

if [ $? -eq 0 ]; then
    python scripts/overlay/eval_compressed_only.py \
        --model qwen3-1.7b-base --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base_v4d_multipass \
        --device cuda:1 --seq_len 1024 --n_eval 50 \
        > scripts/overlay/_eval_qwen3_1_7b_base_v4d.log 2>&1
    cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
       docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json 2>/dev/null
    echo "[v4d] DONE. Result:"
    grep "ppl_ratio" docs/PPL_EVAL_qwen3-1.7b-base-v4d-multipass_2026_05_09.json
fi
