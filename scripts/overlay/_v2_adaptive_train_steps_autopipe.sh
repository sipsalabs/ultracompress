#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UC_ADAPTIVE_TRAIN_STEPS=1

LOG=scripts/overlay/_phi2_retry_autopipe.log
echo "[v2-train-steps] waiting for Phi-2 retry to finish on cuda:1..."
while ! grep -qE "Phi-2 21st arch validated|Phi-2 retry FAILED" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[v2-train-steps] cuda:1 free. Firing V18-C adaptive train_steps v2 on Qwen3-1.7B-Base..."

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1

# Use a fresh output dir for clean A/B vs original
mkdir -p scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps/_teacher_hidden_cache
# Reuse existing teacher hidden cache (it's the same for both runs since same hf_id + calib)
if [ -f scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache/manifest.json ]; then
    echo "[v2-train-steps] symlinking teacher cache (saves ~10 min)"
    rm -rf scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps/_teacher_hidden_cache
    cp -r scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps/_teacher_hidden_cache
fi

UC_ADAPTIVE_TRAIN_STEPS=1 python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-1.7B-Base \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 --skip-cache \
    > scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps.log 2>&1

if [ $? -eq 0 ]; then
    echo "[v2-train-steps] compression DONE. Running PPL eval..."
    python scripts/overlay/eval_compressed_only.py \
        --model qwen3-1.7b-base \
        --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base_v2_train_steps \
        --device cuda:1 --seq_len 1024 --n_eval 50 \
        > scripts/overlay/_eval_qwen3_1_7b_base_v2_train_steps.log 2>&1
    cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
       docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json 2>/dev/null
    echo "[v2-train-steps] DONE. PPL ratio:"
    grep "ppl_ratio" docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json 2>/dev/null
    echo "[v2-train-steps] Compare against:"
    echo "  uniform 5bpw n_eval=50: 1.004876 (apples-to-apples)"
    echo "  per-Linear v1:          1.005097 (refuted)"
fi
