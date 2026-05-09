#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for Phi-2 21st arch compression to finish
LOG=scripts/overlay/_arch21_phi2_autopipe.log
echo "[v18c-adaptive] waiting for Phi-2 compression to finish..."
while ! grep -q "\[arch21\] Phi-2 compression DONE" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[v18c-adaptive] Phi-2 done. Firing V18-C adaptive train_steps experiment on Qwen3-1.7B-Base..."

# Use shared teacher hidden cache from original run
mkdir -p scripts/overlay/_e2e_qwen3_1_7b_base_v18c_adaptive
cp -r scripts/overlay/_e2e_qwen3_1_7b_base/_teacher_hidden_cache \
      scripts/overlay/_e2e_qwen3_1_7b_base_v18c_adaptive/_teacher_hidden_cache 2>/dev/null

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1

UC_ADAPTIVE_TRAIN_STEPS=1 \
python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-1.7B-Base \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_qwen3_1_7b_base_v18c_adaptive \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 --skip-cache \
    > scripts/overlay/_e2e_qwen3_1_7b_base_v18c_adaptive.log 2>&1

echo "[v18c-adaptive] Compression DONE. Firing PPL eval at n_eval=50..."

python scripts/overlay/eval_compressed_only.py \
    --model qwen3-1.7b-base \
    --compressed_dir scripts/overlay/_e2e_qwen3_1_7b_base_v18c_adaptive \
    --device cuda:1 --seq_len 1024 --n_eval 50 \
    > scripts/overlay/_eval_qwen3_1_7b_base_v18c_adaptive.log 2>&1

cp scripts/overlay/artifacts/streaming_compression_qwen3-1.7b-base_eval_only.json \
   docs/PPL_EVAL_qwen3-1.7b-base-v18c-adaptive-train-steps_2026_05_08.json 2>/dev/null

echo "[v18c-adaptive] DONE — result:"
grep -E "ppl_ratio|baseline_ppl|compressed_ppl" docs/PPL_EVAL_qwen3-1.7b-base-v18c-adaptive-train-steps_2026_05_08.json 2>/dev/null
