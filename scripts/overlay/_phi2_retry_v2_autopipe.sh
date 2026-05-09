#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1  # prevent stale .pyc

LOG=scripts/overlay/_v2_adaptive_train_steps_autopipe.log
echo "[phi2-retry-v2] waiting for V18-C v2 train_steps experiment to finish..."
while ! grep -qE "DONE\.|FAILED|ppl_ratio" "$LOG" 2>/dev/null; do
    sleep 90
done
echo "[phi2-retry-v2] V18-C v2 done. Re-firing Phi-2 with patched streaming_teacher (no .pyc)..."

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--microsoft--phi-2/snapshots/810d367871c1d460086d9f82db8696f2e0a0fcd0
rm -rf scripts/overlay/_e2e_phi_2

PYTHONDONTWRITEBYTECODE=1 python scripts/overlay/stream_compress_e2e.py \
    --hf-id microsoft/phi-2 \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_phi_2 \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_phi_2_retry_v2.log 2>&1

if grep -q "\[e2e\] DONE\." scripts/overlay/_e2e_phi_2_retry_v2.log 2>/dev/null; then
    echo "[phi2-retry-v2] Phi-2 compression DONE. Running PPL eval..."
    python scripts/overlay/eval_compressed_only.py \
        --model phi-2 --compressed_dir scripts/overlay/_e2e_phi_2 \
        --device cuda:1 --seq_len 1024 --n_eval 50 \
        > scripts/overlay/_eval_phi_2.log 2>&1
    cp scripts/overlay/artifacts/streaming_compression_phi-2_eval_only.json \
       docs/PPL_EVAL_phi-2_2026_05_08.json 2>/dev/null
    echo "[phi2-retry-v2] DONE. Result:"
    grep "ppl_ratio" docs/PPL_EVAL_phi-2_2026_05_08.json 2>/dev/null
else
    echo "[phi2-retry-v2] STILL FAILED — check _e2e_phi_2_retry_v2.log"
    tail -30 scripts/overlay/_e2e_phi_2_retry_v2.log 2>/dev/null
fi
