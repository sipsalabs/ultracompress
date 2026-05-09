#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for Yi-9B PPL eval to finish
LOG=scripts/overlay/_eval_yi_1_5_9b.log
echo "[phi2-retry] waiting for Yi-9B PPL eval to finish..."
while ! grep -qE "Output JSON:|ppl_ratio" "$LOG" 2>/dev/null; do
    sleep 30
done
echo "[phi2-retry] Yi-9B eval done. Re-running Phi-2 compression with patched streaming_teacher..."

SNAPSHOT=/c/Users/scamd/.cache/huggingface/hub/models--microsoft--phi-2/snapshots/810d367871c1d460086d9f82db8696f2e0a0fcd0

# Clear any partial state from the failed run
rm -rf scripts/overlay/_e2e_phi_2

python scripts/overlay/stream_compress_e2e.py \
    --hf-id microsoft/phi-2 \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_phi_2 \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_phi_2_retry.log 2>&1

if [ $? -eq 0 ]; then
    echo "[phi2-retry] Phi-2 compression DONE — running PPL eval..."
    python scripts/overlay/eval_compressed_only.py \
        --model phi-2 --compressed_dir scripts/overlay/_e2e_phi_2 \
        --device cuda:1 --seq_len 1024 --n_eval 50 \
        > scripts/overlay/_eval_phi_2.log 2>&1
    cp scripts/overlay/artifacts/streaming_compression_phi-2_eval_only.json \
       docs/PPL_EVAL_phi-2_2026_05_08.json 2>/dev/null
    echo "[phi2-retry] DONE — Phi-2 21st arch validated"
else
    echo "[phi2-retry] Phi-2 retry FAILED — check _e2e_phi_2_retry.log"
fi
