#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for Phi-3-mini snapshot dir to populate
SNAPSHOT_BASE=/c/Users/scamd/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots
while [ ! -d "$SNAPSHOT_BASE" ] || [ -z "$(ls -A "$SNAPSHOT_BASE" 2>/dev/null)" ]; do
    sleep 15
done
SNAPSHOT=$(ls -d "$SNAPSHOT_BASE"/*/ | head -1)
echo "[phi3] snapshot ready: $SNAPSHOT"

# Wait for safetensors files to be present
while [ -z "$(ls "$SNAPSHOT"/*.safetensors 2>/dev/null)" ]; do
    sleep 15
done
echo "[phi3] safetensors present, waiting 30s for any in-flight files..."
sleep 30

# Tokenize FineWeb-edu for Phi-3-mini (CPU)
python scripts/data/tokenize_fineweb_for_model.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --n_tokens 10_000_000 \
    --output fineweb_edu_10M_tokens_phi_3_mini_4k_instruct.pt \
    > scripts/overlay/_tokenize_phi3_mini.log 2>&1
echo "[phi3] tokenization done"

# Fire compression on cuda:1
python scripts/overlay/stream_compress_e2e.py \
    --hf-id microsoft/Phi-3-mini-4k-instruct \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_phi_3_mini_4k_instruct \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_phi_3_mini_4k_instruct.log 2>&1
echo "[phi3] compression done"
