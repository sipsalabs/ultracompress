#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for OLMo-Instruct PPL JSON to land
while [ ! -f docs/PPL_EVAL_olmo-2-0425-1b-instruct_2026_05_08.json ]; do
    sleep 10
done
echo "OLMo-Instruct PPL done; firing SmolLM2-Instruct compression."

# Copy SmolLM2 calib cache for instruct slug
cp fineweb_edu_10M_tokens_smollm2_1_7b.pt fineweb_edu_10M_tokens_smollm2_1_7b_instruct.pt

SNAPSHOT=$(ls -d /c/Users/scamd/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/*/ | head -1)

python scripts/overlay/stream_compress_e2e.py \
    --hf-id HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_smollm2_1_7b_instruct \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_smollm2_1_7b_instruct.log 2>&1
echo "SmolLM2-Instruct done."
