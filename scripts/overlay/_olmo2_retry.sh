#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for Qwen3-0.6B's last layer file (layer_027.pt for 28-layer model)
while [ ! -f scripts/overlay/_e2e_qwen3_0_6b/layer_027.pt ]; do
    sleep 10
done
echo "Qwen3-0.6B done, retrying OLMo-2..."

# Clean up partial OLMo-2 output from earlier failed run
rm -rf scripts/overlay/_e2e_olmo_2_0425_1b

python scripts/overlay/stream_compress_e2e.py \
    --hf-id allenai/OLMo-2-0425-1B \
    --shard-dir "C:/Users/scamd/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf" \
    --output scripts/overlay/_e2e_olmo_2_0425_1b \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_olmo_2_0425_1b_retry.log 2>&1

echo "OLMo-2 retry done."
