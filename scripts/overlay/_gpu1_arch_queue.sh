#!/bin/bash
# Sequential arch compression queue on GPU 1 (after SmolLM2 finishes).
# Runs OLMo-2 first, then Qwen3-0.6B, then TinyLlama if time permits.

cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for SmolLM2 to finish (presence of last-layer file)
while [ ! -f scripts/overlay/_e2e_smollm2_1_7b/layer_023.pt ]; do
    sleep 10
done
echo "SmolLM2 done, starting OLMo-2..."

# OLMo-2-0425-1B (12th arch, allenai)
python scripts/overlay/stream_compress_e2e.py \
    --hf-id allenai/OLMo-2-0425-1B \
    --shard-dir "C:/Users/scamd/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf" \
    --output scripts/overlay/_e2e_olmo_2_0425_1b \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_olmo_2_0425_1b.log 2>&1

echo "OLMo-2 done."

# TinyLlama-1.1B-Chat (13th arch — chat variant of Llama-1)
python scripts/overlay/stream_compress_e2e.py \
    --hf-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --shard-dir "C:/Users/scamd/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6" \
    --output scripts/overlay/_e2e_tinyllama_1_1b_chat \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_tinyllama_1_1b_chat.log 2>&1

echo "TinyLlama done."

# Qwen3-0.6B (14th arch — smallest Qwen3)
python scripts/overlay/stream_compress_e2e.py \
    --hf-id Qwen/Qwen3-0.6B \
    --shard-dir "$(ls -d /c/Users/scamd/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)" \
    --output scripts/overlay/_e2e_qwen3_0_6b \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_qwen3_0_6b.log 2>&1

echo "All GPU 1 arch queue done."
