#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=scripts/overlay/_e2e_yi_1_5_9b.log
echo "[arch21] waiting for Yi-9B compression (and apples-to-apples re-eval) to finish..."
while ! grep -q "\[reeval\] DONE" scripts/overlay/_apples_apples_reeval_autopipe.log 2>/dev/null; do
    sleep 60
done
echo "[arch21] Yi-9B done + re-eval done. Downloading microsoft/phi-2..."

python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='microsoft/phi-2', repo_type='model')
print(f'[arch21] downloaded to {p}')
" > scripts/overlay/_arch21_download.log 2>&1

python scripts/data/tokenize_fineweb_for_model.py \
    --model microsoft/phi-2 --n_tokens 10_000_000 \
    --output fineweb_edu_10M_tokens_phi_2.pt \
    > scripts/overlay/_arch21_tokenize.log 2>&1

SNAPSHOT_BASE=/c/Users/scamd/.cache/huggingface/hub/models--microsoft--phi-2/snapshots
SNAPSHOT=$(ls -d "$SNAPSHOT_BASE"/*/ | head -1)
echo "[arch21] snapshot: $SNAPSHOT"

python scripts/overlay/stream_compress_e2e.py \
    --hf-id microsoft/phi-2 \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_phi_2 \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_phi_2.log 2>&1

echo "[arch21] Phi-2 compression DONE"
