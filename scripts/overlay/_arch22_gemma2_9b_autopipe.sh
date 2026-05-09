#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1

LOG=scripts/overlay/_phi2_retry_v2_autopipe.log
echo "[arch22] waiting for Phi-2 retry v2 to finish..."
while ! grep -qE "Result:|STILL FAILED" "$LOG" 2>/dev/null; do
    sleep 120
done
echo "[arch22] Downloading google/gemma-2-9b..."

python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='google/gemma-2-9b', repo_type='model')
print(f'[arch22] downloaded to {p}')
" > scripts/overlay/_arch22_download.log 2>&1

python scripts/data/tokenize_fineweb_for_model.py \
    --model google/gemma-2-9b --n_tokens 10_000_000 \
    --output fineweb_edu_10M_tokens_gemma_2_9b.pt \
    > scripts/overlay/_arch22_tokenize.log 2>&1

SNAPSHOT_BASE=/c/Users/scamd/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots
SNAPSHOT=$(ls -d "$SNAPSHOT_BASE"/*/ 2>/dev/null | head -1)
echo "[arch22] snapshot: $SNAPSHOT"

if [ -z "$SNAPSHOT" ]; then
    echo "[arch22] download failed (likely gated). Trying open alternative: stabilityai/stablelm-2-12b..."
    python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='stabilityai/stablelm-2-12b', repo_type='model')
print(f'[arch22-alt] downloaded to {p}')
" > scripts/overlay/_arch22_alt_download.log 2>&1
    SNAPSHOT_BASE=/c/Users/scamd/.cache/huggingface/hub/models--stabilityai--stablelm-2-12b/snapshots
    SNAPSHOT=$(ls -d "$SNAPSHOT_BASE"/*/ 2>/dev/null | head -1)
    HF_ID="stabilityai/stablelm-2-12b"
    OUT="scripts/overlay/_e2e_stablelm_2_12b"
else
    HF_ID="google/gemma-2-9b"
    OUT="scripts/overlay/_e2e_gemma_2_9b"
fi

if [ -n "$SNAPSHOT" ]; then
    PYTHONDONTWRITEBYTECODE=1 python scripts/overlay/stream_compress_e2e.py \
        --hf-id "$HF_ID" --shard-dir "$SNAPSHOT" --output "$OUT" \
        --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
        --device cuda:1 > "${OUT}.log" 2>&1
    echo "[arch22] $HF_ID compression done"
fi
