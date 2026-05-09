#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for v1 PPL eval to finish (look for autopipe DONE marker)
LOG=scripts/overlay/_per_linear_v1_autopipe.log
echo "[arch20] waiting for v1 PPL eval to finish..."
while ! grep -q "\[autopipe\] DONE" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[arch20] v1 eval DONE. Downloading 01-AI/Yi-1.5-9B..."

# 01-AI/Yi-1.5-9B: 9B params, llama-arch family (no special handling needed)
# Auto-download via from_pretrained (uses snapshot_download under the hood)
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='01-ai/Yi-1.5-9B', repo_type='model')
print(f'[arch20] downloaded to {p}')
" > scripts/overlay/_arch20_download.log 2>&1

# Tokenize Yi calibration
python scripts/data/tokenize_fineweb_for_model.py \
    --model 01-ai/Yi-1.5-9B \
    --n_tokens 10_000_000 \
    --output fineweb_edu_10M_tokens_yi_1_5_9b.pt \
    > scripts/overlay/_arch20_tokenize.log 2>&1

# Find snapshot path
SNAPSHOT_BASE=/c/Users/scamd/.cache/huggingface/hub/models--01-ai--Yi-1.5-9B/snapshots
SNAPSHOT=$(ls -d "$SNAPSHOT_BASE"/*/ | head -1)
echo "[arch20] snapshot: $SNAPSHOT"

# Fire compression on cuda:1
python scripts/overlay/stream_compress_e2e.py \
    --hf-id 01-ai/Yi-1.5-9B \
    --shard-dir "$SNAPSHOT" \
    --output scripts/overlay/_e2e_yi_1_5_9b \
    --bpw 5 --rank 32 --train-steps 200 --train-bs 8 --n-calib 64 --seq-len 1024 \
    --device cuda:1 \
    > scripts/overlay/_e2e_yi_1_5_9b.log 2>&1

echo "[arch20] Yi-1.5-9B compression DONE"
