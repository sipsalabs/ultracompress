#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_XET=1

# Wait for pack to finish (look for "PACK COMPLETE" in log)
LOG=scripts/overlay/_pack_v3_yi_1_5_9b.log
echo "[yi-upload] waiting for Yi-9B pack to finish..."
while ! grep -q "PACK COMPLETE" "$LOG" 2>/dev/null; do
    sleep 30
done
echo "[yi-upload] Yi-9B pack DONE."

# Verify pack
uc verify _packed_yi_1_5_9b_v3 --skip-hash > scripts/overlay/_verify_yi_1_5_9b.log 2>&1
echo "[yi-upload] Pack verified. Firing HF upload via watchdog..."

nohup bash scripts/overlay/_hf_upload_watchdog.sh \
    _packed_yi_1_5_9b_v3 \
    SipsaLabs/yi-1.5-9b-uc-v3-bpw5 \
    _hf_upload_yi_1_5_9b_watchdog \
    > scripts/overlay/_hf_upload_yi_1_5_9b_watchdog.log 2>&1 &

echo "[yi-upload] DONE — Yi-9B upload firing under watchdog."
