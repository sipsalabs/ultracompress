#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_XET=1

LOG=scripts/overlay/_recompress_hermes_3_405b_v3_resume4.log
echo "[hermes-post] waiting for Hermes-405B compression to finish..."
while ! grep -q "\[e2e\] DONE\." "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[hermes-post] Hermes-405B compression DONE. Packing to v3..."

# Pack to v3 (will be v2 legacy format since stream_compress_e2e doesn't persist
# GSQ codecs, but still works for inference)
nohup uc pack scripts/overlay/_e2e_hermes_3_405b_v3 _packed_hermes_3_405b_v3 \
    > scripts/overlay/_pack_v3_hermes_3_405b.log 2>&1
echo "[hermes-post] Pack DONE."

# Verify pack integrity
uc verify _packed_hermes_3_405b_v3 --skip-hash > scripts/overlay/_verify_hermes_3_405b.log 2>&1
echo "[hermes-post] Verify DONE."

# Fire HF upload via watchdog (handles SSL EOF, no email cascade)
echo "[hermes-post] Firing HF upload via watchdog..."
nohup bash scripts/overlay/_hf_upload_watchdog.sh \
    _packed_hermes_3_405b_v3 \
    SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 \
    _hf_upload_hermes_3_405b_watchdog \
    > scripts/overlay/_hf_upload_hermes_3_405b_watchdog.log 2>&1 &

echo "[hermes-post] DONE — Hermes-405B packed + verified + upload firing."
echo "[hermes-post] Watch progress: tail -f scripts/overlay/_hf_upload_hermes_3_405b_watchdog_watchdog.log"
