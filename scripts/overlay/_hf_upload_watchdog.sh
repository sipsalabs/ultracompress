#!/bin/bash
# Auto-retry HF uploads up to 8 times each. Used for residential-bandwidth flakiness
# (SSL EOF / multipart S3 transient errors).
#
# Usage: _hf_upload_watchdog.sh <local_dir> <repo_id> <log_prefix>
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_XET=1

LOCAL=$1
REPO=$2
PREFIX=$3
MAX_RETRIES=8

for attempt in $(seq 1 $MAX_RETRIES); do
    LOG="scripts/overlay/${PREFIX}_attempt${attempt}.log"
    echo "[$(date +%H:%M:%S)] Attempt $attempt/$MAX_RETRIES for $REPO" >> "scripts/overlay/${PREFIX}_watchdog.log"
    python scripts/overlay/_hf_upload_simple.py "$LOCAL" "$REPO" > "$LOG" 2>&1
    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] SUCCESS for $REPO on attempt $attempt" >> "scripts/overlay/${PREFIX}_watchdog.log"
        break
    fi
    echo "[$(date +%H:%M:%S)] FAILED attempt $attempt for $REPO, retrying in 30s..." >> "scripts/overlay/${PREFIX}_watchdog.log"
    sleep 30
done
