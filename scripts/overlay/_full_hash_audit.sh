#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8
echo "[hash-audit] Starting SHA256 audit of all 17 local v3 packs at $(date +%H:%M:%S)"
for pack in _packed_*_v3; do
    if [ -d "$pack" ]; then
        echo ""
        echo "=== $pack ==="
        uc verify "$pack" --compute-hashes 2>&1 | grep -E "VERIFY|sha256|SHA256|layer_" | head -10
    fi
done
echo ""
echo "[hash-audit] DONE at $(date +%H:%M:%S)"
