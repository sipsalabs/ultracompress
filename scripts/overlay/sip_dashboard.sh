#!/bin/bash
# sip_dashboard.sh — at-a-glance status of all in-flight UltraCompress work
# Usage: ./scripts/overlay/sip_dashboard.sh
# Designed to answer "what's happening right now" in 5 seconds.

cd /c/Users/scamd/ultracompress

clear
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║              ULTRACOMPRESS — SIP DASHBOARD                             ║"
echo "║              $(date '+%Y-%m-%d %H:%M:%S')                                          ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "═══ HERMES-3-405B COMPRESSION (cuda:0, headline launch tomorrow AM) ═══"
HERMES_LAYERS=$(ls scripts/overlay/_e2e_hermes_3_405b_v3/layer_*.pt 2>/dev/null | wc -l)
HERMES_PCT=$((HERMES_LAYERS * 100 / 126))
HERMES_LAST=$(grep -E "^\[e2e\] layer.*done" scripts/overlay/_recompress_hermes_3_405b_v3_resume4.log 2>/dev/null | tail -1 | grep -oE 'layer_idx=[0-9.]+' | head -1)
echo "  Layers done: $HERMES_LAYERS / 126  (${HERMES_PCT}%)"
echo "  Last completed: $HERMES_LAST"
if grep -q "\[e2e\] DONE\." scripts/overlay/_recompress_hermes_3_405b_v3_resume4.log 2>/dev/null; then
    echo "  Status: ✓ ALL DONE — pack + upload autopipe will fire next"
else
    REMAINING=$((126 - HERMES_LAYERS))
    EST_MIN=$((REMAINING * 7))
    echo "  Status: ▶ RUNNING — ~$REMAINING layers remaining (~${EST_MIN} min ETA)"
fi

echo ""
echo "═══ Yi-1.5-9B 20TH ARCH COMPRESSION (cuda:1) ═══"
YI_LAYERS=$(ls scripts/overlay/_e2e_yi_1_5_9b/layer_*.pt 2>/dev/null | wc -l)
echo "  Layers done: $YI_LAYERS / 48"
if grep -q "\[e2e\] DONE\." scripts/overlay/_e2e_yi_1_5_9b.log 2>/dev/null; then
    echo "  Status: ✓ DONE — apples-to-apples re-eval autopipe firing next"
elif [ "$YI_LAYERS" -gt 0 ]; then
    YI_REMAINING=$((48 - YI_LAYERS))
    echo "  Status: ▶ RUNNING — $YI_REMAINING layers remaining"
else
    echo "  Status: ◌ WAITING (downloading or tokenizing)"
fi

echo ""
echo "═══ APPLES-TO-APPLES RE-EVAL (per-Linear v1 vs uniform 5bpw) ═══"
if grep -q "\[reeval\] DONE" scripts/overlay/_apples_apples_reeval_autopipe.log 2>/dev/null; then
    echo "  Status: ✓ DONE — see docs/PPL_EVAL_qwen3-1.7b-base-uniform-n50_2026_05_08.json"
else
    echo "  Status: ◌ QUEUED — fires after Yi-9B compression"
fi

echo ""
echo "═══ PHI-2 21ST ARCH (queued after re-eval) ═══"
if grep -q "\[arch21\] Phi-2 compression DONE" scripts/overlay/_arch21_phi2_autopipe.log 2>/dev/null; then
    echo "  Status: ✓ DONE"
else
    echo "  Status: ◌ QUEUED — fires after re-eval done"
fi

echo ""
echo "═══ HF UPLOADS (5 morning + 2 watchdog) ═══"
for arch in llama_3_1_8b qwen3_8b qwen3_14b mixtral_8x7b phi_3_5_moe; do
    log="scripts/overlay/_hf_upload_simple_${arch}.log"
    if [ -f "$log" ]; then
        last_layer=$(tail -3 "$log" 2>/dev/null | grep -oE "layer_[0-9]{3}" | tail -1)
        last_size=$(tail -3 "$log" 2>/dev/null | grep -oE "[0-9.]+(MB|GB)" | tail -1)
        echo "  $arch: $last_layer ($last_size)"
    fi
done
for prefix in llama_3_1_70b_watchdog phi3_mini_watchdog hermes_3_405b_watchdog; do
    log="scripts/overlay/_hf_upload_${prefix}_watchdog.log"
    if [ -f "$log" ]; then
        last=$(tail -1 "$log" 2>/dev/null)
        echo "  ${prefix}: $last"
    fi
done

echo ""
echo "═══ DISK ═══"
df -BG . 2>/dev/null | tail -1 | awk '{printf "  Used %s of %s (%s available, %s)\n", $3, $2, $4, $5}'

echo ""
echo "═══ GPU UTILIZATION ═══"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | head -2 | while read line; do
    echo "  $line"
done

echo ""
echo "═══ RECENT COMMITS (today) ═══"
git log --oneline -10 2>&1 | head -10 | sed 's/^/  /'

echo ""
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo "  Refresh: ./scripts/overlay/sip_dashboard.sh"
echo "  Tomorrow's plan: docs/TOMORROW_MORNING_AT_A_GLANCE_2026_05_09.md"
echo "  Patent filing packet: docs/PATENT_FILING_PACKET_2026_05_09.md"
