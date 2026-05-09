#!/bin/bash
# Auto-generates docs/MORNING_BRIEFING_2026_05_09.md once Hermes-405B is fully
# packed + uploaded. Aggregates every overnight result into one human-readable
# doc so Sip wakes up to a single thing to read instead of 8 log files.
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8

LOG=scripts/overlay/_hf_upload_hermes_3_405b_watchdog_watchdog.log
echo "[morning-briefing] waiting for Hermes-405B HF upload to land..."
while ! grep -qE "SUCCESS|FAILED" "$LOG" 2>/dev/null; do
    sleep 300
done
echo "[morning-briefing] Hermes outcome captured. Building briefing..."

OUT=docs/MORNING_BRIEFING_2026_05_09.md
{
  echo "# Morning Briefing — 2026-05-09"
  echo ""
  echo "Auto-generated when Hermes-405B HF upload landed. Single doc, no log spelunking."
  echo ""
  echo "## Compute outcomes overnight"
  echo ""
  echo "### Hermes-3-Llama-3.1-405B"
  echo ""
  if grep -q "SUCCESS" "$LOG" 2>/dev/null; then
    echo "- HF upload: SUCCESS"
    echo "- Public URL: https://huggingface.co/SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5"
  else
    echo "- HF upload: FAILED at attempt $(tail -1 "$LOG" 2>/dev/null)"
    echo "- Action: re-fire watchdog manually with bash scripts/overlay/_hf_upload_watchdog.sh _packed_hermes_3_405b_v3 SipsaLabs/hermes-3-llama-3.1-405b-uc-v3-bpw5 _hf_upload_hermes_3_405b_watchdog_retry"
  fi
  echo ""
  echo "### V18-C train_steps adaptive v2 experiment (Qwen3-1.7B-Base)"
  echo ""
  if [ -f docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json ]; then
    PPL=$(grep -oE '"ppl_ratio":[ ]*[0-9.]+' docs/PPL_EVAL_qwen3-1.7b-base-v2-adaptive-train-steps_2026_05_08.json | head -1)
    echo "- Result: $PPL"
    echo "- Compare to: uniform 5bpw (1.004876), per-Linear v1 (1.005097)"
    echo "- Decision criterion: <1.0049 => v2 cure validated, file CIP claim 13"
  else
    echo "- Result file missing — check scripts/overlay/_eval_qwen3_1_7b_base_v2_train_steps.log"
  fi
  echo ""
  echo "### Phi-2 21st arch retry (with cleared .pyc)"
  if [ -f docs/PPL_EVAL_phi-2_2026_05_08.json ]; then
    PPL=$(grep -oE '"ppl_ratio":[ ]*[0-9.]+' docs/PPL_EVAL_phi-2_2026_05_08.json | head -1)
    echo "- Result: $PPL"
  else
    echo "- Did not land. Check scripts/overlay/_e2e_phi_2_retry_v2.log"
  fi
  echo ""
  echo "### 22nd arch (Gemma-2-9B or StableLM-2-12B)"
  if [ -d scripts/overlay/_e2e_gemma_2_9b ]; then
    N=$(ls scripts/overlay/_e2e_gemma_2_9b/layer_*.pt 2>/dev/null | wc -l)
    echo "- Gemma-2-9B compression: $N/42 layers"
  elif [ -d scripts/overlay/_e2e_stablelm_2_12b ]; then
    N=$(ls scripts/overlay/_e2e_stablelm_2_12b/layer_*.pt 2>/dev/null | wc -l)
    echo "- StableLM-2-12B (fallback) compression: $N layers"
  else
    echo "- 22nd arch did not start. Check _arch22_gemma2_9b_autopipe.log"
  fi
  echo ""
  echo "## Today's must-do (Sip-only, in order)"
  echo ""
  echo "1. **Read the v2 result above.** If <1.0049, this is a real cure — patentable separately."
  echo "2. **Tomorrow morning launch posts** at 8:30 AM PT. Drafts ready:"
  echo "   - X thread: docs/LAUNCH_THREAD_HERMES_405B_2026_05_09.md"
  echo "   - LinkedIn: docs/LAUNCH_LINKEDIN_HERMES_405B_2026_05_09.md"
  echo "   - Press: docs/PRESS_RELEASE_HERMES_405B_2026_05_09.md"
  echo "3. **Patent filing** (\$390 weekend): docs/PATENT_FILING_PACKET_2026_05_09.md"
  echo "4. **5 cold emails offering \$5K Phase 0 POC** — drafts at docs/OUTREACH_2026_05_08/"
  echo "   Recipients: Lambda Labs, Together AI, CoreWeave, Replicate, Fireworks, Anyscale, Groq."
  echo "   COMPANY-VOICE ONLY. Sign as 'The Sipsa Labs team', sender founder@sipsalabs.com."
  echo "5. **Atlas EIN check** — day 4 of 1-7 window. If landed, fire NASA SBIR Phase 1 + AFWERX Phase 1."
  echo ""
  echo "## Recent commits"
  echo ""
  git log --oneline -15 2>&1 | head -15
  echo ""
  echo "## Run dashboard for full state"
  echo ""
  echo '```'
  echo "bash scripts/overlay/sip_dashboard.sh"
  echo '```'
} > "$OUT"

echo "[morning-briefing] DONE. $OUT generated."
