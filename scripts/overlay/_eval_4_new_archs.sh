#!/bin/bash
cd /c/Users/scamd/ultracompress
export PYTHONIOENCODING=utf-8

declare -A MODELS=(
  ["smollm2-1.7b"]="scripts/overlay/_e2e_smollm2_1_7b"
  ["tinyllama-1.1b-chat"]="scripts/overlay/_e2e_tinyllama_1_1b_chat"
  ["qwen3-0.6b"]="scripts/overlay/_e2e_qwen3_0_6b"
  ["olmo-2-0425-1b"]="scripts/overlay/_e2e_olmo_2_0425_1b"
)

for model in "${!MODELS[@]}"; do
    dir="${MODELS[$model]}"
    out="docs/PPL_EVAL_${model//[.\/]/_}_2026_05_08.json"
    log="scripts/overlay/_eval_full_${model//[.\/]/_}.log"
    echo "[$(date +%H:%M:%S)] Eval $model from $dir -> $out"
    python scripts/overlay/eval_compressed_only.py \
        --model "$model" \
        --compressed_dir "$dir" \
        --n_eval 30 --seq_len 1024 \
        --device cuda:1 \
        --out_json "$out" \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] $model eval done"
done

echo "All 4 evals done."
