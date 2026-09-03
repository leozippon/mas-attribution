#!/usr/bin/env bash
# Run the classic SWE-bench Verified harness against Exp01 Qwen predictions.
# The harness is serialized by default so it does not contend heavily with the
# ongoing model experiment or rootless Docker daemon.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/mas_contribution}"
PRED_ROOT="$PROJECT_ROOT/data/results/official_predictions/exp01_qwen/swebench_verified"
DATASET="$PROJECT_ROOT/data/raw/swebench_verified/official_harness_instances_v2.json"
OUTPUT_ROOT="$PROJECT_ROOT/data/results/official_scores/exp01_qwen/swebench_verified"
LOG="$PROJECT_ROOT/logs/exp01_swebench_v2_official_$(date +%Y%m%d_%H%M%S).log"
MAX_WORKERS="${SWE_MAX_WORKERS:-1}"
TIMEOUT="${SWE_TIMEOUT:-1800}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate swebench-v2

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")"

for prediction_file in "$PRED_ROOT"/*/swebench_verified_predictions.jsonl; do
  [ -f "$prediction_file" ] || continue

  label="$(basename "$(dirname "$prediction_file")")"
  run_id="exp01_qwen_${label}_swebench_v2"
  report="$OUTPUT_ROOT/$label/swebench_official_report.json"
  normalized="$OUTPUT_ROOT/$label/predictions_normalized.jsonl"
  mkdir -p "$(dirname "$report")"

  if [ -f "$report" ]; then
    echo "[skip] $label: report already exists" | tee -a "$LOG"
    continue
  fi

  python "$PROJECT_ROOT/scripts/normalize_swebench_predictions.py" \
    "$prediction_file" "$normalized"

  echo "[start] $label predictions=$(wc -l < "$normalized")" | tee -a "$LOG"
  # Reuse prepared task images across all architecture-seed prediction sets.
  # This avoids rebuilding the same repositories and dependency environments.
  python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" \
    --predictions_path "$normalized" \
    --max_workers "$MAX_WORKERS" \
    --timeout "$TIMEOUT" \
    --cache_level instance \
    --clean false \
    --run_id "$run_id" 2>&1 | tee -a "$LOG"

  harness_report="$PROJECT_ROOT/qwen-local.${run_id}.json"
  if [ -f "$harness_report" ]; then
    cp "$harness_report" "$report"
    echo "[done] $label report=$report" | tee -a "$LOG"
  else
    echo "[warn] $label: harness report was not created" | tee -a "$LOG"
  fi
done

echo "[complete] SWE-bench v2 official post-processing finished" | tee -a "$LOG"
