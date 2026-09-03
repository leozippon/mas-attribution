#!/usr/bin/env bash
# Run the official LiveCodeBench evaluator for every Exp01 architecture/seed.
# The local snapshot removes the evaluator's dependency on a remote HF fetch;
# scoring and test execution remain those of the official LCB runner.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREDICTION_ROOT=${PREDICTION_ROOT:-"$PROJECT_ROOT/data/results/official_predictions/exp01_qwen/livecodebench"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_ROOT/data/results/official_scores/exp01_qwen/livecodebench"}
LCB_PYTHON=${LCB_PYTHON:-"$HOME/miniconda3/envs/vllm-qwen/bin/python"}
LCB_RUNNER_ROOT=${LCB_RUNNER_ROOT:-"$PROJECT_ROOT/external/LiveCodeBench"}
LCB_LOCAL_PROCESSED_TASKS=${LCB_LOCAL_PROCESSED_TASKS:-"$PROJECT_ROOT/data/processed/tasks/livecodebench_tasks.jsonl"}
LCB_TIMEOUT=${LCB_TIMEOUT:-6}
LCB_WORKERS=${LCB_WORKERS:-8}

export LCB_LOCAL_PROCESSED_TASKS
mkdir -p "$RESULT_ROOT" "$PROJECT_ROOT/logs"

for prediction_dir in "$PREDICTION_ROOT"/*; do
  [ -d "$prediction_dir" ] || continue
  config_name=$(basename "$prediction_dir")
  output_dir="$RESULT_ROOT/$config_name"
  report="$output_dir/livecodebench_official_eval.json"
  if [ -f "$report" ]; then
    echo "[skip] $config_name has $report"
    continue
  fi

  mkdir -p "$output_dir"
  log_file="$PROJECT_ROOT/logs/exp01_lcb_${config_name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[start] $config_name"
  python "$PROJECT_ROOT/scripts/run_official_evaluators.py" \
    --prediction-dir "$prediction_dir" \
    --dataset livecodebench \
    --lcb-python "$LCB_PYTHON" \
    --lcb-runner-root "$LCB_RUNNER_ROOT" \
    --lcb-release-version release_latest \
    --lcb-timeout "$LCB_TIMEOUT" \
    --lcb-num-process-evaluate "$LCB_WORKERS" \
    --output-dir "$output_dir" 2>&1 | tee "$log_file"
  echo "[done] $config_name"
done
