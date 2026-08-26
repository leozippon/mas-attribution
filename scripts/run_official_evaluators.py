#!/usr/bin/env python3
"""Run supported official benchmark evaluators on exported prediction files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.evaluation.official_task_eval import evaluate_official_answer_task  # noqa: E402
from mas_contribution_bench.utils.io import ensure_dir, iter_jsonl, write_jsonl  # noqa: E402

SWE_DATASET_NAMES = {
    "swebench_lite": "princeton-nlp/SWE-bench_Lite",
    "swebench_verified": "princeton-nlp/SWE-bench_Verified",
}

LOCAL_OFFICIAL_DATASETS = {
    "aime_2026",
    "gpqa_diamond",
    "hle",
    "arc_agi_2",
    "ifbench",
}

TASK_FILES = {
    "aime_2026": "data/processed/tasks/aime_2026_tasks.jsonl",
    "gpqa_diamond": "data/processed/tasks/gpqa_diamond_tasks.jsonl",
    "hle": "data/processed/tasks/hle_tasks.jsonl",
    "arc_agi_2": "data/processed/tasks/arc_agi_2_tasks.jsonl",
    "ifbench": "data/processed/tasks/ifbench_tasks.jsonl",
}


def run_command(cmd: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"command": cmd, "dry_run": True}
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }


def run_swebench(dataset: str, predictions_path: Path, run_id: str, max_workers: int, cache_level: str, dry_run: bool) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"dataset": dataset, "status": "missing_predictions", "path": str(predictions_path)}
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        SWE_DATASET_NAMES[dataset],
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
    ]
    result = run_command(cmd, dry_run=dry_run)
    result.update({"dataset": dataset, "official_evaluator": "swebench.harness.run_evaluation"})
    return result


def load_task_map(dataset: str) -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / TASK_FILES[dataset]
    tasks = {}
    for row in iter_jsonl(path):
        tasks[str(row.get("task_id"))] = row
    return tasks


def run_local_official(dataset: str, predictions_path: Path, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"dataset": dataset, "status": "missing_predictions", "path": str(predictions_path)}
    output_path = output_dir / f"{dataset}_official_scores.jsonl"
    if dry_run:
        return {
            "dataset": dataset,
            "status": "dry_run",
            "prediction_file": str(predictions_path),
            "output_file": str(output_path),
            "official_evaluator": "mas_contribution_bench.evaluation.official_task_eval",
        }
    tasks = load_task_map(dataset)
    rows = []
    skipped = {"missing_task": 0, "unsupported": 0}
    for pred in iter_jsonl(predictions_path):
        task_id = str(pred.get("task_id") or "")
        task = tasks.get(task_id)
        if task is None:
            skipped["missing_task"] += 1
            continue
        prediction = pred.get("prediction")
        result = evaluate_official_answer_task(task, prediction)
        if result is None:
            skipped["unsupported"] += 1
            continue
        score, passed, failure_type, raw = result
        rows.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "run_id": pred.get("run_id"),
                "architecture_id": pred.get("architecture_id"),
                "model_name": pred.get("model_name"),
                "score": score,
                "passed": passed,
                "failure_type": getattr(failure_type, "value", str(failure_type)),
                "metric": (task.get("evaluation") or {}).get("metric", "official_score"),
                "raw_evaluator_output": raw,
            }
        )
    write_jsonl(output_path, rows)
    mean_score = sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0
    return {
        "dataset": dataset,
        "status": "ok",
        "official_evaluator": "mas_contribution_bench.evaluation.official_task_eval",
        "prediction_file": str(predictions_path),
        "output_file": str(output_path),
        "records": len(rows),
        "mean_score": mean_score,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official evaluators for exported predictions.")
    parser.add_argument("--prediction-dir", default="data/results/official_predictions")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        choices=[
            "aime_2026",
            "gpqa_diamond",
            "hle",
            "arc_agi_2",
            "ifbench",
            "swebench_lite",
            "swebench_verified",
            "livecodebench",
            "teambench",
            "tau2_bench",
            "marble",
            "multiagentbench",
        ],
    )
    parser.add_argument("--run-id", default="mas_qwen_official")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--cache-level", default="env", choices=["none", "base", "env", "instance"])
    parser.add_argument("--output-dir", default="data/results/official_scores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pred_dir = PROJECT_ROOT / args.prediction_dir
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    outputs = []
    for dataset in args.dataset:
        pred_path = pred_dir / f"{dataset}_predictions.jsonl"
        if dataset in SWE_DATASET_NAMES:
            outputs.append(run_swebench(dataset, pred_path, args.run_id, args.max_workers, args.cache_level, args.dry_run))
        elif dataset in LOCAL_OFFICIAL_DATASETS:
            outputs.append(run_local_official(dataset, pred_path, output_dir, args.dry_run))
        else:
            outputs.append(
                {
                    "dataset": dataset,
                    "status": "not_integrated",
                    "prediction_file": str(pred_path),
                    "reason": "This benchmark needs its official package/environment loop wired separately. Prediction export is available, but official scoring is not safely callable from this script yet.",
                }
            )
    print(json.dumps({"outputs": outputs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
