#!/usr/bin/env python3
"""Run supported official benchmark evaluators on exported prediction files."""

from __future__ import annotations

import argparse
import json
import os
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
    "swebench_lite": "lite",
    "swebench_verified": "verified",
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


def run_command(
    cmd: list[str],
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if dry_run:
        result = {"command": cmd, "dry_run": True}
        if cwd is not None:
            result["cwd"] = str(cwd)
        return result
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env, cwd=cwd)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "cwd": str(cwd) if cwd is not None else None,
    }


def run_swebench(
    dataset: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    timeout: int,
    report_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"dataset": dataset, "status": "missing_predictions", "path": str(predictions_path)}
    if shutil.which("swebench") is None:
        return {
            "dataset": dataset,
            "status": "missing_dependency",
            "prediction_file": str(predictions_path),
            "reason": "The `swebench` CLI is not available in the active environment.",
            "install_hint": "Install the official SWE-bench package in the evaluation environment.",
        }
    local_dataset = os.getenv("MAS_SWEBENCH_DATASET_PATH", "").strip()
    dataset_source = local_dataset or SWE_DATASET_NAMES[dataset]
    cmd = [
        "swebench",
        "eval",
        dataset_source,
        "--predictions",
        str(predictions_path),
        "--run-id",
        run_id,
        "--workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--report-dir",
        str(report_dir),
    ]
    result = run_command(cmd, dry_run=dry_run)
    result.update(
        {
            "dataset": dataset,
            "official_evaluator": "swebench",
            "prediction_file": str(predictions_path),
            "report_dir": str(report_dir),
            "dataset_source": dataset_source,
            "offline_dataset": bool(local_dataset),
        }
    )
    return result


def run_livecodebench(
    predictions_path: Path,
    output_dir: Path,
    release_version: str,
    timeout: int,
    num_process_evaluate: int,
    lcb_python: str,
    lcb_runner_root: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"dataset": "livecodebench", "status": "missing_predictions", "path": str(predictions_path)}
    env = None
    root = Path(lcb_runner_root).expanduser().resolve() if lcb_runner_root else None
    if root:
        env = dict(**os.environ)
        env["PYTHONPATH"] = str(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    probe = run_command(
        [
            lcb_python,
            "-c",
            "import lcb_runner.runner.custom_evaluator",
        ],
        dry_run=False,
        env=env,
        cwd=root,
    )
    if probe.get("returncode") != 0:
        return {
            "dataset": "livecodebench",
            "status": "missing_dependency",
            "prediction_file": str(predictions_path),
            "reason": "The official LiveCodeBench runner package `lcb_runner` could not be imported by the selected Python.",
            "import_stderr": probe.get("stderr", ""),
            "install_hint": "Create/install an official LiveCodeBench environment, then pass --lcb-python and optionally --lcb-runner-root.",
            "expected_command": [
                lcb_python,
                "-m",
                "lcb_runner.runner.custom_evaluator",
                "--custom_output_file",
                str(predictions_path),
                "--scenario",
                "codegeneration",
                "--release_version",
                release_version,
                "--timeout",
                str(timeout),
                "--num_process_evaluate",
                str(num_process_evaluate),
            ],
        }
    output_path = output_dir / "livecodebench_official_eval.json"
    cmd = [
        lcb_python,
        "-m",
        "lcb_runner.runner.custom_evaluator",
        "--custom_output_file",
        str(predictions_path),
        "--scenario",
        "codegeneration",
        "--release_version",
        release_version,
        "--timeout",
        str(timeout),
        "--num_process_evaluate",
        str(num_process_evaluate),
    ]
    result = run_command(cmd, dry_run=dry_run, env=env, cwd=root)
    result.update(
        {
            "dataset": "livecodebench",
            "official_evaluator": "lcb_runner.runner.custom_evaluator",
            "prediction_file": str(predictions_path),
            "output_file": str(output_path),
            "release_version": release_version,
            "scenario": "codegeneration",
            "lcb_python": lcb_python,
            "lcb_runner_root": lcb_runner_root,
        }
    )
    if not dry_run:
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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
    parser.add_argument("--timeout", type=int, default=1800, help="Per-instance timeout for official SWE-bench evaluation.")
    parser.add_argument("--lcb-release-version", default="release_latest")
    parser.add_argument("--lcb-timeout", type=int, default=6)
    parser.add_argument("--lcb-num-process-evaluate", type=int, default=12)
    parser.add_argument("--lcb-python", default=sys.executable, help="Python executable with official LiveCodeBench dependencies.")
    parser.add_argument(
        "--lcb-runner-root",
        default=None,
        help="Optional path to a cloned official LiveCodeBench repository to prepend to PYTHONPATH.",
    )
    parser.add_argument(
        "--swebench-report-dir",
        default="data/results/official_scores/swebench_reports",
        help="Directory where the official SWE-bench CLI writes reports.",
    )
    parser.add_argument(
        "--cache-level",
        default="env",
        choices=["none", "base", "env", "instance"],
        help="Deprecated compatibility option; the current swebench CLI does not use this flag.",
    )
    parser.add_argument("--output-dir", default="data/results/official_scores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pred_dir = PROJECT_ROOT / args.prediction_dir
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    swebench_report_dir = ensure_dir(PROJECT_ROOT / args.swebench_report_dir)
    outputs = []
    for dataset in args.dataset:
        if dataset in SWE_DATASET_NAMES:
            pred_path = pred_dir / f"{dataset}_predictions.jsonl"
            outputs.append(
                run_swebench(
                    dataset,
                    pred_path,
                    args.run_id,
                    args.max_workers,
                    args.timeout,
                    swebench_report_dir,
                    args.dry_run,
                )
            )
        elif dataset == "livecodebench":
            pred_path = pred_dir / "livecodebench_predictions.json"
            outputs.append(
                run_livecodebench(
                    pred_path,
                    output_dir,
                    args.lcb_release_version,
                    args.lcb_timeout,
                    args.lcb_num_process_evaluate,
                    args.lcb_python,
                    args.lcb_runner_root,
                    args.dry_run,
                )
            )
        elif dataset in LOCAL_OFFICIAL_DATASETS:
            pred_path = pred_dir / f"{dataset}_predictions.jsonl"
            outputs.append(run_local_official(dataset, pred_path, output_dir, args.dry_run))
        else:
            pred_path = pred_dir / f"{dataset}_predictions.jsonl"
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
