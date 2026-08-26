"""Optional export of task-level metadata from processed TaskRecord JSONL files.

This script reads `data/processed/tasks/*.jsonl` and writes a lightweight task
index to `data/processed/metadata/task_metadata.jsonl`.

It is an optional export only. It is not the canonical sampling path and Exp08
does not depend on its output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.data.loaders import (  # noqa: E402
    DATASET_TASK_FILES,
    ensure_unique_task_ids,
    load_jsonl_tasks,
    summarize_tasks,
)
from mas_contribution_bench.data.schemas import DatasetName, TaskRecord  # noqa: E402


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _text_len(value: str | None) -> int:
    return len(value or "")


def _task_metadata_row(task: TaskRecord) -> dict[str, Any]:
    source = task.source
    evaluation = task.evaluation
    mas = task.mas_metadata
    difficulty = task.difficulty
    contribution = task.contribution_metadata

    prompt_length = _text_len(task.prompt)
    context_length = _text_len(task.context)
    tests_length = _text_len(task.tests)
    reference_solution_length = _text_len(task.reference_solution)

    return {
        "task_id": task.task_id,
        "dataset": _enum_value(task.dataset),
        "split": task.split,
        "task_type": _enum_value(task.task_type),
        "prompt_length": prompt_length,
        "context_length": context_length,
        "tests_length": tests_length,
        "reference_solution_length": reference_solution_length,
        "total_text_length": prompt_length + context_length + tests_length,
        "has_context": task.context is not None,
        "has_tests": task.tests is not None,
        "has_reference_solution": task.reference_solution is not None,
        "input_format": task.input_format,
        "output_format": task.output_format,
        "entry_point": task.entry_point,
        "evaluator_type": _enum_value(evaluation.evaluator_type),
        "metric": evaluation.metric,
        "timeout_seconds": evaluation.timeout_seconds,
        "sandbox": evaluation.sandbox,
        "evaluator_script_path": evaluation.script_path,
        "difficulty_source": difficulty.source,
        "difficulty_level": difficulty.level,
        "single_agent_score": difficulty.single_agent_score,
        "input_length": difficulty.input_length,
        "constraint_count": difficulty.constraint_count,
        "requires_cross_file_edit": difficulty.requires_cross_file_edit,
        "requires_planning": mas.requires_planning,
        "requires_coding": mas.requires_coding,
        "requires_verification": mas.requires_verification,
        "requires_research": mas.requires_research,
        "requires_tool_use": mas.requires_tool_use,
        "estimated_agents": mas.estimated_agents,
        "eligible_roles": contribution.eligible_roles,
        "default_architectures": contribution.default_architectures,
        "permission_requirements": contribution.permission_requirements,
        "intervention_tags": contribution.intervention_tags,
        "source_raw_dataset": source.raw_dataset,
        "source_raw_task_id": source.raw_task_id,
        "source_raw_file_path": source.raw_file_path,
        "source_license": source.license,
        "conversion_version": source.conversion_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/processed/metadata/task_metadata.jsonl from processed tasks."
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "tasks",
        help="Directory containing processed task JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "metadata" / "task_metadata.jsonl",
        help="Output metadata JSONL path.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_TASK_FILES),
        help="Dataset to include. Can be passed multiple times. Defaults to all available task files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first malformed task row.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = args.dataset or sorted(DATASET_TASK_FILES)

    all_tasks: list[TaskRecord] = []
    input_files: list[str] = []
    skipped: dict[str, str] = {}

    for dataset in datasets:
        path = args.tasks_dir / DATASET_TASK_FILES[dataset]
        if not path.exists():
            skipped[dataset] = f"missing task file: {path}"
            continue

        tasks = load_jsonl_tasks(path, strict=args.strict)
        assert isinstance(tasks, list)
        all_tasks.extend(tasks)
        input_files.append(str(path))

    ensure_unique_task_ids(all_tasks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(_task_metadata_row(task), ensure_ascii=False) + "\n")

    summary = {
        "status": "ok",
        "records": len(all_tasks),
        "output": str(args.output),
        "input_files": input_files,
        "skipped": skipped,
        "task_summary": summarize_tasks(all_tasks),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
