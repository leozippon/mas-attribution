#!/usr/bin/env python3
"""Export MAS score JSONL rows to official benchmark prediction formats.

This script is intentionally conservative: it exports predictions only when the
required artifact can be extracted from `final_answer`. It does not turn fallback
scores into official scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.evaluation.official_extractors import (  # noqa: E402
    extract_python_or_text,
    extract_unified_diff,
    normalize_instance_id,
)
from mas_contribution_bench.utils.io import ensure_dir, iter_jsonl, write_jsonl  # noqa: E402


def load_tasks(task_dir: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(task_dir.glob("*_tasks.jsonl")):
        for row in iter_jsonl(path):
            tasks[str(row.get("task_id"))] = row
    return tasks


def latest_by_run(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep all architecture/seed outputs. Official eval may later aggregate by run.
    return list(rows)


def export_swebench(rows: list[dict[str, Any]], dataset: str, output_dir: Path, model_name: str) -> dict[str, Any]:
    exported = []
    skipped = Counter()
    for row in rows:
        patch = extract_unified_diff(row.get("final_answer"))
        if not patch:
            skipped["missing_patch"] += 1
            continue
        exported.append(
            {
                "instance_id": normalize_instance_id(str(row.get("task_id")), dataset),
                "model_name_or_path": model_name,
                "model_patch": patch,
                "run_id": row.get("run_id"),
                "architecture_id": row.get("architecture_id"),
                "score_file_dataset": dataset,
            }
        )
    path = output_dir / f"{dataset}_predictions.jsonl"
    write_jsonl(path, exported)
    return {"dataset": dataset, "path": str(path), "records": len(exported), "skipped": dict(skipped)}


def export_livecodebench(rows: list[dict[str, Any]], dataset: str, output_dir: Path, model_name: str) -> dict[str, Any]:
    exported = []
    skipped = Counter()
    for row in rows:
        code = extract_python_or_text(row.get("final_answer"))
        if not code:
            skipped["missing_code"] += 1
            continue
        question_id = normalize_instance_id(str(row.get("task_id")), dataset)
        exported.append(
            {
                "question_id": question_id,
                "code_list": [code],
                "model_name": model_name,
                "run_id": row.get("run_id"),
                "architecture_id": row.get("architecture_id"),
            }
        )
    path = output_dir / f"{dataset}_predictions.jsonl"
    write_jsonl(path, exported)
    return {"dataset": dataset, "path": str(path), "records": len(exported), "skipped": dict(skipped)}


def export_generic_answers(rows: list[dict[str, Any]], dataset: str, output_dir: Path, model_name: str) -> dict[str, Any]:
    exported = []
    skipped = Counter()
    for row in rows:
        answer = extract_python_or_text(row.get("final_answer"))
        if not answer:
            skipped["missing_answer"] += 1
            continue
        exported.append(
            {
                "task_id": row.get("task_id"),
                "prediction": answer,
                "model_name": model_name,
                "run_id": row.get("run_id"),
                "architecture_id": row.get("architecture_id"),
            }
        )
    path = output_dir / f"{dataset}_predictions.jsonl"
    write_jsonl(path, exported)
    return {"dataset": dataset, "path": str(path), "records": len(exported), "skipped": dict(skipped)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export official evaluator prediction files from MAS score JSONL.")
    parser.add_argument("--score-file", required=True, help="Evaluation score JSONL containing final_answer.")
    parser.add_argument("--output-dir", default="data/results/official_predictions", help="Directory for exported predictions.")
    parser.add_argument("--model-name", default="qwen-local")
    parser.add_argument("--dataset", action="append", default=None, help="Optional dataset filter; repeatable.")
    args = parser.parse_args()

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    selected = set(args.dataset or [])
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(PROJECT_ROOT / args.score_file):
        dataset = str(row.get("dataset"))
        if selected and dataset not in selected:
            continue
        rows_by_dataset[dataset].append(row)

    summaries = []
    for dataset, rows in sorted(rows_by_dataset.items()):
        rows = latest_by_run(rows)
        if dataset in {"swebench_lite", "swebench_verified"}:
            summaries.append(export_swebench(rows, dataset, output_dir, args.model_name))
        elif dataset == "livecodebench":
            summaries.append(export_livecodebench(rows, dataset, output_dir, args.model_name))
        else:
            summaries.append(export_generic_answers(rows, dataset, output_dir, args.model_name))

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps({"score_file": args.score_file, "outputs": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_path), "outputs": summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
