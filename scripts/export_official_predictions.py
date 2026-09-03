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
    # Keep the latest row for each task inside one exported model/configuration.
    # Official evaluators expect at most one prediction per task instance.
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[str(row.get("task_id"))] = row
    return list(deduped.values())


def load_run_index(run_file: Path | None) -> dict[str, dict[str, Any]]:
    if run_file is None or not run_file.exists():
        return {}
    return {str(row.get("run_id")): row for row in iter_jsonl(run_file) if row.get("run_id")}


def config_output_dir(base_dir: Path, dataset: str, architecture_id: str, seed: Any) -> Path:
    safe_arch = str(architecture_id or "unknown_architecture").replace("/", "_")
    safe_seed = str(seed if seed is not None else "unknown_seed").replace("/", "_")
    return ensure_dir(base_dir / dataset / f"{safe_arch}__seed_{safe_seed}")


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
    """Export LiveCodeBench custom-evaluator predictions.

    LiveCodeBench's official custom evaluator expects one JSON array file whose
    entries contain a question_id and a list of candidate programs. Keeping this
    format separate from JSONL avoids silently feeding the official runner a file
    it cannot parse.
    """
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
    path = output_dir / f"{dataset}_predictions.json"
    path.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset": dataset,
        "path": str(path),
        "format": "livecodebench_custom_json",
        "records": len(exported),
        "skipped": dict(skipped),
    }


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
    parser.add_argument("--architecture-id", action="append", default=None, help="Optional architecture filter; repeatable.")
    parser.add_argument("--seed", action="append", type=int, default=None, help="Optional seed filter; repeatable. Requires --run-file when seed is absent from score rows.")
    parser.add_argument("--run-file", default=None, help="Optional run JSONL used to join seed/config metadata by run_id.")
    parser.add_argument("--split-by-config", action="store_true", help="Write one official prediction file per dataset/architecture/seed.")
    args = parser.parse_args()

    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    selected = set(args.dataset or [])
    selected_architectures = set(args.architecture_id or [])
    selected_seeds = set(args.seed or [])
    run_file = PROJECT_ROOT / args.run_file if args.run_file else None
    run_index = load_run_index(run_file)
    rows_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(PROJECT_ROOT / args.score_file):
        row = dict(row)
        dataset = str(row.get("dataset"))
        if selected and dataset not in selected:
            continue
        architecture_id = str(row.get("architecture_id") or "")
        if selected_architectures and architecture_id not in selected_architectures:
            continue
        run_meta = run_index.get(str(row.get("run_id")), {})
        seed = row.get("seed", run_meta.get("seed"))
        row["_seed"] = seed
        if selected_seeds and seed not in selected_seeds:
            continue
        if args.split_by_config:
            key = (dataset, architecture_id, seed)
        else:
            key = (dataset,)
        rows_by_group[key].append(row)

    summaries = []
    for key, rows in sorted(rows_by_group.items(), key=lambda item: tuple(str(part) for part in item[0])):
        dataset = str(key[0])
        architecture_id = str(key[1]) if len(key) > 1 else None
        seed = key[2] if len(key) > 2 else None
        group_output_dir = config_output_dir(output_dir, dataset, architecture_id, seed) if args.split_by_config else output_dir
        rows = latest_by_run(rows)
        if dataset in {"swebench_lite", "swebench_verified"}:
            summary = export_swebench(rows, dataset, group_output_dir, args.model_name)
        elif dataset == "livecodebench":
            summary = export_livecodebench(rows, dataset, group_output_dir, args.model_name)
        else:
            summary = export_generic_answers(rows, dataset, group_output_dir, args.model_name)
        if args.split_by_config:
            summary["architecture_id"] = architecture_id
            summary["seed"] = seed
        summaries.append(summary)

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps({"score_file": args.score_file, "outputs": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_path), "outputs": summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
