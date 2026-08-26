"""Run task- and communication-conditioned descriptive summaries.

Exp08 is analysis-only by design. It derives selected-task metadata from the
configured normalized task files, then joins existing attribution records and
traces produced by Exp03-Exp07. Score files are not used as covariates. No LLM
calls are made. Communication summaries and correlations are descriptive and
non-causal.

Every configured dataset task_file and every path listed in
inputs.attribution_files, coalition_files, and trace_files must exist and be
non-empty. The run also fails if no metadata rows are selected or no
attribution rows load for those tasks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

from mas_contribution_bench.config import ExperimentSpec
from mas_contribution_bench.runners.common import load_experiment, select_tasks
from mas_contribution_bench.utils.io import ensure_dir, iter_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


REQUIRED_INPUT_FILE_KEYS = (
    "attribution_files",
    "coalition_files",
    "trace_files",
)

BANNED_TASK_CONDITION_AXES = frozenset(
    {
        "single_agent_success_bin",
        "full_mas_success_bin",
        "dominant_failure_type",
    }
)


def _resolve_paths(root: Path, values: Iterable[str | Path]) -> list[Path]:
    return [_as_path(root, value) for value in values]


def _dataset_task_paths(root: Path, raw: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for dataset in raw.get("datasets") or []:
        task_file = dataset.get("task_file") if isinstance(dataset, dict) else None
        if task_file:
            paths.append(_as_path(root, task_file))
    return paths


def _listed_input_paths(root: Path, raw: dict[str, Any]) -> list[tuple[str, Path]]:
    input_cfg = raw.get("inputs") or {}
    listed = [("task_file", path) for path in _dataset_task_paths(root, raw)]
    for key in REQUIRED_INPUT_FILE_KEYS:
        for value in input_cfg.get(key) or []:
            listed.append((key, _as_path(root, value)))
    return listed


def _preflight_required_inputs(root: Path, raw: dict[str, Any]) -> None:
    problems: list[str] = []
    for label, path in _listed_input_paths(root, raw):
        if not path.is_file():
            problems.append(f"{label} missing: {path}")
        elif path.stat().st_size == 0:
            problems.append(f"{label} empty: {path}")
    if problems:
        raise FileNotFoundError(
            "exp08 generalization requires every configured dataset task_file and every "
            "path listed in inputs.attribution_files, coalition_files, and "
            "trace_files to exist and be non-empty. "
            + "; ".join(problems)
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _bin_prompt_length(length: Any) -> str:
    value = int(_safe_float(length, 0))
    if value <= 500:
        return "short_<=500"
    if value <= 1000:
        return "medium_501_1000"
    if value <= 2000:
        return "long_1001_2000"
    return "very_long_>2000"


def _bin_count(value: Any, *, small: int, medium: int, label: str) -> str:
    n = int(_safe_float(value, 0))
    if n <= small:
        return f"{label}_low_<={small}"
    if n <= medium:
        return f"{label}_mid_{small + 1}_{medium}"
    return f"{label}_high_>{medium}"


def _task_to_metadata(task: dict[str, Any]) -> dict[str, Any]:
    difficulty = task.get("difficulty") or {}
    if not isinstance(difficulty, dict):
        difficulty = {}
    prompt = str(task.get("prompt") or "")
    prompt_length = len(prompt)
    return {
        "task_id": str(task.get("task_id") or ""),
        "dataset": str(task.get("dataset") or "unknown"),
        "split": str(task.get("split") or ""),
        "task_type": str(task.get("task_type") or "unknown"),
        "prompt_length": prompt_length,
        "prompt_length_bin": _bin_prompt_length(prompt_length),
        "difficulty_level": str(difficulty.get("level") or "unknown"),
        "input_length": difficulty.get("input_length") or prompt_length,
        "metadata_source": "configured_task_file",
    }


def _select_task_metadata(
    experiment: ExperimentSpec,
    max_tasks: int | None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    selected = select_tasks(experiment)
    if max_tasks is not None:
        selected = selected[:max_tasks]
    metadata_by_task: dict[str, dict[str, Any]] = {}
    for task in selected:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        metadata_by_task[task_id] = _task_to_metadata(task)
    return metadata_by_task, set(metadata_by_task)


def _load_trace_stats(
    root: Path,
    trace_files: list[str],
    selected_tasks: set[str],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "run_id": "",
            "task_id": "",
            "event_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_call_events": 0,
            "null_agent_events": 0,
            "roles": set(),
        }
    )
    for path in _resolve_paths(root, trace_files):
        for row in iter_jsonl(path):
            task_id = str(row.get("task_id"))
            if task_id not in selected_tasks:
                continue
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            item = stats[run_id]
            item["run_id"] = run_id
            item["task_id"] = task_id
            item["event_count"] += 1
            item["input_tokens"] += int(_safe_float(row.get("input_tokens")))
            item["output_tokens"] += int(_safe_float(row.get("output_tokens")))
            if row.get("tool_call"):
                item["tool_call_events"] += 1
            if (row.get("metadata") or {}).get("null_agent"):
                item["null_agent_events"] += 1
            role = row.get("role") or row.get("sender")
            if role:
                item["roles"].add(str(role))

    finalized: dict[str, dict[str, Any]] = {}
    for run_id, item in stats.items():
        roles = sorted(item.pop("roles"))
        item["role_count"] = len(roles)
        item["roles"] = roles
        item["total_tokens"] = item["input_tokens"] + item["output_tokens"]
        item["trace_length_bin"] = _bin_count(item["event_count"], small=4, medium=8, label="events")
        item["input_token_bin"] = _bin_count(item["input_tokens"], small=2000, medium=8000, label="input_tokens")
        item["output_token_bin"] = _bin_count(item["output_tokens"], small=2000, medium=8000, label="output_tokens")
        item["role_count_bin"] = _bin_count(item["role_count"], small=3, medium=6, label="roles")
        item["null_agent_bin"] = "has_null_agent" if item["null_agent_events"] else "no_null_agent"
        finalized[run_id] = item
    return finalized


def _load_coalition_run_map(
    root: Path,
    coalition_files: list[str],
    selected_tasks: set[str],
) -> dict[str, str]:
    """Map coalition identifiers to run ids for intervention attribution joins."""

    mapping: dict[str, str] = {}
    for path in _resolve_paths(root, coalition_files):
        for row in iter_jsonl(path):
            task_id = str(row.get("task_id"))
            if task_id not in selected_tasks:
                continue
            coalition_id = str(row.get("coalition_id") or "")
            run_id = str(row.get("run_id") or "")
            if coalition_id and run_id:
                mapping[coalition_id] = run_id
    return mapping


def _load_attribution_rows(
    root: Path,
    attribution_files: list[str],
    selected_tasks: set[str],
    metadata_by_task: dict[str, dict[str, Any]],
    trace_stats: dict[str, dict[str, Any]],
    coalition_run_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _resolve_paths(root, attribution_files):
        source_file = _display_path(root, path)
        for row in iter_jsonl(path):
            task_id = str(row.get("task_id"))
            if task_id not in selected_tasks:
                continue
            metadata = metadata_by_task.get(task_id, {})
            row_metadata = row.get("metadata") or {}
            full_run_id = row_metadata.get("full_run_id") or coalition_run_map.get(
                str(row_metadata.get("full_coalition_id") or "")
            )
            ablated_run_id = row_metadata.get("ablated_run_id") or coalition_run_map.get(
                str(row_metadata.get("ablated_coalition_id") or "")
            )
            communication = trace_stats.get(str(full_run_id)) or trace_stats.get(str(ablated_run_id)) or {}
            score = _safe_float(row.get("score"))
            rows.append(
                {
                    "task_id": task_id,
                    "dataset": str(row.get("dataset") or metadata.get("dataset") or "unknown"),
                    "task_type": str(metadata.get("task_type") or "unknown"),
                    "difficulty_level": str(metadata.get("difficulty_level") or "unknown"),
                    "prompt_length_bin": str(metadata.get("prompt_length_bin") or "unknown"),
                    "experiment_source": str(row.get("experiment_id") or "unknown"),
                    "source_file": source_file,
                    "architecture_id": str(row.get("architecture_id") or "unknown"),
                    "method": str(row.get("method") or "unknown"),
                    "role": str(row.get("role") or row.get("agent_id") or "unknown"),
                    "score": score,
                    "contribution_sign": "negative" if score < 0 else ("zero" if score == 0 else "positive"),
                    "full_team_score": _safe_float(row.get("full_team_score")),
                    "ablated_score": _safe_float(row.get("ablated_score")),
                    "full_run_id": str(full_run_id or ""),
                    "ablated_run_id": str(ablated_run_id or ""),
                    "trace_length_bin": str(communication.get("trace_length_bin") or "unknown"),
                    "input_token_bin": str(communication.get("input_token_bin") or "unknown"),
                    "output_token_bin": str(communication.get("output_token_bin") or "unknown"),
                    "role_count_bin": str(communication.get("role_count_bin") or "unknown"),
                    "null_agent_bin": str(communication.get("null_agent_bin") or "unknown"),
                    "trace_event_count": int(communication.get("event_count") or 0),
                    "trace_input_tokens": int(communication.get("input_tokens") or 0),
                    "trace_output_tokens": int(communication.get("output_tokens") or 0),
                    "trace_total_tokens": int(communication.get("total_tokens") or 0),
                    "trace_role_count": int(communication.get("role_count") or 0),
                }
            )
    return rows


def _group_mean(rows: list[dict[str, Any]], keys: list[str], value_key: str = "score") -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "unknown") for key in keys)].append(_safe_float(row.get(value_key)))
    output = []
    for key_values, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        record = {key: value for key, value in zip(keys, key_values)}
        record.update(
            {
                "n": len(values),
                "mean_contribution": round(_mean(values) or 0.0, 6),
                "min_contribution": round(min(values), 6),
                "max_contribution": round(max(values), 6),
            }
        )
        output.append(record)
    return output


def _task_condition_slices(rows: list[dict[str, Any]], axes: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for axis in axes:
        keys = ["dataset", "method", "role", axis]
        for record in _group_mean(rows, keys):
            record["axis"] = axis
            record["axis_value"] = record.pop(axis)
            output.append(record)
    return output


def _communication_slices(rows: list[dict[str, Any]], axes: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for axis in axes:
        keys = ["dataset", "method", "role", axis]
        for record in _group_mean(rows, keys):
            record["axis"] = axis
            record["axis_value"] = record.pop(axis)
            record["analysis_kind"] = "descriptive_non_causal"
            record["causal_claim"] = False
            output.append(record)
    return output


def _rank_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped = _group_mean(rows, keys + ["role"])
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped:
        by_group[tuple(row[key] for key in keys)].append(row)
    ranked = []
    for group_key, group_rows in by_group.items():
        sorted_rows = sorted(group_rows, key=lambda r: r["mean_contribution"], reverse=True)
        for rank, row in enumerate(sorted_rows, start=1):
            record = {key: value for key, value in zip(keys, group_key)}
            record.update(
                {
                    "rank": rank,
                    "role": row["role"],
                    "n": row["n"],
                    "mean_contribution": row["mean_contribution"],
                }
            )
            ranked.append(record)
    return ranked


def _communication_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "trace_event_count",
        "trace_input_tokens",
        "trace_output_tokens",
        "trace_total_tokens",
        "trace_role_count",
    ]
    output = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("trace_event_count", 0) <= 0:
            continue
        grouped[(row["dataset"], row["method"], row["role"])].append(row)
    for (dataset, method, role), group in sorted(grouped.items()):
        ys = [_safe_float(row["score"]) for row in group]
        for metric in metrics:
            xs = [_safe_float(row.get(metric)) for row in group]
            corr = _pearson(xs, ys)
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "role": role,
                    "metric": metric,
                    "n": len(group),
                    "pearson_r": round(corr, 6) if corr is not None else None,
                    "analysis_kind": "descriptive_non_causal",
                    "causal_claim": False,
                }
            )
    return output


def _summary_records(
    rows: list[dict[str, Any]],
    metadata_by_task: dict[str, dict[str, Any]],
    trace_stats: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_counts = Counter(row["dataset"] for row in rows)
    method_counts = Counter(row["method"] for row in rows)
    experiment_counts = Counter(row["experiment_source"] for row in rows)
    sign_counts = Counter(row["contribution_sign"] for row in rows)
    top_roles = _rank_rows(rows, ["dataset", "method"])[:50]
    return [
        {
            "record_type": "overview",
            "selected_tasks": len(metadata_by_task),
            "attribution_rows": len(rows),
            "trace_runs": len(trace_stats),
            "datasets": dict(sorted(dataset_counts.items())),
            "methods": dict(sorted(method_counts.items())),
            "experiment_sources": dict(sorted(experiment_counts.items())),
            "contribution_signs": dict(sorted(sign_counts.items())),
        },
        {
            "record_type": "top_role_rankings",
            "rankings": top_roles,
        },
    ]


def run_generalization(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    root = experiment.benchmark.project_root
    analysis_cfg = experiment.raw.get("analysis") or {}
    outputs = experiment.raw.get("outputs") or {}

    _preflight_required_inputs(root, experiment.raw)

    metadata_by_task, selected_tasks = _select_task_metadata(experiment, max_tasks)
    if not metadata_by_task:
        raise ValueError(
            "exp08 generalization selected 0 task metadata rows. "
            "Check datasets/splits against the configured normalized task files."
        )
    input_cfg = experiment.raw.get("inputs") or {}
    attribution_files = list(input_cfg.get("attribution_files") or [])
    coalition_files = list(input_cfg.get("coalition_files") or [])
    trace_files = list(input_cfg.get("trace_files") or [])

    trace_stats = _load_trace_stats(root, trace_files, selected_tasks)
    coalition_run_map = _load_coalition_run_map(root, coalition_files, selected_tasks)
    attribution_rows = _load_attribution_rows(
        root,
        attribution_files,
        selected_tasks,
        metadata_by_task,
        trace_stats,
        coalition_run_map,
    )
    if not attribution_rows:
        raise ValueError(
            "exp08 generalization loaded 0 attribution rows for the selected tasks. "
            "Complete upstream attribution experiments before running exp08."
        )

    task_axes = list(
        analysis_cfg.get("task_condition_axes")
        or [
            "dataset",
            "difficulty_level",
            "prompt_length_bin",
            "task_type",
        ]
    )
    banned = sorted(axis for axis in task_axes if axis in BANNED_TASK_CONDITION_AXES)
    if banned:
        raise ValueError(
            "exp08 generalization rejects post-treatment task-condition axes: "
            + ", ".join(banned)
        )
    communication_axes = list(
        analysis_cfg.get("communication_condition_axes")
        or [
            "trace_length_bin",
            "input_token_bin",
            "output_token_bin",
            "role_count_bin",
            "null_agent_bin",
        ]
    )

    task_slice_rows = _task_condition_slices(attribution_rows, task_axes)
    communication_slice_rows = _communication_slices(attribution_rows, communication_axes)
    ranking_rows = _rank_rows(attribution_rows, ["dataset", "method", "experiment_source"])
    correlation_rows = _communication_correlations(attribution_rows)
    summary_rows = _summary_records(attribution_rows, metadata_by_task, trace_stats)

    statistics_rows = (
        [{"record_type": "task_slice", **row} for row in task_slice_rows]
        + [{"record_type": "communication_slice", **row} for row in communication_slice_rows]
        + [{"record_type": "role_ranking", **row} for row in ranking_rows]
        + [{"record_type": "communication_correlation", **row} for row in correlation_rows]
        + summary_rows
    )

    output_paths = {
        "summary_file": outputs.get("summary_file", "data/results/statistics/generalization_summary.jsonl"),
        "task_slice_file": outputs.get("task_slice_file", "data/results/statistics/generalization_task_slices.jsonl"),
        "communication_slice_file": outputs.get(
            "communication_slice_file",
            "data/results/statistics/generalization_communication_slices.jsonl",
        ),
        "ranking_file": outputs.get("ranking_file", "data/results/statistics/generalization_role_rankings.jsonl"),
        "correlation_file": outputs.get(
            "correlation_file",
            "data/results/statistics/generalization_communication_correlations.jsonl",
        ),
        "feature_file": outputs.get("feature_file", "data/results/features/generalization_attribution_features.jsonl"),
        "statistics_file": outputs.get("statistics_file", "data/results/statistics/generalization.jsonl"),
    }

    for relative in output_paths.values():
        ensure_dir(_as_path(root, relative).parent)

    write_jsonl(_as_path(root, output_paths["summary_file"]), summary_rows)
    write_jsonl(_as_path(root, output_paths["task_slice_file"]), task_slice_rows)
    write_jsonl(_as_path(root, output_paths["communication_slice_file"]), communication_slice_rows)
    write_jsonl(_as_path(root, output_paths["ranking_file"]), ranking_rows)
    write_jsonl(_as_path(root, output_paths["correlation_file"]), correlation_rows)
    write_jsonl(_as_path(root, output_paths["feature_file"]), attribution_rows)
    write_jsonl(_as_path(root, output_paths["statistics_file"]), statistics_rows)

    return {
        "experiment_id": experiment.experiment_id,
        "selected_tasks": len(metadata_by_task),
        "score_runs": 0,
        "trace_runs": len(trace_stats),
        "coalition_run_map_entries": len(coalition_run_map),
        "attribution_rows": len(attribution_rows),
        "task_slice_rows": len(task_slice_rows),
        "communication_slice_rows": len(communication_slice_rows),
        "ranking_rows": len(ranking_rows),
        "correlation_rows": len(correlation_rows),
        "outputs": {key: str(_as_path(root, value)) for key, value in output_paths.items()},
        "llm_calls": 0,
    }
