"""Run full-system MAS experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mas_contribution_bench.runners.common import (
    backup_existing_file,
    completed_run_ids,
    load_experiment,
    print_progress,
    run_mas_once,
    select_architectures,
    select_tasks,
    write_run_bundle,
)
from mas_contribution_bench.utils.io import append_jsonl


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def run_full_system(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    architectures = select_architectures(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    outputs = experiment.raw.get("outputs", {})
    run_file = f"{outputs.get('run_dir', 'data/runs/full_system')}/runs.jsonl"
    trace_file = f"{outputs.get('trace_dir', 'data/traces')}/{experiment.experiment_id}_traces.jsonl"
    evaluation_file = outputs.get("score_file", f"data/results/scores/{experiment.experiment_id}.jsonl")

    if not _use_checkpointing():
        runs, traces, evaluations = [], [], []
        for seed in seeds:
            for architecture_id in architectures:
                if architecture_id not in experiment.benchmark.architectures:
                    continue
                for task in tasks:
                    run, trace, evaluation = run_mas_once(experiment, task, architecture_id, int(seed))
                    runs.append(run)
                    traces.extend(trace)
                    evaluations.append(evaluation)
        return write_run_bundle(
            experiment,
            runs,
            traces,
            evaluations,
            run_file=run_file,
            trace_file=trace_file,
            evaluation_file=evaluation_file,
        )

    root = experiment.benchmark.project_root
    run_path = root / run_file
    trace_path = root / trace_file
    eval_path = root / evaluation_file
    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if fresh:
        for path in (run_path, trace_path, eval_path):
            backup = backup_existing_file(path)
            if backup:
                print_progress(f"[backup] {path} -> {backup}")
            path.unlink(missing_ok=True)

    done = completed_run_ids(run_path)
    total = len(tasks) * len(seeds) * sum(1 for arch in architectures if arch in experiment.benchmark.architectures)
    completed = len(done)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    print_progress(f"[start] experiment={experiment.experiment_id} total_runs={total} already_done={completed}")
    for task_index, task in enumerate(tasks, start=1):
        for architecture_id in architectures:
            if architecture_id not in experiment.benchmark.architectures:
                print_progress(f"[skip_arch] unknown architecture={architecture_id}")
                continue
            for seed in seeds:
                run_id_seed = f"task={task.get('task_id')} arch={architecture_id} seed={seed}"
                from mas_contribution_bench.utils.io import stable_id

                architecture = experiment.benchmark.architectures[architecture_id]
                role_map_items = sorted((role, role) for role in architecture.roles)
                run_id = stable_id(
                    experiment.experiment_id,
                    task["task_id"],
                    architecture_id,
                    seed,
                    [],
                    "default",
                    role_map_items,
                    [],
                )
                if run_id in done:
                    print_progress(f"[skip] {completed}/{total} {run_id_seed} run_id={run_id}")
                    continue
                print_progress(f"[run] {completed + 1}/{total} task_index={task_index}/{len(tasks)} {run_id_seed}")
                try:
                    run, trace, evaluation = run_mas_once(experiment, task, architecture_id, seed)
                except Exception as exc:
                    print_progress(f"[error] {run_id_seed} {type(exc).__name__}: {exc}")
                    raise
                append_jsonl(run_path, [run])
                append_jsonl(trace_path, trace)
                append_jsonl(eval_path, [evaluation])
                done.add(run.run_id)
                completed += 1
                written_runs += 1
                written_traces += len(trace)
                written_evaluations += 1
                print_progress(
                    f"[done] {completed}/{total} run_id={run.run_id} "
                    f"score={getattr(evaluation, 'score', None)} passed={getattr(evaluation, 'passed', None)} "
                    f"failure={getattr(evaluation, 'failure_type', None)} traces={len(trace)}"
                )
    summary = {
        "runs": completed,
        "new_runs": written_runs,
        "traces": written_traces,
        "evaluations": completed,
        "new_evaluations": written_evaluations,
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(eval_path),
        "checkpointing": True,
    }
    print_progress(f"[complete] {summary}")
    return summary
