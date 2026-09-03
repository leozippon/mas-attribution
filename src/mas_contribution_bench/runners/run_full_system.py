"""Run full-system MAS experiments."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from mas_contribution_bench.runners.common import (
    backup_existing_file,
    completed_run_ids,
    identity_role_map,
    load_experiment,
    mas_run_id,
    print_progress,
    run_mas_once,
    select_architectures,
    select_tasks,
    write_run_bundle,
)
from mas_contribution_bench.utils.io import append_jsonl


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def _run_concurrency() -> int:
    raw = os.getenv("MAS_RUN_CONCURRENCY", "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"MAS_RUN_CONCURRENCY must be an integer, got {raw!r}") from None
    return max(1, value)


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
    concurrency = _run_concurrency()
    print_progress(
        f"[start] experiment={experiment.experiment_id} total_runs={total} "
        f"already_done={completed} concurrency={concurrency}"
    )
    pending: list[tuple[int, int, dict[str, Any], str, int, str]] = []
    for task_index, task in enumerate(tasks, start=1):
        for architecture_id in architectures:
            if architecture_id not in experiment.benchmark.architectures:
                print_progress(f"[skip_arch] unknown architecture={architecture_id}")
                continue
            for seed in seeds:
                run_id_seed = f"task={task.get('task_id')} arch={architecture_id} seed={seed}"
                architecture = experiment.benchmark.architectures[architecture_id]
                run_id = mas_run_id(
                    experiment,
                    task["task_id"],
                    architecture_id,
                    seed,
                    removed_agents=[],
                    removal_protocol="none",
                    role_map_items=sorted(identity_role_map(architecture.roles).items()),
                    permission_overrides=None,
                    condition_id=None,
                )
                if run_id in done:
                    print_progress(f"[skip] {completed}/{total} {run_id_seed} run_id={run_id}")
                    continue
                pending.append((task_index, len(tasks), task, architecture_id, seed, run_id_seed))

    if concurrency <= 1:
        for task_index, task_count, task, architecture_id, seed, run_id_seed in pending:
            print_progress(f"[run] {completed + 1}/{total} task_index={task_index}/{task_count} {run_id_seed}")
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
    else:
        pending_iter = iter(pending)
        in_flight: dict[Any, tuple[int, int, str]] = {}

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            try:
                task_index, task_count, task, architecture_id, seed, run_id_seed = next(pending_iter)
            except StopIteration:
                return False
            print_progress(
                f"[submit] {completed + len(in_flight) + 1}/{total} "
                f"task_index={task_index}/{task_count} {run_id_seed}"
            )
            future = executor.submit(run_mas_once, experiment, task, architecture_id, seed)
            in_flight[future] = (task_index, task_count, run_id_seed)
            return True

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(concurrency):
                if not submit_next(executor):
                    break
            while in_flight:
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    task_index, task_count, run_id_seed = in_flight.pop(future)
                    try:
                        run, trace, evaluation = future.result()
                    except Exception as exc:
                        print_progress(f"[error] {run_id_seed} {type(exc).__name__}: {exc}")
                        executor.shutdown(wait=False, cancel_futures=True)
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
                        f"[done] {completed}/{total} task_index={task_index}/{task_count} "
                        f"run_id={run.run_id} score={getattr(evaluation, 'score', None)} "
                        f"passed={getattr(evaluation, 'passed', None)} "
                        f"failure={getattr(evaluation, 'failure_type', None)} traces={len(trace)}"
                    )
                    submit_next(executor)
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
