"""Run attribution experiments with checkpointed outputs."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from mas_contribution_bench.data.schemas import (
    AttributionMethod,
    AttributionRecord,
    CoalitionInfo,
    RemovalProtocol,
)
from mas_contribution_bench.runners.common import (
    WIRED_ATTRIBUTION_METHODS,
    backup_existing_file,
    completed_run_ids,
    execution_fingerprint,
    execution_treatment,
    identity_role_map,
    listed_attribution_methods,
    load_experiment,
    mas_run_id,
    print_progress,
    require_evaluation_score,
    run_mas_once,
    select_architectures,
    select_tasks,
    validate_attribution_methods,
)
from mas_contribution_bench.utils.io import append_jsonl, iter_jsonl, stable_id


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _standard_error(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    avg = _mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def _reuse_full_system_runs_requested(experiment: Any) -> bool:
    value = (experiment.raw.get("attribution") or {}).get("reuse_full_system_runs")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _reject_full_system_reuse(experiment: Any) -> None:
    if not _reuse_full_system_runs_requested(experiment):
        return
    raise ValueError(
        "attribution.reuse_full_system_runs=true is rejected: "
        "treatment-aware donor reuse is not implemented. "
        "This experiment must compute its own grand coalition under the current "
        "executable specification and runtime treatment. "
        "Set attribution.reuse_full_system_runs to false."
    )


def _completed_attribution_ids(path: Path) -> set[str]:
    return {str(row.get("attribution_id")) for row in iter_jsonl(path) if row.get("attribution_id")}


def _coalition_key(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    active_agents: set[str] | list[str] | tuple[str, ...],
    fingerprint: str,
) -> str:
    return stable_id(
        experiment_id,
        task_id,
        architecture_id,
        seed,
        "coalition",
        sorted(active_agents),
        fingerprint,
    )


def _loo_attribution_id(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    agent: str,
    fingerprint: str,
) -> str:
    return stable_id(experiment_id, task_id, architecture_id, seed, agent, "loo", fingerprint)


def _sampled_attribution_id(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    method: str,
    agent: str,
    fingerprint: str,
) -> str:
    return stable_id(experiment_id, task_id, architecture_id, seed, method, agent, fingerprint)


def _load_coalition_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = row.get("coalition_id")
        if key:
            cache[str(key)] = row
    return cache


def _cfg_int(cfg: dict[str, Any], *paths: str, default: int) -> int:
    for path in paths:
        current: Any = cfg
        ok = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok and isinstance(current, (int, float, str)):
            try:
                return int(current)
            except (TypeError, ValueError):
                pass
    return default


def _removal_protocol(attribution_cfg: dict[str, Any]) -> str:
    return str(
        attribution_cfg.get("primary_removal_protocol")
        or attribution_cfg.get("removal_protocol")
        or "null_agent_replacement"
    )


def _resolve_attribution_methods(experiment: Any) -> list[str]:
    attribution_cfg = experiment.raw.get("attribution") or {}
    methods = listed_attribution_methods(attribution_cfg)
    if not methods:
        experiment_id = str(experiment.experiment_id)
        if "loo" in experiment_id:
            methods = ["loo"]
        elif any(key in experiment_id for key in ("shapley", "banzhaf")):
            raise ValueError(
                f"{experiment_id} must list attribution.methods as shapley_sampled and/or "
                "banzhaf_sampled. Silent fallback to LOO is disabled. "
                "Myerson, Owen, and exact Shapley/Banzhaf estimators remain unwired and are rejected."
            )
        else:
            raise ValueError(
                f"{experiment_id} does not specify an attribution method. "
                f"Supported methods: {', '.join(sorted(WIRED_ATTRIBUTION_METHODS))}. "
                "Myerson, Owen, and exact Shapley/Banzhaf estimators remain unwired and are rejected."
            )
    return validate_attribution_methods(
        methods,
        WIRED_ATTRIBUTION_METHODS,
        context=f"attribution runner ({experiment.experiment_id})",
    )


def run_loo_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    _reject_full_system_reuse(experiment)
    methods = listed_attribution_methods(experiment.raw.get("attribution") or {})
    if methods:
        validate_attribution_methods(
            methods,
            {"loo"},
            context=f"LOO attribution ({experiment.experiment_id})",
        )
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    architectures = select_architectures(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    attribution_cfg = experiment.raw.get("attribution", {})
    protocol = _removal_protocol(attribution_cfg)
    fingerprint = execution_fingerprint(experiment, removal_protocol=protocol)
    treatment = execution_treatment(experiment, removal_protocol=protocol)
    outputs = experiment.raw.get("outputs", {})

    root = experiment.benchmark.project_root
    run_path = root / f"{outputs.get('run_dir', 'data/runs/loo')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/loo')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get("attribution_file", f"data/results/attribution/{experiment.experiment_id}.jsonl")
    evaluation_path = root / outputs.get("score_file", f"data/results/scores/{experiment.experiment_id}_scores.jsonl")

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, evaluation_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done_runs = completed_run_ids(run_path) if _use_checkpointing() else set()
    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()

    architecture_roles = [
        (architecture_id, experiment.benchmark.architectures[architecture_id].roles)
        for architecture_id in architectures
        if architecture_id in experiment.benchmark.architectures
    ]
    total = len(tasks) * len(seeds) * sum(len(roles) for _, roles in architecture_roles)
    completed = len(done_attr)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    print_progress(f"[start] experiment={experiment.experiment_id} total_attributions={total} already_done={completed}")

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles in architecture_roles:
            architecture = experiment.benchmark.architectures[architecture_id]
            role_map_items = sorted(identity_role_map(architecture.roles).items())
            for seed in seeds:
                pending = [
                    agent
                    for agent in roles
                    if _loo_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                if not pending:
                    print_progress(
                        f"[skip-group] task={task['task_id']} arch={architecture_id} seed={seed} "
                        f"completed_attributions={len(roles)}"
                    )
                    continue

                print_progress(
                    f"[full] task={task['task_id']} arch={architecture_id} seed={seed}; "
                    "running current experiment grand coalition"
                )
                full_run, full_traces, full_eval = run_mas_once(
                    experiment,
                    task,
                    architecture_id,
                    int(seed),
                    removed_agents=set(),
                    removal_protocol=protocol,
                )
                full_score = require_evaluation_score(
                    full_eval,
                    dataset=task.get("dataset"),
                    task_id=task.get("task_id"),
                )
                append_jsonl(run_path, [full_run])
                append_jsonl(trace_path, full_traces)
                append_jsonl(evaluation_path, [full_eval])
                done_runs.add(full_run.run_id)
                written_runs += 1
                written_traces += len(full_traces)
                written_evaluations += 1
                full_info = {
                    "score": full_score,
                    "run_id": full_run.run_id,
                    "passed": getattr(full_eval, "passed", None),
                    "failure_type": getattr(full_eval, "failure_type", None),
                    "source": "current_experiment_grand_coalition",
                }

                for agent in roles:
                    attribution_id = _loo_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        agent,
                        fingerprint,
                    )
                    label = f"task={task['task_id']} arch={architecture_id} seed={seed} remove={agent}"
                    if attribution_id in done_attr:
                        print_progress(f"[skip] {completed}/{total} {label} attribution_id={attribution_id}")
                        continue

                    ablated_run_id = mas_run_id(
                        experiment,
                        task["task_id"],
                        architecture_id,
                        seed,
                        removed_agents=[agent],
                        removal_protocol=protocol,
                        role_map_items=role_map_items,
                    )
                    print_progress(f"[run] {completed + 1}/{total} task_index={task_index}/{len(tasks)} {label}")
                    if ablated_run_id in done_runs:
                        print_progress(f"[warn] ablated run already exists without attribution; rerunning {ablated_run_id}")

                    ablated_run, traces, ablated_eval = run_mas_once(
                        experiment,
                        task,
                        architecture_id,
                        int(seed),
                        removed_agents={agent},
                        removal_protocol=protocol,
                    )
                    ablated_score = require_evaluation_score(
                        ablated_eval,
                        dataset=task.get("dataset"),
                        task_id=task.get("task_id"),
                    )

                    record = AttributionRecord(
                        attribution_id=attribution_id,
                        experiment_id=experiment.experiment_id,
                        task_id=task["task_id"],
                        dataset=task["dataset"],
                        architecture_id=architecture_id,
                        agent_id=agent,
                        role=agent,
                        method=AttributionMethod.LOO,
                        score=full_score - ablated_score,
                        baseline_score=0.0,
                        coalition=CoalitionInfo(
                            active_agents=[role for role in architecture.roles if role != agent],
                            removed_agents=[agent],
                        ),
                        removal_protocol=RemovalProtocol(protocol),
                        full_team_score=full_score,
                        ablated_score=ablated_score,
                        sampling_seed=int(seed),
                        metadata={
                            "full_run_id": full_info.get("run_id"),
                            "ablated_run_id": ablated_run.run_id,
                            "full_passed": full_info.get("passed"),
                            "full_failure_type": full_info.get("failure_type"),
                            "full_source": "current_experiment_grand_coalition",
                            "execution_fingerprint": fingerprint,
                            "execution_treatment": treatment,
                        },
                    )

                    append_jsonl(run_path, [ablated_run])
                    append_jsonl(trace_path, traces)
                    append_jsonl(evaluation_path, [ablated_eval])
                    append_jsonl(attribution_path, [record])

                    done_runs.add(ablated_run.run_id)
                    done_attr.add(record.attribution_id)
                    completed += 1
                    written_runs += 1
                    written_traces += len(traces)
                    written_evaluations += 1
                    written_attribution += 1
                    print_progress(
                        f"[done] {completed}/{total} attribution_id={record.attribution_id} "
                        f"full={full_score} ablated={ablated_score} contribution={record.score} "
                        f"failure={ablated_eval.failure_type} traces={len(traces)}"
                    )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "attribution_file": str(attribution_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "checkpointing": _use_checkpointing(),
        "execution_fingerprint": fingerprint,
    }
    print_progress(f"[complete] {summary}")
    return summary


def run_coalition_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    """Run sampled Shapley/Banzhaf attribution with null-agent coalition replacement.

    A coalition is evaluated by keeping active_agents real and replacing all
    other agents with null agents through run_mas_once(..., removed_agents=...).
    """

    experiment = load_experiment(config_path)
    _reject_full_system_reuse(experiment)
    methods = listed_attribution_methods(experiment.raw.get("attribution") or {})
    methods = validate_attribution_methods(
        methods,
        {"shapley_sampled", "banzhaf_sampled"},
        context=f"coalition attribution ({experiment.experiment_id})",
    )
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    architectures = select_architectures(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    attribution_cfg = experiment.raw.get("attribution", {})
    protocol = _removal_protocol(attribution_cfg)
    fingerprint = execution_fingerprint(experiment, removal_protocol=protocol)
    treatment = execution_treatment(experiment, removal_protocol=protocol)

    shapley_samples = _cfg_int(
        attribution_cfg,
        "shapley.num_permutations",
        "shapley.num_samples",
        "num_samples",
        default=8,
    )
    banzhaf_samples = _cfg_int(
        attribution_cfg,
        "banzhaf.num_coalitions",
        "banzhaf.num_samples",
        "num_samples",
        default=8,
    )

    outputs = experiment.raw.get("outputs", {})
    root = experiment.benchmark.project_root

    run_path = root / f"{outputs.get('run_dir', 'data/runs/shapley')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/shapley')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get(
        "attribution_file",
        f"data/results/attribution/{experiment.experiment_id}.jsonl",
    )
    coalition_path = root / outputs.get(
        "coalition_file",
        f"data/results/attribution/{experiment.experiment_id}_coalitions.jsonl",
    )
    evaluation_path = root / outputs.get(
        "score_file",
        f"data/results/scores/{experiment.experiment_id}_scores.jsonl",
    )

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, coalition_path, evaluation_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()
    coalition_cache = _load_coalition_cache(coalition_path) if _use_checkpointing() else {}

    architecture_roles = [
        (architecture_id, experiment.benchmark.architectures[architecture_id].roles)
        for architecture_id in architectures
        if architecture_id in experiment.benchmark.architectures
    ]

    total = len(tasks) * len(seeds) * sum(len(roles) for _, roles in architecture_roles) * len(methods)
    completed = len(done_attr)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    written_coalitions = 0

    print_progress(
        f"[start] experiment={experiment.experiment_id} methods={methods} "
        f"total_attributions={total} already_done={completed}"
    )

    def evaluate_coalition(
        task: dict[str, Any],
        architecture_id: str,
        seed: int,
        roles: list[str],
        active_agents: set[str],
    ) -> dict[str, Any]:
        active_agents = set(active_agents)
        removed_agents = set(roles) - active_agents
        coalition_id = _coalition_key(
            experiment.experiment_id,
            str(task["task_id"]),
            architecture_id,
            seed,
            active_agents,
            fingerprint,
        )

        cached = coalition_cache.get(coalition_id)
        if cached is not None:
            require_evaluation_score(
                cached,
                dataset=task.get("dataset") or cached.get("dataset"),
                task_id=task.get("task_id") or cached.get("task_id"),
            )
            return cached

        print_progress(
            f"[coalition] task={task['task_id']} arch={architecture_id} seed={seed} "
            f"active={sorted(active_agents)} removed={sorted(removed_agents)}"
        )
        run, traces, evaluation = run_mas_once(
            experiment,
            task,
            architecture_id,
            int(seed),
            removed_agents=removed_agents,
            removal_protocol=protocol,
        )
        score = require_evaluation_score(
            evaluation,
            dataset=task.get("dataset"),
            task_id=task.get("task_id"),
        )
        row = {
            "coalition_id": coalition_id,
            "experiment_id": experiment.experiment_id,
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "architecture_id": architecture_id,
            "sampling_seed": int(seed),
            "active_agents": sorted(active_agents),
            "removed_agents": sorted(removed_agents),
            "score": score,
            "run_id": run.run_id,
            "passed": getattr(evaluation, "passed", None),
            "failure_type": getattr(evaluation, "failure_type", None),
            "source": "current_experiment_grand_coalition" if not removed_agents else "coalition_run",
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        }
        coalition_cache[coalition_id] = row

        append_jsonl(run_path, [run])
        append_jsonl(trace_path, traces)
        append_jsonl(evaluation_path, [evaluation])
        append_jsonl(coalition_path, [row])

        nonlocal written_runs, written_traces, written_evaluations, written_coalitions
        written_runs += 1
        written_traces += len(traces)
        written_evaluations += 1
        written_coalitions += 1
        return row

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles in architecture_roles:
            roles = list(roles)
            all_agents = set(roles)

            for seed in seeds:
                full_info = evaluate_coalition(task, architecture_id, int(seed), roles, all_agents)
                full_score = require_evaluation_score(
                    full_info,
                    dataset=task.get("dataset"),
                    task_id=task.get("task_id"),
                )

                for method in methods:
                    rng = random.Random(
                        stable_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            method,
                        )
                    )

                    marginals_by_agent: dict[str, list[float]] = defaultdict(list)
                    example_coalition_by_agent: dict[str, list[str]] = {}
                    example_permutation_by_agent: dict[str, list[str]] = {}

                    if method == "shapley_sampled":
                        for sample_index in range(shapley_samples):
                            permutation = roles[:]
                            rng.shuffle(permutation)

                            active: set[str] = set()
                            prev_score = require_evaluation_score(
                                evaluate_coalition(task, architecture_id, int(seed), roles, active),
                                dataset=task.get("dataset"),
                                task_id=task.get("task_id"),
                            )

                            for agent in permutation:
                                before = set(active)
                                active.add(agent)
                                current_score = require_evaluation_score(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, active),
                                    dataset=task.get("dataset"),
                                    task_id=task.get("task_id"),
                                )
                                marginal = current_score - prev_score
                                marginals_by_agent[agent].append(marginal)
                                example_coalition_by_agent.setdefault(agent, sorted(before))
                                example_permutation_by_agent.setdefault(agent, permutation[:])
                                prev_score = current_score

                            print_progress(
                                f"[sample] method={method} task={task['task_id']} arch={architecture_id} "
                                f"seed={seed} sample={sample_index + 1}/{shapley_samples}"
                            )

                    elif method == "banzhaf_sampled":
                        for sample_index in range(banzhaf_samples):
                            for agent in roles:
                                others = [role for role in roles if role != agent]
                                subset = {role for role in others if rng.random() < 0.5}
                                with_agent = set(subset)
                                with_agent.add(agent)

                                without_score = require_evaluation_score(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, subset),
                                    dataset=task.get("dataset"),
                                    task_id=task.get("task_id"),
                                )
                                with_score = require_evaluation_score(
                                    evaluate_coalition(task, architecture_id, int(seed), roles, with_agent),
                                    dataset=task.get("dataset"),
                                    task_id=task.get("task_id"),
                                )
                                marginal = with_score - without_score
                                marginals_by_agent[agent].append(marginal)
                                example_coalition_by_agent.setdefault(agent, sorted(subset))

                            print_progress(
                                f"[sample] method={method} task={task['task_id']} arch={architecture_id} "
                                f"seed={seed} sample={sample_index + 1}/{banzhaf_samples}"
                            )

                    for agent in roles:
                        sample_count = len(marginals_by_agent.get(agent, []))
                        attribution_id = _sampled_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            method,
                            agent,
                            fingerprint,
                        )
                        label = (
                            f"method={method} task={task['task_id']} arch={architecture_id} "
                            f"seed={seed} agent={agent}"
                        )
                        if attribution_id in done_attr:
                            print_progress(f"[skip] {completed}/{total} {label} attribution_id={attribution_id}")
                            continue

                        values = marginals_by_agent.get(agent, [])
                        score = _mean(values)
                        stderr = _standard_error(values)

                        active_example = example_coalition_by_agent.get(agent, [])
                        removed_example = [role for role in roles if role not in set(active_example)]

                        record = AttributionRecord(
                            attribution_id=attribution_id,
                            experiment_id=experiment.experiment_id,
                            task_id=task["task_id"],
                            dataset=task["dataset"],
                            architecture_id=architecture_id,
                            agent_id=agent,
                            role=agent,
                            method=AttributionMethod(method),
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=active_example,
                                removed_agents=removed_example,
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=None,
                            sampling_seed=int(seed),
                            num_samples=sample_count,
                            permutation_order=example_permutation_by_agent.get(agent),
                            standard_error=stderr,
                            metadata={
                                "method": method,
                                "marginal_values": values,
                                "coalition_cache_size": len(coalition_cache),
                                "shapley_samples": shapley_samples if method == "shapley_sampled" else None,
                                "banzhaf_samples": banzhaf_samples if method == "banzhaf_sampled" else None,
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )

                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1

                        print_progress(
                            f"[done] {completed}/{total} method={method} attribution_id={record.attribution_id} "
                            f"agent={agent} score={score} samples={sample_count} stderr={stderr}"
                        )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "coalitions": written_coalitions,
        "attribution_file": str(attribution_path),
        "coalition_file": str(coalition_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "checkpointing": _use_checkpointing(),
        "methods": methods,
        "shapley_samples": shapley_samples,
        "banzhaf_samples": banzhaf_samples,
        "execution_fingerprint": fingerprint,
    }
    print_progress(f"[complete] {summary}")
    return summary


def run_attribution(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    _reject_full_system_reuse(experiment)
    methods = _resolve_attribution_methods(experiment)
    loo_methods = [method for method in methods if method == "loo"]
    sampled_methods = [method for method in methods if method in {"shapley_sampled", "banzhaf_sampled"}]
    if loo_methods and sampled_methods:
        raise ValueError(
            "Cannot mix loo with shapley_sampled/banzhaf_sampled in one attribution run. "
            "Use separate experiment configs."
        )
    if loo_methods:
        return run_loo_attribution(config_path, max_tasks=max_tasks)
    return run_coalition_attribution(config_path, max_tasks=max_tasks)
