"""Run intervention experiments with schema-compatible outputs."""

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
from mas_contribution_bench.graphs.architectures import controlled_architecture
from mas_contribution_bench.runners.common import (
    WIRED_INTERVENTION_METHODS,
    backup_existing_file,
    execution_fingerprint,
    execution_treatment,
    listed_attribution_methods,
    load_experiment,
    print_progress,
    require_evaluation_score,
    run_mas_once,
    select_architectures,
    select_tasks,
    validate_attribution_methods,
    validate_permission_toggles,
)
from mas_contribution_bench.utils.io import append_jsonl, iter_jsonl, stable_id


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _standard_error(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    avg = _mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def _completed_attribution_ids(path: Path) -> set[str]:
    return {str(row.get("attribution_id")) for row in iter_jsonl(path) if row.get("attribution_id")}


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


def _intervention_methods(experiment: Any) -> list[str]:
    return validate_attribution_methods(
        listed_attribution_methods(experiment.raw.get("attribution") or {}, default=["loo"]),
        WIRED_INTERVENTION_METHODS,
        context=f"intervention runner ({experiment.experiment_id})",
    )


def _variant_edges(variant: dict[str, Any]) -> list[tuple[str, str]]:
    edges = []
    for edge in variant.get("edges", []):
        if len(edge) != 2:
            continue
        src, dst = edge
        edges.append((str(src), str(dst)))
    return edges


def _graph_feature_row(variant_id: str, variant: dict[str, Any], roles: list[str]) -> dict[str, Any]:
    edges = _variant_edges(variant)
    role_set = set(roles)
    in_degree = {role: 0 for role in roles}
    out_degree = {role: 0 for role in roles}

    for src, dst in edges:
        if src in role_set:
            out_degree[src] += 1
        if dst in role_set:
            in_degree[dst] += 1

    return {
        "variant_id": variant_id,
        "template": variant.get("template"),
        "center": variant.get("center"),
        "num_roles": len(roles),
        "num_edges": len(edges),
        "roles": roles,
        "edges": [[src, dst] for src, dst in edges],
        "degree": {role: in_degree[role] + out_degree[role] for role in roles},
        "in_degree": in_degree,
        "out_degree": out_degree,
        "fan_in": max(in_degree.values()) if in_degree else 0,
        "fan_out": max(out_degree.values()) if out_degree else 0,
    }


def _clone_architecture(base_architecture: Any, variant: dict[str, Any], roles: list[str]) -> Any:
    variant_id = str(variant["id"])
    raw = dict(variant)
    raw["roles"] = list(roles)
    raw["controlled_role_set"] = list(roles)
    raw["edges"] = [[src, dst] for src, dst in _variant_edges(variant)]
    raw["name"] = variant.get("description", variant_id)
    raw["family"] = variant.get("template", "controlled")
    raw["entrypoint"] = roles[0] if roles else ""
    raw["terminal_nodes"] = ["final_answer"]
    orchestration = dict(variant.get("orchestration") or {})
    orchestration["max_rounds"] = 1
    template = str(variant.get("template") or "")
    if template in {"chain", "dag"}:
        orchestration.setdefault("execution_mode", "dag")
    elif orchestration.get("execution_mode") in {None, "", "debate"}:
        orchestration["execution_mode"] = "single_pass"
    raw["orchestration"] = orchestration
    return controlled_architecture(raw, architecture_id=variant_id)


def _inject_topology_variants(experiment: Any) -> list[tuple[str, list[str], dict[str, Any]]]:
    roles = [str(role) for role in experiment.raw.get("controlled_role_set", [])]
    variants = experiment.raw.get("topology_variants", [])

    if not roles:
        raise ValueError("exp05 requires controlled_role_set.")
    if not variants:
        raise ValueError("exp05 requires topology_variants.")

    architectures = getattr(experiment.benchmark, "architectures", {})
    if not architectures:
        raise ValueError("No benchmark architectures loaded; cannot build controlled topology variants.")

    base_architecture = next(iter(architectures.values()))

    injected: list[tuple[str, list[str], dict[str, Any]]] = []
    for variant in variants:
        variant_id = str(variant["id"])
        architectures[variant_id] = _clone_architecture(base_architecture, variant, roles)
        injected.append((variant_id, roles, variant))

    return injected


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


def _topology_attribution_id(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    method: str,
    agent: str,
    fingerprint: str,
) -> str:
    return stable_id(experiment_id, task_id, architecture_id, seed, method, agent, fingerprint)


def run_topology_intervention(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    attribution_cfg = experiment.raw.get("attribution", {})
    methods = _intervention_methods(experiment)
    protocol = str(attribution_cfg.get("removal_protocol", "null_agent_replacement"))
    fingerprint = execution_fingerprint(experiment, removal_protocol=protocol)
    treatment = execution_treatment(experiment, removal_protocol=protocol)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    topology_variants = _inject_topology_variants(experiment)
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    shapley_samples = _cfg_int(
        attribution_cfg,
        "shapley.num_permutations",
        "shapley.num_samples",
        "num_samples",
        "permutation_samples",
        default=8,
    )

    outputs = experiment.raw.get("outputs", {})
    root = experiment.benchmark.project_root

    run_path = root / f"{outputs.get('run_dir', 'data/runs/interventions/topology')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/interventions/topology')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get(
        "attribution_file",
        "data/results/attribution/topology_intervention_attribution.jsonl",
    )
    coalition_path = root / outputs.get(
        "coalition_file",
        "data/results/attribution/topology_intervention_coalitions.jsonl",
    )
    evaluation_path = root / outputs.get(
        "score_file",
        f"data/results/scores/{experiment.experiment_id}_scores.jsonl",
    )
    statistics_path = root / outputs.get(
        "statistics_file",
        "data/results/statistics/topology_intervention.jsonl",
    )

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, coalition_path, evaluation_path, statistics_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()
    coalition_cache = _load_coalition_cache(coalition_path) if _use_checkpointing() else {}

    total = len(tasks) * len(seeds) * sum(len(roles) * len(methods) for _, roles, _ in topology_variants)
    completed = len(done_attr)

    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    written_coalitions = 0
    coalition_cache_hits = 0

    print_progress(
        f"[start] experiment={experiment.experiment_id} variants={len(topology_variants)} "
        f"methods={methods} total_attributions={total} already_done={completed}"
    )

    feature_rows = [
        _graph_feature_row(variant_id, variant, roles)
        for variant_id, roles, variant in topology_variants
    ]
    if feature_rows and not statistics_path.exists():
        append_jsonl(statistics_path, feature_rows)

    def evaluate_coalition(
        task: dict[str, Any],
        architecture_id: str,
        seed: int,
        roles: list[str],
        active_agents: set[str],
    ) -> dict[str, Any]:
        nonlocal written_runs, written_traces, written_evaluations, written_coalitions, coalition_cache_hits

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
            coalition_cache_hits += 1
            require_evaluation_score(
                cached,
                dataset=task.get("dataset") or cached.get("dataset"),
                task_id=task.get("task_id") or cached.get("task_id"),
            )
            return cached

        print_progress(
            f"[coalition] task={task['task_id']} topology={architecture_id} seed={seed} "
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

        row = {
            "coalition_id": coalition_id,
            "experiment_id": experiment.experiment_id,
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "architecture_id": architecture_id,
            "topology_id": architecture_id,
            "sampling_seed": int(seed),
            "active_agents": sorted(active_agents),
            "removed_agents": sorted(removed_agents),
            "score": require_evaluation_score(
                evaluation,
                dataset=task.get("dataset"),
                task_id=task.get("task_id"),
            ),
            "run_id": run.run_id,
            "passed": getattr(evaluation, "passed", None),
            "failure_type": getattr(evaluation, "failure_type", None),
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        }
        coalition_cache[coalition_id] = row

        append_jsonl(run_path, [run])
        append_jsonl(trace_path, traces)
        append_jsonl(evaluation_path, [evaluation])
        append_jsonl(coalition_path, [row])

        written_runs += 1
        written_traces += len(traces)
        written_evaluations += 1
        written_coalitions += 1
        return row

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles, variant in topology_variants:
            role_set = set(roles)

            for seed in seeds:
                pending_loo = [
                    agent
                    for agent in roles
                    if _topology_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        "loo",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                pending_shapley = [
                    agent
                    for agent in roles
                    if _topology_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        "shapley_sampled",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                if "loo" not in methods:
                    pending_loo = []
                if "shapley_sampled" not in methods:
                    pending_shapley = []
                if not pending_loo and not pending_shapley:
                    completed_for_group = len(roles) * len(methods)
                    print_progress(
                        f"[skip-group] task_index={task_index}/{len(tasks)} topology={architecture_id} "
                        f"seed={seed} completed_attributions={completed_for_group}"
                    )
                    continue

                full_row = evaluate_coalition(task, architecture_id, int(seed), roles, role_set)
                full_score = require_evaluation_score(
                    full_row,
                    dataset=task.get("dataset"),
                    task_id=task.get("task_id"),
                )

                if "loo" in methods:
                    for agent in roles:
                        attribution_id = _topology_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            "loo",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(f"[skip] {completed}/{total} method=loo topology={architecture_id} agent={agent}")
                            continue

                        active_agents = role_set - {agent}
                        ablated_row = evaluate_coalition(task, architecture_id, int(seed), roles, active_agents)
                        ablated_score = require_evaluation_score(
                            ablated_row,
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
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=full_score - ablated_score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=sorted(active_agents),
                                removed_agents=[agent],
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=ablated_score,
                            sampling_seed=int(seed),
                            metadata={
                                "topology_id": architecture_id,
                                "topology_template": variant.get("template"),
                                "full_coalition_id": full_row.get("coalition_id"),
                                "ablated_coalition_id": ablated_row.get("coalition_id"),
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=loo task_index={task_index}/{len(tasks)} "
                            f"topology={architecture_id} agent={agent} score={record.score}"
                        )

                if "shapley_sampled" in methods:
                    if not pending_shapley:
                        print_progress(
                            f"[skip] {completed}/{total} method=shapley_sampled topology={architecture_id} "
                            f"seed={seed} all roles already done"
                        )
                        continue
                    rng = random.Random(
                        stable_id(experiment.experiment_id, task["task_id"], architecture_id, seed, "shapley_sampled")
                    )
                    marginals_by_agent: dict[str, list[float]] = defaultdict(list)
                    example_coalition_by_agent: dict[str, list[str]] = {}
                    example_permutation_by_agent: dict[str, list[str]] = {}

                    for sample_index in range(shapley_samples):
                        permutation = list(roles)
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
                            f"[sample] method=shapley_sampled task={task['task_id']} topology={architecture_id} "
                            f"seed={seed} sample={sample_index + 1}/{shapley_samples}"
                        )

                    for agent in roles:
                        attribution_id = _topology_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            "shapley_sampled",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(
                                f"[skip] {completed}/{total} method=shapley_sampled topology={architecture_id} agent={agent}"
                            )
                            continue

                        values = marginals_by_agent.get(agent, [])
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
                            method=AttributionMethod("shapley_sampled"),
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=_mean(values),
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=active_example,
                                removed_agents=removed_example,
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=None,
                            sampling_seed=int(seed),
                            num_samples=len(values),
                            permutation_order=example_permutation_by_agent.get(agent),
                            standard_error=_standard_error(values),
                            metadata={
                                "topology_id": architecture_id,
                                "topology_template": variant.get("template"),
                                "marginal_values": values,
                                "shapley_samples": shapley_samples,
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=shapley_sampled task_index={task_index}/{len(tasks)} "
                            f"topology={architecture_id} agent={agent} score={record.score}"
                        )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "coalitions": written_coalitions,
        "coalition_cache_hits": coalition_cache_hits,
        "coalition_cache_entries": len(coalition_cache),
        "variants": [variant_id for variant_id, _, _ in topology_variants],
        "methods": methods,
        "shapley_samples": shapley_samples if "shapley_sampled" in methods else 0,
        "attribution_file": str(attribution_path),
        "coalition_file": str(coalition_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "statistics_file": str(statistics_path),
        "checkpointing": _use_checkpointing(),
        "execution_fingerprint": fingerprint,
    }
    print_progress(f"[complete] {summary}")
    return summary


def _role_map_key(agent_role_map: dict[str, str]) -> list[tuple[str, str]]:
    return sorted((str(position), str(functional)) for position, functional in agent_role_map.items())


def _role_coalition_key(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    active_agents: set[str] | list[str] | tuple[str, ...],
    agent_role_map: dict[str, str],
    fingerprint: str,
) -> str:
    return stable_id(
        experiment_id,
        task_id,
        architecture_id,
        seed,
        "role_coalition",
        sorted(active_agents),
        _role_map_key(agent_role_map),
        fingerprint,
    )


def _condition_attribution_id(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    condition_id: str,
    method: str,
    agent: str,
    fingerprint: str,
) -> str:
    return stable_id(
        experiment_id,
        task_id,
        architecture_id,
        seed,
        condition_id,
        method,
        agent,
        fingerprint,
    )


def _role_swap_conditions(
    architecture_id: str,
    roles: list[str],
    role_swaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    role_set = set(roles)
    conditions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for swap_cfg in role_swaps:
        swap_id = str(swap_cfg.get("id") or "_".join(str(x) for x in swap_cfg.get("swap", [])))
        pair = [str(role) for role in swap_cfg.get("swap", [])]
        if len(pair) != 2:
            skipped.append(
                {
                    "architecture_id": architecture_id,
                    "role_swap_id": swap_id,
                    "reason": "swap must contain exactly two roles",
                    "swap": pair,
                }
            )
            continue

        first, second = pair
        if first not in role_set or second not in role_set:
            skipped.append(
                {
                    "architecture_id": architecture_id,
                    "role_swap_id": swap_id,
                    "reason": "one or both roles are absent from this architecture",
                    "swap": pair,
                    "available_roles": roles,
                }
            )
            continue

        identity_map = {role: role for role in roles}
        swapped_map = dict(identity_map)
        swapped_map[first] = second
        swapped_map[second] = first

        conditions.append(
            {
                "condition_id": f"{architecture_id}__{swap_id}__baseline",
                "condition": "baseline",
                "role_swap_id": swap_id,
                "swap": pair,
                "agent_role_map": identity_map,
                "controls": list(swap_cfg.get("controls", [])),
            }
        )
        conditions.append(
            {
                "condition_id": f"{architecture_id}__{swap_id}__swapped",
                "condition": "swapped",
                "role_swap_id": swap_id,
                "swap": pair,
                "agent_role_map": swapped_map,
                "controls": list(swap_cfg.get("controls", [])),
            }
        )

    for row in skipped:
        print_progress(
            f"[skip-swap] architecture={row['architecture_id']} swap={row['role_swap_id']} "
            f"reason={row['reason']}"
        )

    return conditions


def _role_intervention_feature_rows(
    experiment: Any,
    architectures: list[str],
    role_swaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture_id in architectures:
        architecture = experiment.benchmark.architectures.get(architecture_id)
        if architecture is None:
            rows.append(
                {
                    "architecture_id": architecture_id,
                    "status": "missing_architecture",
                }
            )
            continue
        roles = list(architecture.roles)
        for swap_cfg in role_swaps:
            swap_id = str(swap_cfg.get("id") or "_".join(str(x) for x in swap_cfg.get("swap", [])))
            pair = [str(role) for role in swap_cfg.get("swap", [])]
            applicable = len(pair) == 2 and all(role in set(roles) for role in pair)
            rows.append(
                {
                    "architecture_id": architecture_id,
                    "role_swap_id": swap_id,
                    "swap": pair,
                    "applicable": applicable,
                    "available_roles": roles,
                    "controls": list(swap_cfg.get("controls", [])),
                    "design": "topology_preserving_role_swap",
                    "position_roles_fixed": True,
                    "functional_roles_swapped": applicable,
                }
            )
    return rows


def run_role_intervention(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    """Run topology-preserving role-swap interventions.

    In this experiment, graph positions stay fixed while the functional role
    prompt/permission assigned to selected positions is swapped. Attribution is
    still reported over graph-position roles; metadata records the functional
    role used by each position.
    """

    experiment = load_experiment(config_path)
    attribution_cfg = experiment.raw.get("attribution", {})
    methods = _intervention_methods(experiment)
    protocol = str(attribution_cfg.get("removal_protocol", "null_agent_replacement"))
    fingerprint = execution_fingerprint(experiment, removal_protocol=protocol)
    treatment = execution_treatment(experiment, removal_protocol=protocol)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    architectures = select_architectures(experiment)
    if not architectures:
        raise ValueError("exp06 requires base_architectures or architectures.include.")

    role_swaps = list(experiment.raw.get("role_swaps", []))
    if not role_swaps:
        raise ValueError("exp06 requires role_swaps.")

    missing = [architecture_id for architecture_id in architectures if architecture_id not in experiment.benchmark.architectures]
    if missing:
        raise ValueError(f"Unknown architectures in exp06 base_architectures: {missing}")

    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    shapley_samples = _cfg_int(
        attribution_cfg,
        "shapley.num_permutations",
        "shapley.num_samples",
        "num_samples",
        "permutation_samples",
        default=8,
    )

    outputs = experiment.raw.get("outputs", {})
    root = experiment.benchmark.project_root

    run_path = root / f"{outputs.get('run_dir', 'data/runs/interventions/role')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/interventions/role')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get(
        "attribution_file",
        "data/results/attribution/role_intervention_attribution.jsonl",
    )
    coalition_path = root / outputs.get(
        "coalition_file",
        "data/results/attribution/role_intervention_coalitions.jsonl",
    )
    evaluation_path = root / outputs.get(
        "score_file",
        f"data/results/scores/{experiment.experiment_id}_scores.jsonl",
    )
    statistics_path = root / outputs.get(
        "statistics_file",
        "data/results/statistics/role_intervention.jsonl",
    )

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, coalition_path, evaluation_path, statistics_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()
    coalition_cache = _load_coalition_cache(coalition_path) if _use_checkpointing() else {}

    architecture_conditions: list[tuple[str, list[str], dict[str, Any]]] = []
    for architecture_id in architectures:
        roles = list(experiment.benchmark.architectures[architecture_id].roles)
        for condition in _role_swap_conditions(architecture_id, roles, role_swaps):
            architecture_conditions.append((architecture_id, roles, condition))

    if not architecture_conditions:
        raise ValueError("No applicable role_swaps for the selected base_architectures.")

    total = len(tasks) * len(seeds) * sum(
        len(roles) * len(methods)
        for _, roles, _ in architecture_conditions
    )
    completed = len(done_attr)

    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    written_coalitions = 0
    coalition_cache_hits = 0

    print_progress(
        f"[start] experiment={experiment.experiment_id} architectures={architectures} "
        f"conditions={len(architecture_conditions)} methods={methods} "
        f"total_attributions={total} already_done={completed}"
    )

    feature_rows = _role_intervention_feature_rows(experiment, architectures, role_swaps)
    if feature_rows and (fresh or not statistics_path.exists()):
        append_jsonl(statistics_path, feature_rows)

    def evaluate_coalition(
        task: dict[str, Any],
        architecture_id: str,
        seed: int,
        roles: list[str],
        active_agents: set[str],
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal written_runs, written_traces, written_evaluations, written_coalitions, coalition_cache_hits

        active_agents = set(active_agents)
        removed_agents = set(roles) - active_agents
        role_map = {str(k): str(v) for k, v in condition["agent_role_map"].items()}
        coalition_id = _role_coalition_key(
            experiment.experiment_id,
            str(task["task_id"]),
            architecture_id,
            seed,
            active_agents,
            role_map,
            fingerprint,
        )

        cached = coalition_cache.get(coalition_id)
        if cached is not None:
            coalition_cache_hits += 1
            require_evaluation_score(
                cached,
                dataset=task.get("dataset") or cached.get("dataset"),
                task_id=task.get("task_id") or cached.get("task_id"),
            )
            return cached

        print_progress(
            f"[coalition] task={task['task_id']} architecture={architecture_id} seed={seed} "
            f"condition={condition['condition_id']} active={sorted(active_agents)} "
            f"removed={sorted(removed_agents)} role_map={role_map}"
        )
        run, traces, evaluation = run_mas_once(
            experiment,
            task,
            architecture_id,
            int(seed),
            removed_agents=removed_agents,
            removal_protocol=protocol,
            agent_role_map=role_map,
            condition_id=str(condition["condition_id"]),
        )

        row = {
            "coalition_id": coalition_id,
            "experiment_id": experiment.experiment_id,
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "architecture_id": architecture_id,
            "sampling_seed": int(seed),
            "condition_id": condition["condition_id"],
            "condition": condition["condition"],
            "role_swap_id": condition["role_swap_id"],
            "swap": condition["swap"],
            "agent_role_map": role_map,
            "active_agents": sorted(active_agents),
            "removed_agents": sorted(removed_agents),
            "active_functional_roles": sorted({role_map.get(role, role) for role in active_agents}),
            "removed_functional_roles": sorted({role_map.get(role, role) for role in removed_agents}),
            "score": require_evaluation_score(
                evaluation,
                dataset=task.get("dataset"),
                task_id=task.get("task_id"),
            ),
            "run_id": run.run_id,
            "passed": getattr(evaluation, "passed", None),
            "failure_type": getattr(evaluation, "failure_type", None),
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        }
        coalition_cache[coalition_id] = row

        append_jsonl(run_path, [run])
        append_jsonl(trace_path, traces)
        append_jsonl(evaluation_path, [evaluation])
        append_jsonl(coalition_path, [row])

        written_runs += 1
        written_traces += len(traces)
        written_evaluations += 1
        written_coalitions += 1
        return row

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles, condition in architecture_conditions:
            role_set = set(roles)
            role_map = {str(k): str(v) for k, v in condition["agent_role_map"].items()}

            for seed in seeds:
                pending_loo = [
                    agent
                    for agent in roles
                    if _condition_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        condition["condition_id"],
                        "loo",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                pending_shapley = [
                    agent
                    for agent in roles
                    if _condition_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        condition["condition_id"],
                        "shapley_sampled",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                if "loo" not in methods:
                    pending_loo = []
                if "shapley_sampled" not in methods:
                    pending_shapley = []
                if not pending_loo and not pending_shapley:
                    completed_for_group = len(roles) * len(methods)
                    print_progress(
                        f"[skip-group] task_index={task_index}/{len(tasks)} architecture={architecture_id} "
                        f"condition={condition['condition_id']} seed={seed} "
                        f"completed_attributions={completed_for_group}"
                    )
                    continue

                full_row = evaluate_coalition(task, architecture_id, int(seed), roles, role_set, condition)
                full_score = require_evaluation_score(
                    full_row,
                    dataset=task.get("dataset"),
                    task_id=task.get("task_id"),
                )

                if "loo" in methods:
                    for agent in roles:
                        attribution_id = _condition_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "loo",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(
                                f"[skip] {completed}/{total} method=loo architecture={architecture_id} "
                                f"condition={condition['condition_id']} agent={agent}"
                            )
                            continue

                        active_agents = role_set - {agent}
                        ablated_row = evaluate_coalition(
                            task,
                            architecture_id,
                            int(seed),
                            roles,
                            active_agents,
                            condition,
                        )
                        ablated_score = require_evaluation_score(
                            ablated_row,
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
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=full_score - ablated_score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=sorted(active_agents),
                                removed_agents=[agent],
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=ablated_score,
                            sampling_seed=int(seed),
                            metadata={
                                "condition_id": condition["condition_id"],
                                "condition": condition["condition"],
                                "role_swap_id": condition["role_swap_id"],
                                "swap": condition["swap"],
                                "position_role": agent,
                                "functional_role": role_map.get(agent, agent),
                                "agent_role_map": role_map,
                                "full_coalition_id": full_row.get("coalition_id"),
                                "ablated_coalition_id": ablated_row.get("coalition_id"),
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=loo task_index={task_index}/{len(tasks)} "
                            f"architecture={architecture_id} condition={condition['condition_id']} "
                            f"position={agent} functional={role_map.get(agent, agent)} score={record.score}"
                        )

                if "shapley_sampled" in methods:
                    if not pending_shapley:
                        print_progress(
                            f"[skip] {completed}/{total} method=shapley_sampled architecture={architecture_id} "
                            f"condition={condition['condition_id']} seed={seed} all roles already done"
                        )
                        continue

                    rng = random.Random(
                        stable_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "shapley_sampled",
                        )
                    )
                    marginals_by_agent: dict[str, list[float]] = defaultdict(list)
                    example_coalition_by_agent: dict[str, list[str]] = {}
                    example_permutation_by_agent: dict[str, list[str]] = {}

                    for sample_index in range(shapley_samples):
                        permutation = list(roles)
                        rng.shuffle(permutation)

                        active: set[str] = set()
                        prev_score = require_evaluation_score(
                            evaluate_coalition(task, architecture_id, int(seed), roles, active, condition),
                            dataset=task.get("dataset"),
                            task_id=task.get("task_id"),
                        )

                        for agent in permutation:
                            before = set(active)
                            active.add(agent)
                            current_score = require_evaluation_score(
                                evaluate_coalition(task, architecture_id, int(seed), roles, active, condition),
                                dataset=task.get("dataset"),
                                task_id=task.get("task_id"),
                            )
                            marginal = current_score - prev_score
                            marginals_by_agent[agent].append(marginal)
                            example_coalition_by_agent.setdefault(agent, sorted(before))
                            example_permutation_by_agent.setdefault(agent, permutation[:])
                            prev_score = current_score

                        print_progress(
                            f"[sample] method=shapley_sampled task={task['task_id']} architecture={architecture_id} "
                            f"condition={condition['condition_id']} seed={seed} sample={sample_index + 1}/{shapley_samples}"
                        )

                    for agent in roles:
                        attribution_id = _condition_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "shapley_sampled",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(
                                f"[skip] {completed}/{total} method=shapley_sampled architecture={architecture_id} "
                                f"condition={condition['condition_id']} agent={agent}"
                            )
                            continue

                        values = marginals_by_agent.get(agent, [])
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
                            method=AttributionMethod("shapley_sampled"),
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=_mean(values),
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=active_example,
                                removed_agents=removed_example,
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=None,
                            sampling_seed=int(seed),
                            num_samples=len(values),
                            permutation_order=example_permutation_by_agent.get(agent),
                            standard_error=_standard_error(values),
                            metadata={
                                "condition_id": condition["condition_id"],
                                "condition": condition["condition"],
                                "role_swap_id": condition["role_swap_id"],
                                "swap": condition["swap"],
                                "position_role": agent,
                                "functional_role": role_map.get(agent, agent),
                                "agent_role_map": role_map,
                                "marginal_values": values,
                                "shapley_samples": shapley_samples,
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=shapley_sampled task_index={task_index}/{len(tasks)} "
                            f"architecture={architecture_id} condition={condition['condition_id']} "
                            f"position={agent} functional={role_map.get(agent, agent)} score={record.score}"
                        )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "coalitions": written_coalitions,
        "coalition_cache_hits": coalition_cache_hits,
        "coalition_cache_entries": len(coalition_cache),
        "architectures": architectures,
        "conditions": len(architecture_conditions),
        "methods": methods,
        "shapley_samples": shapley_samples if "shapley_sampled" in methods else 0,
        "attribution_file": str(attribution_path),
        "coalition_file": str(coalition_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "statistics_file": str(statistics_path),
        "checkpointing": _use_checkpointing(),
        "execution_fingerprint": fingerprint,
    }
    print_progress(f"[complete] {summary}")
    return summary


def _permission_value_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(" ", "_")


def _permission_toggle_names(intervention: dict[str, Any]) -> list[str]:
    names: list[str] = []
    toggle = intervention.get("toggle")
    if toggle not in (None, ""):
        names.append(str(toggle))
    names.extend(str(item) for item in intervention.get("toggles", []) or [])
    return names


def _validate_configured_permission_interventions(interventions: list[dict[str, Any]]) -> None:
    for intervention in interventions:
        explicit_overrides = intervention.get("permission_overrides")
        if explicit_overrides is not None:
            if not isinstance(explicit_overrides, dict):
                raise ValueError(
                    f"Permission condition {intervention.get('id')} permission_overrides must be a mapping."
                )
            for role, overrides in explicit_overrides.items():
                if not isinstance(overrides, dict):
                    raise ValueError(
                        f"Permission condition {intervention.get('id')} overrides for {role} must be a mapping."
                    )
                validate_permission_toggles(
                    [str(name) for name in overrides],
                    context=f"permission condition {intervention.get('id')} role {role}",
                )
            continue
        validate_permission_toggles(
            _permission_toggle_names(intervention),
            context=f"permission intervention {intervention.get('id')}",
        )


def _permission_level_overrides(intervention: dict[str, Any], level: Any) -> dict[str, bool]:
    toggle = intervention.get("toggle")
    toggles = [str(item) for item in intervention.get("toggles", [])]
    validate_permission_toggles(
        _permission_toggle_names(intervention),
        context=f"permission intervention {intervention.get('id')}",
    )

    if toggle:
        return {str(toggle): bool(level)}

    if set(toggles) == {"read_memory", "write_memory"}:
        if level == "none":
            return {"read_memory": False, "write_memory": False}
        if level == "read_only":
            return {"read_memory": True, "write_memory": False}
        if level == "read_write":
            return {"read_memory": True, "write_memory": True}

    if toggles:
        if isinstance(level, dict):
            return {str(k): bool(v) for k, v in level.items() if str(k) in set(toggles)}
        enabled = bool(level)
        return {name: enabled for name in toggles}

    raise ValueError(f"Permission intervention {intervention.get('id')} has neither toggle nor toggles.")


def _permission_conditions(
    architecture_id: str,
    roles: list[str],
    interventions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    role_set = set(roles)
    conditions: list[dict[str, Any]] = []

    for intervention in interventions:
        intervention_id = str(intervention["id"])
        explicit_overrides = intervention.get("permission_overrides")
        if explicit_overrides is not None:
            overrides = {
                str(role): {str(name): bool(value) for name, value in values.items()}
                for role, values in explicit_overrides.items()
            }
            affected_roles = sorted(overrides)
            absent_roles = sorted(set(affected_roles) - role_set)
            if absent_roles:
                print_progress(
                    f"[skip-permission] architecture={architecture_id} intervention={intervention_id} "
                    f"roles={absent_roles} reason=role_absent"
                )
                continue
            conditions.append(
                {
                    "condition_id": f"{architecture_id}__{intervention_id}",
                    "permission_intervention_id": intervention_id,
                    "intervention_type": str(intervention.get("intervention_type", "configured")),
                    "target_role": intervention.get("target_role"),
                    "source_role": intervention.get("source_role"),
                    "destination_role": intervention.get("destination_role"),
                    "target_roles": affected_roles,
                    "level": intervention.get("level", "configured"),
                    "level_label": str(intervention.get("level_label", "configured")),
                    "toggle": intervention.get("toggle"),
                    "toggles": list(intervention.get("toggles", [])),
                    "permission_overrides": overrides,
                    "description": intervention.get("description"),
                    "controls": ["same_topology", "same_roles", "same_prompts", "same_model", "same_task"],
                }
            )
            continue

        target_role = str(intervention["role"])
        levels = list(intervention.get("levels", []))
        if target_role not in role_set:
            print_progress(
                f"[skip-permission] architecture={architecture_id} intervention={intervention_id} "
                f"role={target_role} reason=role_absent"
            )
            continue
        if not levels:
            raise ValueError(f"Permission intervention {intervention_id} must define levels.")

        for level in levels:
            overrides = _permission_level_overrides(intervention, level)
            level_label = _permission_value_label(level)
            conditions.append(
                {
                    "condition_id": f"{architecture_id}__{intervention_id}__{level_label}",
                    "permission_intervention_id": intervention_id,
                    "intervention_type": "toggle",
                    "target_role": target_role,
                    "source_role": target_role,
                    "destination_role": None,
                    "target_roles": [target_role],
                    "level": level,
                    "level_label": level_label,
                    "toggle": intervention.get("toggle"),
                    "toggles": list(intervention.get("toggles", [])),
                    "permission_overrides": {target_role: overrides},
                    "controls": ["same_topology", "same_role", "same_model", "same_task"],
                }
            )

    return conditions


def _permission_condition_metadata(condition: dict[str, Any]) -> dict[str, Any]:
    """Keep transfer and revocation provenance with every coalition and attribution row."""
    return {
        "intervention_type": condition.get("intervention_type", "toggle"),
        "source_role": condition.get("source_role"),
        "destination_role": condition.get("destination_role"),
        "target_roles": list(condition.get("target_roles", [])),
        "description": condition.get("description"),
    }


def _permission_feature_rows(
    experiment: Any,
    architectures: list[str],
    interventions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture_id in architectures:
        architecture = experiment.benchmark.architectures.get(architecture_id)
        if architecture is None:
            rows.append({"architecture_id": architecture_id, "status": "missing_architecture"})
            continue

        roles = list(architecture.roles)
        role_set = set(roles)
        for intervention in interventions:
            explicit_overrides = intervention.get("permission_overrides")
            if explicit_overrides is not None:
                affected_roles = sorted(str(role) for role in explicit_overrides)
                applicable = set(affected_roles).issubset(role_set)
                rows.append(
                    {
                        "architecture_id": architecture_id,
                        "permission_intervention_id": intervention.get("id"),
                        "intervention_type": intervention.get("intervention_type", "configured"),
                        "source_role": intervention.get("source_role"),
                        "destination_role": intervention.get("destination_role"),
                        "target_roles": affected_roles,
                        "toggle": intervention.get("toggle"),
                        "toggles": list(intervention.get("toggles", [])),
                        "permission_overrides": explicit_overrides,
                        "applicable": applicable,
                        "available_roles": roles,
                        "design": "runtime_permission_revocation_or_transfer",
                        "roles_fixed": True,
                        "topology_fixed": True,
                        "prompts_fixed": True,
                        "model_fixed": True,
                    }
                )
                continue

            target_role = str(intervention.get("role"))
            levels = list(intervention.get("levels", []))
            applicable = target_role in role_set
            rows.append(
                {
                    "architecture_id": architecture_id,
                    "permission_intervention_id": intervention.get("id"),
                    "target_role": target_role,
                    "toggle": intervention.get("toggle"),
                    "toggles": list(intervention.get("toggles", [])),
                    "levels": levels,
                    "applicable": applicable,
                    "available_roles": roles,
                    "design": "permission_only_intervention",
                    "roles_fixed": True,
                    "topology_fixed": True,
                    "model_fixed": True,
                    "permission_changed_only_for_target_role": applicable,
                }
            )
    return rows


def _permission_map_key(permission_overrides: dict[str, dict[str, bool]]) -> list[tuple[str, list[tuple[str, bool]]]]:
    return sorted(
        (str(role), sorted((str(name), bool(value)) for name, value in overrides.items()))
        for role, overrides in permission_overrides.items()
    )


def _permission_coalition_key(
    experiment_id: str,
    task_id: str,
    architecture_id: str,
    seed: int,
    active_agents: set[str] | list[str] | tuple[str, ...],
    condition_id: str,
    permission_overrides: dict[str, dict[str, bool]],
    fingerprint: str,
) -> str:
    return stable_id(
        experiment_id,
        task_id,
        architecture_id,
        seed,
        "permission_coalition",
        condition_id,
        sorted(active_agents),
        _permission_map_key(permission_overrides),
        fingerprint,
    )


def _run_permission_diagnostics(run: Any) -> dict[str, Any]:
    metadata = getattr(run, "metadata", None) or {}
    return dict(metadata.get("permission_diagnostics") or {})


def run_permission_intervention(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    """Run permission-only interventions.

    Roles, topology, prompts, model, task, and seed stay fixed. Each condition
    toggles one target role's permission set, then attribution is reported over
    graph-position roles under that permission condition.
    """

    experiment = load_experiment(config_path)
    permission_interventions = list(experiment.raw.get("permission_interventions", []))
    permission_conditions = list(experiment.raw.get("permission_conditions", []))
    permission_specs = permission_interventions + permission_conditions
    if not permission_specs:
        raise ValueError("exp07 requires permission_interventions or permission_conditions.")
    _validate_configured_permission_interventions(permission_specs)
    attribution_cfg = experiment.raw.get("attribution", {})
    methods = _intervention_methods(experiment)
    protocol = str(attribution_cfg.get("removal_protocol", "null_agent_replacement"))
    fingerprint = execution_fingerprint(experiment, removal_protocol=protocol)
    treatment = execution_treatment(experiment, removal_protocol=protocol)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    architectures = select_architectures(experiment)
    if not architectures:
        raise ValueError("exp07 requires base_architectures or architectures.include.")

    missing = [architecture_id for architecture_id in architectures if architecture_id not in experiment.benchmark.architectures]
    if missing:
        raise ValueError(f"Unknown architectures in exp07 base_architectures: {missing}")

    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    shapley_samples = _cfg_int(
        attribution_cfg,
        "shapley.num_permutations",
        "shapley.num_samples",
        "num_samples",
        "permutation_samples",
        default=8,
    )

    outputs = experiment.raw.get("outputs", {})
    root = experiment.benchmark.project_root

    run_path = root / f"{outputs.get('run_dir', 'data/runs/interventions/permission')}/runs.jsonl"
    trace_path = root / f"{outputs.get('trace_dir', 'data/traces/interventions/permission')}/{experiment.experiment_id}_traces.jsonl"
    attribution_path = root / outputs.get(
        "attribution_file",
        "data/results/attribution/permission_intervention_attribution.jsonl",
    )
    coalition_path = root / outputs.get(
        "coalition_file",
        "data/results/attribution/permission_intervention_coalitions.jsonl",
    )
    evaluation_path = root / outputs.get(
        "score_file",
        f"data/results/scores/{experiment.experiment_id}_scores.jsonl",
    )
    statistics_path = root / outputs.get(
        "statistics_file",
        "data/results/statistics/permission_intervention.jsonl",
    )

    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, attribution_path, coalition_path, evaluation_path, statistics_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done_attr = _completed_attribution_ids(attribution_path) if _use_checkpointing() else set()
    coalition_cache = _load_coalition_cache(coalition_path) if _use_checkpointing() else {}

    architecture_conditions: list[tuple[str, list[str], dict[str, Any]]] = []
    for architecture_id in architectures:
        roles = list(experiment.benchmark.architectures[architecture_id].roles)
        for condition in _permission_conditions(architecture_id, roles, permission_specs):
            architecture_conditions.append((architecture_id, roles, condition))

    if not architecture_conditions:
        raise ValueError("No applicable permission intervention conditions for the selected base_architectures.")

    total = len(tasks) * len(seeds) * sum(
        len(roles) * len(methods)
        for _, roles, _ in architecture_conditions
    )
    completed = len(done_attr)

    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    written_attribution = 0
    written_coalitions = 0
    coalition_cache_hits = 0

    print_progress(
        f"[start] experiment={experiment.experiment_id} architectures={architectures} "
        f"conditions={len(architecture_conditions)} methods={methods} "
        f"total_attributions={total} already_done={completed}"
    )

    feature_rows = _permission_feature_rows(experiment, architectures, permission_specs)
    if feature_rows and (fresh or not statistics_path.exists()):
        append_jsonl(statistics_path, feature_rows)

    def evaluate_coalition(
        task: dict[str, Any],
        architecture_id: str,
        seed: int,
        roles: list[str],
        active_agents: set[str],
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal written_runs, written_traces, written_evaluations, written_coalitions, coalition_cache_hits

        active_agents = set(active_agents)
        removed_agents = set(roles) - active_agents
        permission_overrides = {
            str(role): {str(name): bool(value) for name, value in overrides.items()}
            for role, overrides in condition["permission_overrides"].items()
        }
        coalition_id = _permission_coalition_key(
            experiment.experiment_id,
            str(task["task_id"]),
            architecture_id,
            seed,
            active_agents,
            str(condition["condition_id"]),
            permission_overrides,
            fingerprint,
        )

        cached = coalition_cache.get(coalition_id)
        if cached is not None:
            coalition_cache_hits += 1
            require_evaluation_score(
                cached,
                dataset=task.get("dataset") or cached.get("dataset"),
                task_id=task.get("task_id") or cached.get("task_id"),
            )
            return cached

        print_progress(
            f"[coalition] task={task['task_id']} architecture={architecture_id} seed={seed} "
            f"condition={condition['condition_id']} active={sorted(active_agents)} "
            f"removed={sorted(removed_agents)} permission_overrides={permission_overrides}"
        )
        run, traces, evaluation = run_mas_once(
            experiment,
            task,
            architecture_id,
            int(seed),
            removed_agents=removed_agents,
            removal_protocol=protocol,
            permission_overrides=permission_overrides,
            condition_id=str(condition["condition_id"]),
        )

        row = {
            "coalition_id": coalition_id,
            "experiment_id": experiment.experiment_id,
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "architecture_id": architecture_id,
            "sampling_seed": int(seed),
            "condition_id": condition["condition_id"],
            "permission_intervention_id": condition["permission_intervention_id"],
            **_permission_condition_metadata(condition),
            "target_role": condition["target_role"],
            "level": condition["level"],
            "level_label": condition["level_label"],
            "toggle": condition.get("toggle"),
            "toggles": condition.get("toggles", []),
            "permission_overrides": permission_overrides,
            "active_agents": sorted(active_agents),
            "removed_agents": sorted(removed_agents),
            "score": require_evaluation_score(
                evaluation,
                dataset=task.get("dataset"),
                task_id=task.get("task_id"),
            ),
            "run_id": run.run_id,
            "passed": getattr(evaluation, "passed", None),
            "failure_type": getattr(evaluation, "failure_type", None),
            "permission_diagnostics": _run_permission_diagnostics(run),
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        }
        coalition_cache[coalition_id] = row

        append_jsonl(run_path, [run])
        append_jsonl(trace_path, traces)
        append_jsonl(evaluation_path, [evaluation])
        append_jsonl(coalition_path, [row])

        written_runs += 1
        written_traces += len(traces)
        written_evaluations += 1
        written_coalitions += 1
        return row

    for task_index, task in enumerate(tasks, start=1):
        for architecture_id, roles, condition in architecture_conditions:
            role_set = set(roles)

            for seed in seeds:
                pending_loo = [
                    agent
                    for agent in roles
                    if _condition_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        condition["condition_id"],
                        "loo",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                pending_shapley = [
                    agent
                    for agent in roles
                    if _condition_attribution_id(
                        experiment.experiment_id,
                        task["task_id"],
                        architecture_id,
                        seed,
                        condition["condition_id"],
                        "shapley_sampled",
                        agent,
                        fingerprint,
                    )
                    not in done_attr
                ]
                if "loo" not in methods:
                    pending_loo = []
                if "shapley_sampled" not in methods:
                    pending_shapley = []
                if not pending_loo and not pending_shapley:
                    completed_for_group = len(roles) * len(methods)
                    print_progress(
                        f"[skip-group] task_index={task_index}/{len(tasks)} architecture={architecture_id} "
                        f"condition={condition['condition_id']} seed={seed} "
                        f"completed_attributions={completed_for_group}"
                    )
                    continue

                full_row = evaluate_coalition(task, architecture_id, int(seed), roles, role_set, condition)
                full_score = require_evaluation_score(
                    full_row,
                    dataset=task.get("dataset"),
                    task_id=task.get("task_id"),
                )

                if "loo" in methods:
                    for agent in roles:
                        attribution_id = _condition_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "loo",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(
                                f"[skip] {completed}/{total} method=loo architecture={architecture_id} "
                                f"condition={condition['condition_id']} agent={agent}"
                            )
                            continue

                        active_agents = role_set - {agent}
                        ablated_row = evaluate_coalition(
                            task,
                            architecture_id,
                            int(seed),
                            roles,
                            active_agents,
                            condition,
                        )
                        ablated_score = require_evaluation_score(
                            ablated_row,
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
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=full_score - ablated_score,
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=sorted(active_agents),
                                removed_agents=[agent],
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=ablated_score,
                            sampling_seed=int(seed),
                            metadata={
                                "condition_id": condition["condition_id"],
                                "permission_intervention_id": condition["permission_intervention_id"],
                                **_permission_condition_metadata(condition),
                                "target_role": condition["target_role"],
                                "level": condition["level"],
                                "level_label": condition["level_label"],
                                "toggle": condition.get("toggle"),
                                "toggles": condition.get("toggles", []),
                                "permission_overrides": condition["permission_overrides"],
                                "attributed_role_is_target": agent == condition["target_role"],
                                "full_coalition_id": full_row.get("coalition_id"),
                                "ablated_coalition_id": ablated_row.get("coalition_id"),
                                "full_permission_diagnostics": full_row.get("permission_diagnostics", {}),
                                "ablated_permission_diagnostics": ablated_row.get("permission_diagnostics", {}),
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=loo task_index={task_index}/{len(tasks)} "
                            f"architecture={architecture_id} condition={condition['condition_id']} "
                            f"role={agent} target={condition['target_role']} level={condition['level_label']} "
                            f"score={record.score}"
                        )

                if "shapley_sampled" in methods:
                    if not pending_shapley:
                        print_progress(
                            f"[skip] {completed}/{total} method=shapley_sampled architecture={architecture_id} "
                            f"condition={condition['condition_id']} seed={seed} all roles already done"
                        )
                        continue

                    rng = random.Random(
                        stable_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "shapley_sampled",
                        )
                    )
                    marginals_by_agent: dict[str, list[float]] = defaultdict(list)
                    example_coalition_by_agent: dict[str, list[str]] = {}
                    example_permutation_by_agent: dict[str, list[str]] = {}

                    for sample_index in range(shapley_samples):
                        permutation = list(roles)
                        rng.shuffle(permutation)

                        active: set[str] = set()
                        prev_score = require_evaluation_score(
                            evaluate_coalition(task, architecture_id, int(seed), roles, active, condition),
                            dataset=task.get("dataset"),
                            task_id=task.get("task_id"),
                        )

                        for agent in permutation:
                            before = set(active)
                            active.add(agent)
                            current_score = require_evaluation_score(
                                evaluate_coalition(task, architecture_id, int(seed), roles, active, condition),
                                dataset=task.get("dataset"),
                                task_id=task.get("task_id"),
                            )
                            marginal = current_score - prev_score
                            marginals_by_agent[agent].append(marginal)
                            example_coalition_by_agent.setdefault(agent, sorted(before))
                            example_permutation_by_agent.setdefault(agent, permutation[:])
                            prev_score = current_score

                        print_progress(
                            f"[sample] method=shapley_sampled task={task['task_id']} architecture={architecture_id} "
                            f"condition={condition['condition_id']} seed={seed} "
                            f"sample={sample_index + 1}/{shapley_samples}"
                        )

                    for agent in roles:
                        attribution_id = _condition_attribution_id(
                            experiment.experiment_id,
                            task["task_id"],
                            architecture_id,
                            seed,
                            condition["condition_id"],
                            "shapley_sampled",
                            agent,
                            fingerprint,
                        )
                        if attribution_id in done_attr:
                            print_progress(
                                f"[skip] {completed}/{total} method=shapley_sampled architecture={architecture_id} "
                                f"condition={condition['condition_id']} agent={agent}"
                            )
                            continue

                        values = marginals_by_agent.get(agent, [])
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
                            method=AttributionMethod("shapley_sampled"),
                            utility_type=attribution_cfg.get("utility", "task"),
                            score=_mean(values),
                            baseline_score=0.0,
                            coalition=CoalitionInfo(
                                active_agents=active_example,
                                removed_agents=removed_example,
                            ),
                            removal_protocol=RemovalProtocol(protocol),
                            full_team_score=full_score,
                            ablated_score=None,
                            sampling_seed=int(seed),
                            num_samples=len(values),
                            permutation_order=example_permutation_by_agent.get(agent),
                            standard_error=_standard_error(values),
                            metadata={
                                "condition_id": condition["condition_id"],
                                "permission_intervention_id": condition["permission_intervention_id"],
                                **_permission_condition_metadata(condition),
                                "target_role": condition["target_role"],
                                "level": condition["level"],
                                "level_label": condition["level_label"],
                                "toggle": condition.get("toggle"),
                                "toggles": condition.get("toggles", []),
                                "permission_overrides": condition["permission_overrides"],
                                "attributed_role_is_target": agent == condition["target_role"],
                                "marginal_values": values,
                                "shapley_samples": shapley_samples,
                                "full_coalition_id": full_row.get("coalition_id"),
                                "full_permission_diagnostics": full_row.get("permission_diagnostics", {}),
                                "execution_fingerprint": fingerprint,
                                "execution_treatment": treatment,
                            },
                        )
                        append_jsonl(attribution_path, [record])
                        done_attr.add(record.attribution_id)
                        completed += 1
                        written_attribution += 1
                        print_progress(
                            f"[done] {completed}/{total} method=shapley_sampled task_index={task_index}/{len(tasks)} "
                            f"architecture={architecture_id} condition={condition['condition_id']} "
                            f"role={agent} target={condition['target_role']} level={condition['level_label']} "
                            f"score={record.score}"
                        )

    summary = {
        "records": completed,
        "new_records": written_attribution,
        "runs": written_runs,
        "traces": written_traces,
        "evaluations": written_evaluations,
        "coalitions": written_coalitions,
        "coalition_cache_hits": coalition_cache_hits,
        "coalition_cache_entries": len(coalition_cache),
        "architectures": architectures,
        "conditions": len(architecture_conditions),
        "methods": methods,
        "shapley_samples": shapley_samples if "shapley_sampled" in methods else 0,
        "attribution_file": str(attribution_path),
        "coalition_file": str(coalition_path),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(evaluation_path),
        "statistics_file": str(statistics_path),
        "checkpointing": _use_checkpointing(),
        "execution_fingerprint": fingerprint,
    }
    print_progress(f"[complete] {summary}")
    return summary


def run_intervention(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    if "topology_variants" in experiment.raw:
        return run_topology_intervention(config_path, max_tasks=max_tasks)
    if "role_swaps" in experiment.raw:
        return run_role_intervention(config_path, max_tasks=max_tasks)
    if "permission_interventions" in experiment.raw or "permission_conditions" in experiment.raw:
        return run_permission_intervention(config_path, max_tasks=max_tasks)

    raise NotImplementedError(
        f"No intervention runner implemented for {experiment.experiment_id}. "
        "Currently supported: topology_variants, role_swaps, permission_interventions, permission_conditions."
    )
