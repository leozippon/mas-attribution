"""Run single-agent, solo-role, and random-team baseline experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import random
from pathlib import Path
from typing import Any

from mas_contribution_bench.agents import build_agents
from mas_contribution_bench.data.schemas import (
    CoalitionInfo,
    RemovalInfo,
    RemovalProtocol,
    RunRecord,
    RunStatus,
)
from mas_contribution_bench.evaluation import evaluate_task_output
from mas_contribution_bench.runners.common import (
    backup_existing_file,
    build_model_client,
    completed_run_ids,
    execution_fingerprint,
    execution_treatment,
    identity_role_map,
    load_experiment,
    mas_run_id,
    print_progress,
    sandbox_backend,
    should_execute_code,
    select_tasks,
)
from mas_contribution_bench.tracing import build_trace_records, trace_cost
from mas_contribution_bench.utils.io import append_jsonl, stable_id
from mas_contribution_bench.utils.seeds import set_seed


def _use_checkpointing() -> bool:
    return os.getenv("MAS_DISABLE_CHECKPOINT", "").lower() not in {"1", "true", "yes", "y"}


def _final_role(roles: list[str]) -> str | None:
    for preferred in ("finalizer", "aggregator", "supervisor", "verifier", "coder", "executor"):
        if preferred in roles:
            return preferred
    return roles[-1] if roles else None


def _invoke_roles(
    experiment,
    task: dict[str, Any],
    roles: list[str],
    seed: int | None = None,
) -> tuple[dict[str, Any], str]:
    model_overrides = dict(experiment.raw.get("model", {}) or {})
    if seed is not None:
        model_overrides["seed"] = int(seed)
    agents = build_agents(
        experiment.benchmark.agents,
        roles,
        model_client=build_model_client(experiment),
        model_overrides=model_overrides,
    )
    state: dict[str, Any] = {"task": task, "messages": [], "agent_outputs": {}}
    if seed is not None:
        state["seed"] = int(seed)
    for role in roles:
        output = agents[role].invoke(state)
        payload = {
            "agent_id": output.agent_id,
            "role": output.role,
            "content": output.content,
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "tool_calls": output.tool_calls,
            "metadata": output.metadata or {},
        }
        state["agent_outputs"][role] = payload
        state["messages"].append({"sender": role, "receiver": "next", "content": output.content})
    final_role = _final_role(roles)
    final_answer = state["agent_outputs"][final_role]["content"] if final_role else ""
    state["final_answer"] = final_answer
    return state, final_answer


def _baseline_specs(experiment) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    agent_roles = sorted(experiment.benchmark.agents)
    for baseline in experiment.raw.get("baselines", []):
        baseline_id = baseline["id"]
        baseline_type = baseline.get("type")
        if baseline_type == "single_agent":
            role = baseline["agent"]
            specs.append({"id": baseline_id, "type": baseline_type, "roles": [role], "sample": 0})
        elif baseline_type == "solo_role":
            for role in baseline.get("roles", []):
                specs.append({"id": f"{baseline_id}_{role}", "type": baseline_type, "roles": [role], "sample": 0})
        elif baseline_type == "random_team":
            team_size = int(baseline.get("team_size", 5))
            num_samples = int(baseline.get("num_samples_per_task", 1))
            for sample in range(num_samples):
                specs.append(
                    {
                        "id": f"{baseline_id}_{sample}",
                        "type": baseline_type,
                        "roles": agent_roles,
                        "team_size": team_size,
                        "sample": sample,
                    }
                )
        else:
            print_progress(f"[skip_baseline] unsupported baseline={baseline_id} type={baseline_type}")
    return specs


def _baseline_run_id(experiment, task: dict[str, Any], spec: dict[str, Any], seed: int, roles: list[str]) -> str:
    return mas_run_id(
        experiment,
        task["task_id"],
        spec["id"],
        int(seed),
        removed_agents=[],
        removal_protocol="none",
        role_map_items=sorted(identity_role_map(roles).items()),
        permission_overrides=None,
        condition_id="baseline",
    )


def _roles_for_spec(spec: dict[str, Any], task_id: str, seed: int) -> list[str]:
    roles = list(spec["roles"])
    if spec.get("type") != "random_team":
        return roles
    rng = random.Random(stable_id(task_id, seed, spec.get("id"), spec.get("sample")))
    team_size = min(int(spec.get("team_size", 5)), len(roles))
    sampled = rng.sample(roles, team_size)
    if "finalizer" in roles and "finalizer" not in sampled:
        sampled[-1] = "finalizer"
    return sampled


def _run_baseline_once(experiment, task: dict[str, Any], spec: dict[str, Any], seed: int):
    roles = _roles_for_spec(spec, task["task_id"], seed)
    baseline_id = spec["id"]
    set_seed(seed)
    treatment = execution_treatment(experiment, removal_protocol="none")
    fingerprint = execution_fingerprint(experiment, removal_protocol="none")
    run_id = _baseline_run_id(experiment, task, spec, seed, roles)
    started = datetime.now(timezone.utc)
    state, final_answer = _invoke_roles(experiment, task, roles, seed=seed)
    traces = build_trace_records(run_id, task["task_id"], state)
    cost = trace_cost(traces)
    evaluation = evaluate_task_output(
        run_id=run_id,
        task=task,
        architecture_id=baseline_id,
        final_answer=final_answer,
        cost=cost,
        execute_code=should_execute_code(experiment),
        sandbox_backend=sandbox_backend(experiment),
    )
    ended = datetime.now(timezone.utc)
    run = RunRecord(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        task_id=task["task_id"],
        dataset=task["dataset"],
        architecture_id=baseline_id,
        seed=seed,
        coalition=CoalitionInfo(active_agents=roles, removed_agents=[]),
        removal=RemovalInfo(protocol=RemovalProtocol.NONE, removed_agents=[]),
        config_hash=experiment.config_hash,
        started_at=started,
        ended_at=ended,
        status=RunStatus.SUCCEEDED,
        cost=cost,
        failure_type=evaluation.failure_type,
        metadata={
            "baseline_type": spec.get("type"),
            "baseline_id": baseline_id,
            "roles": roles,
            "sample": spec.get("sample"),
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        },
    )
    evaluation.metadata.update(
        {
            "baseline_type": spec.get("type"),
            "roles": roles,
            "sample": spec.get("sample"),
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        }
    )
    return run, traces, evaluation


def run_single_agent_baseline(config_path: str | Path, max_tasks: int | None = None) -> dict[str, Any]:
    experiment = load_experiment(config_path)
    tasks = select_tasks(experiment)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    seeds = [int(seed) for seed in experiment.raw.get("seeds", [0])]
    specs = _baseline_specs(experiment)
    outputs = experiment.raw.get("outputs", {})
    run_file = f"{outputs.get('run_dir', 'data/runs/single_agent')}/runs.jsonl"
    trace_file = f"{outputs.get('trace_dir', 'data/traces/baselines')}/{experiment.experiment_id}_traces.jsonl"
    evaluation_file = outputs.get("score_file", f"data/results/scores/{experiment.experiment_id}.jsonl")

    root = experiment.benchmark.project_root
    run_path = root / run_file
    trace_path = root / trace_file
    eval_path = root / evaluation_file
    fresh = os.getenv("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}
    if _use_checkpointing() and fresh:
        for file_path in (run_path, trace_path, eval_path):
            backup = backup_existing_file(file_path)
            if backup:
                print_progress(f"[backup] {file_path} -> {backup}")
            file_path.unlink(missing_ok=True)

    done = completed_run_ids(run_path) if _use_checkpointing() else set()
    total = len(tasks) * len(seeds) * len(specs)
    completed = len(done)
    written_runs = 0
    written_traces = 0
    written_evaluations = 0
    print_progress(f"[start] experiment={experiment.experiment_id} total_runs={total} already_done={completed}")
    for task_index, task in enumerate(tasks, start=1):
        for spec in specs:
            for seed in seeds:
                roles = _roles_for_spec(spec, task["task_id"], seed)
                run_id = _baseline_run_id(experiment, task, spec, seed, roles)
                label = f"task={task.get('task_id')} baseline={spec['id']} seed={seed} roles={','.join(roles)}"
                if run_id in done:
                    print_progress(f"[skip] {completed}/{total} {label} run_id={run_id}")
                    continue
                print_progress(f"[run] {completed + 1}/{total} task_index={task_index}/{len(tasks)} {label}")
                run, traces, evaluation = _run_baseline_once(experiment, task, spec, seed)
                append_jsonl(run_path, [run])
                append_jsonl(trace_path, traces)
                append_jsonl(eval_path, [evaluation])
                done.add(run.run_id)
                completed += 1
                written_runs += 1
                written_traces += len(traces)
                written_evaluations += 1
                print_progress(
                    f"[done] {completed}/{total} run_id={run.run_id} score={evaluation.score} "
                    f"passed={evaluation.passed} failure={evaluation.failure_type} traces={len(traces)}"
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
        "checkpointing": _use_checkpointing(),
    }
    print_progress(f"[complete] {summary}")
    return summary
