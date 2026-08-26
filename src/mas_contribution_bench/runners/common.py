"""Shared runner primitives."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any

from mas_contribution_bench.agents import DeepSeekModelClient, DryRunModelClient, OpenAICompatibleModelClient, build_agents
from mas_contribution_bench.config import ExperimentSpec, load_experiment_spec
from mas_contribution_bench.data.loaders import load_jsonl_tasks
from mas_contribution_bench.data.schemas import (
    CoalitionInfo,
    RemovalInfo,
    RemovalProtocol,
    RunRecord,
    RunStatus,
)
from mas_contribution_bench.evaluation import evaluate_task_output
from mas_contribution_bench.graphs import MASGraphBuilder
from mas_contribution_bench.tracing import build_trace_records, trace_cost
from mas_contribution_bench.utils.io import append_jsonl, ensure_dir, iter_jsonl, stable_hash, stable_id, write_jsonl
from mas_contribution_bench.utils.seeds import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DRY_RUN_BACKENDS = frozenset({"dry-run", "dry_run", "dryrun", "mock"})
OPENAI_COMPATIBLE_BACKENDS = frozenset(
    {"vllm", "qwen", "openai", "openai_compatible", "openai-compatible"}
)
WIRED_ATTRIBUTION_METHODS = frozenset({"loo", "shapley_sampled", "banzhaf_sampled"})
WIRED_INTERVENTION_METHODS = frozenset({"loo", "shapley_sampled"})
METHOD_ALIASES = {
    "sampled_shapley": "shapley_sampled",
    "sampled_banzhaf": "banzhaf_sampled",
}


def should_execute_code(experiment: ExperimentSpec) -> bool:
    env_value = os.getenv("MAS_EXECUTE_CODE")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "y"}
    evaluation_cfg = experiment.raw.get("evaluation") or {}
    return bool(evaluation_cfg.get("execute_code", False))



def model_backend(experiment: ExperimentSpec) -> str:
    return (
        os.getenv("MAS_MODEL_BACKEND")
        or (experiment.raw.get("model") or {}).get("backend")
        or "dry-run"
    ).lower()


def resolved_model_name(experiment: ExperimentSpec) -> str:
    backend = model_backend(experiment)
    model_cfg = experiment.raw.get("model") or {}
    if backend in DRY_RUN_BACKENDS:
        return "dry-run"
    if backend == "deepseek":
        return str(
            os.getenv("DEEPSEEK_MODEL") or model_cfg.get("name") or model_cfg.get("model") or "deepseek-chat"
        )
    if backend in OPENAI_COMPATIBLE_BACKENDS:
        return str(
            os.getenv("MODEL_NAME")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("VLLM_MODEL")
            or model_cfg.get("name")
            or model_cfg.get("model")
            or "qwen-local"
        )
    return str(model_cfg.get("name") or model_cfg.get("model") or backend)


def build_model_client(experiment: ExperimentSpec):
    backend = model_backend(experiment)
    if backend in DRY_RUN_BACKENDS:
        return DryRunModelClient()
    model_cfg = experiment.raw.get("model") or {}
    default_model = resolved_model_name(experiment)
    if backend == "deepseek":
        return DeepSeekModelClient(
            default_model=default_model,
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT", model_cfg.get("timeout_seconds", 120))),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", model_cfg.get("max_retries", 3))),
            retry_backoff_seconds=float(os.getenv("DEEPSEEK_RETRY_BACKOFF", model_cfg.get("retry_backoff_seconds", 2.0))),
        )
    if backend in OPENAI_COMPATIBLE_BACKENDS:
        return OpenAICompatibleModelClient(
            default_model=default_model,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL") or model_cfg.get("base_url") or "http://127.0.0.1:8000/v1",
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "dummy",
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT") or os.getenv("VLLM_TIMEOUT") or model_cfg.get("timeout_seconds", 120)),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES") or os.getenv("VLLM_MAX_RETRIES") or model_cfg.get("max_retries", 3)),
            retry_backoff_seconds=float(os.getenv("OPENAI_RETRY_BACKOFF") or os.getenv("VLLM_RETRY_BACKOFF") or model_cfg.get("retry_backoff_seconds", 2.0)),
            provider_name=backend.replace("-", "_"),
        )
    raise ValueError(f"Unsupported MAS_MODEL_BACKEND: {backend}")

def sandbox_backend(experiment: ExperimentSpec) -> str:
    return (
        os.getenv("MAS_SANDBOX_BACKEND")
        or (experiment.raw.get("evaluation") or {}).get("sandbox_backend")
        or "auto"
    )


def execution_treatment(experiment: ExperimentSpec, removal_protocol: str = "none") -> dict[str, Any]:
    """Runtime knobs that can change outcomes but are outside the YAML bundle hash."""

    return {
        "config_hash": experiment.config_hash,
        "model_backend": model_backend(experiment),
        "model_name": resolved_model_name(experiment),
        "execute_code": should_execute_code(experiment),
        "sandbox_backend": sandbox_backend(experiment),
        "removal_protocol": str(removal_protocol or "none"),
    }


def execution_fingerprint(experiment: ExperimentSpec, removal_protocol: str = "none") -> str:
    return stable_hash(execution_treatment(experiment, removal_protocol))


def identity_role_map(
    architecture_roles: list[str],
    agent_role_map: dict[str, str] | None = None,
) -> dict[str, str]:
    return {role: str((agent_role_map or {}).get(role, role)) for role in architecture_roles}


def permission_override_identity(
    permission_overrides: dict[str, dict[str, bool]] | None,
) -> list[tuple[str, list[tuple[str, bool]]]]:
    return sorted(
        (str(role), sorted((str(name), bool(value)) for name, value in overrides.items()))
        for role, overrides in (permission_overrides or {}).items()
    )


def mas_run_id(
    experiment: ExperimentSpec,
    task_id: str,
    architecture_id: str,
    seed: int,
    *,
    removed_agents: set[str] | list[str] | tuple[str, ...] | None = None,
    removal_protocol: str = "none",
    role_map_items: list[tuple[str, str]] | None = None,
    permission_overrides: dict[str, dict[str, bool]] | None = None,
    condition_id: str | None = None,
) -> str:
    fingerprint = execution_fingerprint(experiment, removal_protocol=removal_protocol)
    return stable_id(
        experiment.experiment_id,
        task_id,
        architecture_id,
        seed,
        sorted(removed_agents or []),
        condition_id or "default",
        list(role_map_items or []),
        permission_override_identity(permission_overrides),
        fingerprint,
    )


def normalize_attribution_method(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))


def listed_attribution_methods(
    attribution_cfg: dict[str, Any] | None,
    *,
    default: list[str] | None = None,
) -> list[str]:
    cfg = attribution_cfg or {}
    if "methods" in cfg:
        return [normalize_attribution_method(str(method)) for method in (cfg.get("methods") or [])]
    if cfg.get("method") not in (None, ""):
        return [normalize_attribution_method(str(cfg.get("method")))]
    return list(default or [])


def validate_attribution_methods(
    methods: list[str],
    supported: set[str] | frozenset[str],
    *,
    context: str,
) -> list[str]:
    supported_text = ", ".join(sorted(supported))
    if not methods:
        raise ValueError(
            f"{context} requires at least one attribution method. "
            f"Supported methods: {supported_text}. "
            "Myerson, Owen, and exact Shapley/Banzhaf estimators remain unwired and are rejected."
        )
    unsupported = [method for method in methods if method not in supported]
    if unsupported:
        raise ValueError(
            f"{context} does not support attribution method(s): {', '.join(unsupported)}. "
            f"Supported methods: {supported_text}. "
            "Myerson, Owen, and exact Shapley/Banzhaf estimators remain unwired and are rejected."
        )
    return methods


def permission_enforcement_enabled(experiment: ExperimentSpec, permission_overrides: dict[str, dict[str, bool]] | None) -> bool:
    if not permission_overrides:
        return False
    env_value = os.getenv("MAS_PERMISSION_ENFORCEMENT")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "y", "strict"}
    cfg = experiment.raw.get("permission_enforcement") or {}
    return str(cfg.get("mode", "")).lower() == "strict" or bool(cfg.get("enabled", False))


def load_experiment(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ExperimentSpec:
    return load_experiment_spec(config_path, project_root)


def select_tasks(experiment: ExperimentSpec, seed: int | None = None) -> list[dict[str, Any]]:
    tasks = []
    root = experiment.benchmark.project_root
    for dataset_spec in experiment.raw.get("datasets", []):
        task_file = root / dataset_spec["task_file"]
        loaded = load_jsonl_tasks(task_file)
        split = dataset_spec.get("split")
        if split:
            loaded = [task for task in loaded if task.split == split]
        max_tasks = dataset_spec.get("max_tasks")
        if max_tasks is not None:
            loaded = loaded[: int(max_tasks)]
        tasks.extend(task.model_dump(mode="json") for task in loaded)
    return tasks


def select_architectures(experiment: ExperimentSpec) -> list[str]:
    raw = experiment.raw
    if isinstance(raw.get("architectures"), dict):
        return list(raw["architectures"].get("include", []))
    if raw.get("base_architectures"):
        return list(raw["base_architectures"])
    return []


def run_mas_once(
    experiment: ExperimentSpec,
    task: dict[str, Any],
    architecture_id: str,
    seed: int,
    removed_agents: set[str] | None = None,
    removal_protocol: str = "none",
    agent_role_map: dict[str, str] | None = None,
    permission_overrides: dict[str, dict[str, bool]] | None = None,
    condition_id: str | None = None,
) -> tuple[RunRecord, list[Any], Any]:
    set_seed(seed)
    architecture = experiment.benchmark.architectures[architecture_id]
    active_roles = [role for role in architecture.roles if role not in (removed_agents or set())]
    model_client = build_model_client(experiment)
    role_map = {
        role: str((agent_role_map or {}).get(role, role))
        for role in architecture.roles
    }
    functional_roles = sorted(set(architecture.roles) | set(role_map.values()))
    functional_agents = build_agents(
        experiment.benchmark.agents,
        functional_roles,
        model_client=model_client,
        model_overrides=experiment.raw.get("model", {}),
    )
    agents = {}
    for position_role in architecture.roles:
        functional_role = role_map.get(position_role, position_role)
        if functional_role not in functional_agents:
            raise ValueError(
                f"Cannot build role intervention agent: functional role {functional_role!r} "
                f"for graph position {position_role!r} is not available in benchmark agents."
            )
        source_agent = functional_agents[functional_role]
        position_agent = functional_agents[position_role]
        permissions = dict(position_agent.permissions)
        permissions.update({k: bool(v) for k, v in (permission_overrides or {}).get(position_role, {}).items()})
        if functional_role == position_role:
            agents[position_role] = source_agent.__class__(
                agent_id=position_role,
                role=functional_role,
                prompt=source_agent.prompt,
                permissions=permissions,
                model_client=source_agent.model_client,
                model_kwargs=source_agent.model_kwargs,
            )
            continue
        agents[position_role] = source_agent.__class__(
            agent_id=position_role,
            role=functional_role,
            prompt=source_agent.prompt,
            permissions=permissions,
            model_client=source_agent.model_client,
            model_kwargs=source_agent.model_kwargs,
        )
    null_replacement = removal_protocol == "null_agent_replacement"
    graph = MASGraphBuilder(
        architecture=architecture,
        agents=agents,
        removed_agents=removed_agents or set(),
        null_replacement=null_replacement,
    ).build()
    role_map_items = sorted(role_map.items())
    treatment = execution_treatment(experiment, removal_protocol=removal_protocol)
    fingerprint = execution_fingerprint(experiment, removal_protocol=removal_protocol)
    run_id = mas_run_id(
        experiment,
        task["task_id"],
        architecture_id,
        seed,
        removed_agents=removed_agents,
        removal_protocol=removal_protocol,
        role_map_items=role_map_items,
        permission_overrides=permission_overrides,
        condition_id=condition_id,
    )
    started = datetime.now(timezone.utc)
    strict_permission_enforcement = permission_enforcement_enabled(experiment, permission_overrides)
    result = graph.invoke(
        {
            "task": task,
            "messages": [],
            "respect_final_answer_permission": bool(permission_overrides),
            "strict_permission_enforcement": strict_permission_enforcement,
            "permission_overrides": permission_overrides or {},
        }
    )
    trace_records = build_trace_records(run_id, task["task_id"], result.state)
    cost = trace_cost(trace_records)
    cache_usage = summarize_agent_cache_usage(result.state)
    evaluation = evaluate_task_output(
        run_id=run_id,
        task=task,
        architecture_id=architecture_id,
        final_answer=result.final_answer,
        cost=cost,
        execute_code=should_execute_code(experiment),
        sandbox_backend=sandbox_backend(experiment),
    )
    ended = datetime.now(timezone.utc)
    permission_diagnostics = collect_permission_diagnostics(result.state)
    run = RunRecord(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        task_id=task["task_id"],
        dataset=task["dataset"],
        architecture_id=architecture_id,
        seed=seed,
        coalition=CoalitionInfo(active_agents=active_roles, removed_agents=sorted(removed_agents or [])),
        removal=RemovalInfo(protocol=RemovalProtocol(removal_protocol), removed_agents=sorted(removed_agents or [])),
        config_hash=experiment.config_hash,
        started_at=started,
        ended_at=ended,
        status=RunStatus.SUCCEEDED,
        cost=cost,
        failure_type=evaluation.failure_type,
        metadata={
            "dry_run": model_backend(experiment) in DRY_RUN_BACKENDS,
            "model_backend": model_backend(experiment),
            "model_name": resolved_model_name(experiment),
            "cache_usage": cache_usage,
            "condition_id": condition_id,
            "agent_role_map": role_map,
            "permission_overrides": permission_overrides or {},
            "role_intervention": any(position != functional for position, functional in role_map.items()),
            "permission_intervention": bool(permission_overrides),
            "strict_permission_enforcement": strict_permission_enforcement,
            "permission_diagnostics": permission_diagnostics,
            "execution_fingerprint": fingerprint,
            "execution_treatment": treatment,
        },
    )
    return run, trace_records, evaluation


def collect_permission_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    outputs = state.get("agent_outputs") or {}
    blocked_outputs: dict[str, list[str]] = {}
    blocked_tool_calls: dict[str, list[str]] = {}
    explicit_overrides: dict[str, dict[str, bool]] = {}

    for role, output in outputs.items():
        metadata = output.get("metadata") or {}
        diagnostics = metadata.get("permission_diagnostics") or {}
        if diagnostics.get("blocked_outputs"):
            blocked_outputs[str(role)] = list(diagnostics.get("blocked_outputs") or [])
        if diagnostics.get("blocked_tool_calls"):
            blocked_tool_calls[str(role)] = list(diagnostics.get("blocked_tool_calls") or [])
        if diagnostics.get("explicit_permission_overrides"):
            explicit_overrides[str(role)] = dict(diagnostics.get("explicit_permission_overrides") or {})

    graph_diagnostics = dict(state.get("permission_diagnostics") or {})
    graph_diagnostics.update(
        {
            "blocked_outputs_by_role": blocked_outputs,
            "blocked_tool_calls_by_role": blocked_tool_calls,
            "explicit_permission_overrides_by_role": explicit_overrides,
        }
    )
    return graph_diagnostics


def summarize_agent_cache_usage(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate provider/local cache metadata from agent outputs.

    This is intentionally metadata-only: it does not change scoring, traces, or
    final answers. It lets experiments report how much API work was served from
    the local response cache or DeepSeek context cache.
    """

    outputs = state.get("agent_outputs") or {}
    summary: dict[str, Any] = {
        "agents": 0,
        "local_cache_hits": 0,
        "local_cache_misses": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_cache_hit_tokens": 0,
        "provider_cache_miss_tokens": 0,
    }

    for output in outputs.values():
        metadata = output.get("metadata") or {}
        if metadata.get("null_agent"):
            continue
        usage = metadata.get("model_usage") or {}
        cache = metadata.get("cache") or {}
        summary["agents"] += 1
        if cache.get("local_cache_hit") is True:
            summary["local_cache_hits"] += 1
        elif cache:
            summary["local_cache_misses"] += 1

        summary["prompt_tokens"] += int(usage.get("prompt_tokens") or output.get("input_tokens") or 0)
        summary["completion_tokens"] += int(usage.get("completion_tokens") or output.get("output_tokens") or 0)
        summary["total_tokens"] += int(usage.get("total_tokens") or 0)
        summary["provider_cache_hit_tokens"] += int(
            usage.get("prompt_cache_hit_tokens")
            or cache.get("provider_cache_hit_tokens")
            or 0
        )
        summary["provider_cache_miss_tokens"] += int(
            usage.get("prompt_cache_miss_tokens")
            or cache.get("provider_cache_miss_tokens")
            or 0
        )

    provider_total = summary["provider_cache_hit_tokens"] + summary["provider_cache_miss_tokens"]
    summary["provider_cache_hit_rate"] = (
        summary["provider_cache_hit_tokens"] / provider_total if provider_total else None
    )
    local_total = summary["local_cache_hits"] + summary["local_cache_misses"]
    summary["local_cache_hit_rate"] = summary["local_cache_hits"] / local_total if local_total else None
    return summary



def completed_run_ids(path: str | Path) -> set[str]:
    return {str(row.get("run_id")) for row in iter_jsonl(path) if row.get("run_id")}


def backup_existing_file(path: str | Path, *, enabled: bool = True) -> Path | None:
    path = Path(path)
    if not enabled or not path.exists() or path.stat().st_size == 0:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def print_progress(message: str) -> None:
    print(message, flush=True)

def write_run_bundle(
    experiment: ExperimentSpec,
    runs: list[RunRecord],
    traces: list[Any],
    evaluations: list[Any],
    run_file: str | Path,
    trace_file: str | Path,
    evaluation_file: str | Path,
) -> dict[str, Any]:
    root = experiment.benchmark.project_root
    run_path = root / run_file
    trace_path = root / trace_file
    eval_path = root / evaluation_file
    write_jsonl(run_path, runs)
    write_jsonl(trace_path, traces)
    write_jsonl(eval_path, evaluations)
    return {
        "runs": len(runs),
        "traces": len(traces),
        "evaluations": len(evaluations),
        "run_file": str(run_path),
        "trace_file": str(trace_path),
        "evaluation_file": str(eval_path),
    }
