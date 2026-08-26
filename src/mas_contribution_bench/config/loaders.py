"""Load experiment, architecture, agent, prompt, and permission YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_contribution_bench.utils.io import load_yaml, stable_hash


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LoadedAgentSpec:
    role: str
    canonical_role: str
    description: str
    model: str
    temperature: float
    max_tokens: int
    permissions: dict[str, bool]
    prompt_file: str
    prompt: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LoadedArchitectureSpec:
    architecture_id: str
    name: str
    family: str
    roles: list[str]
    canonical_roles: dict[str, str]
    entrypoint: str
    terminal_nodes: list[str]
    edges: list[tuple[str, str]]
    orchestration: dict[str, Any]
    default_permissions: dict[str, str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkSpec:
    project_root: Path
    architecture_dir: Path
    agent_dir: Path
    prompt_dir: Path
    permission_file: Path
    permissions: dict[str, dict[str, Any]]
    agents: dict[str, LoadedAgentSpec]
    architectures: dict[str, LoadedArchitectureSpec]


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    experiment_id: str
    raw: dict[str, Any]
    benchmark: BenchmarkSpec
    config_hash: str


def _resolve(project_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def _resolve_permission_set(name: str, permission_sets: dict[str, dict[str, Any]]) -> dict[str, bool]:
    seen: set[str] = set()
    current_name = name
    merged: dict[str, bool] = {}
    while current_name:
        if current_name in seen:
            raise ValueError(f"Cyclic permission inheritance: {name}")
        seen.add(current_name)
        spec = permission_sets.get(current_name)
        if spec is None:
            raise KeyError(f"Unknown permission set: {current_name}")
        parent = spec.get("inherit_from")
        values = {k: bool(v) for k, v in spec.items() if k != "inherit_from"}
        merged = {**values, **merged}
        current_name = parent
    return merged


def _load_agent(path: Path, project_root: Path, permission_sets: dict[str, dict[str, Any]]) -> LoadedAgentSpec:
    raw = load_yaml(path)
    role = raw["role"]
    prompt_file = raw["prompt_file"]
    prompt_path = _resolve(project_root, prompt_file)
    permissions_spec = raw.get("permissions") or {}
    inherited = permissions_spec.get("inherit_from", role)
    permissions = _resolve_permission_set(inherited, permission_sets)
    overrides = {k: bool(v) for k, v in permissions_spec.items() if k != "inherit_from"}
    permissions.update(overrides)
    return LoadedAgentSpec(
        role=role,
        canonical_role=raw.get("canonical_role", role),
        description=raw.get("description", ""),
        model=raw.get("model", "${MODEL_NAME}"),
        temperature=float(raw.get("temperature", 0.2)),
        max_tokens=int(raw.get("max_tokens", 2048)),
        permissions=permissions,
        prompt_file=prompt_file,
        prompt=prompt_path.read_text(encoding="utf-8"),
        raw=raw,
    )


def _load_architecture(path: Path) -> LoadedArchitectureSpec:
    raw = load_yaml(path)
    edges = [tuple(edge) for edge in raw.get("edges", [])]
    return LoadedArchitectureSpec(
        architecture_id=raw["id"],
        name=raw.get("name", raw["id"]),
        family=raw.get("family", "unknown"),
        roles=list(raw.get("roles", [])),
        canonical_roles=dict(raw.get("canonical_roles", {})),
        entrypoint=raw.get("entrypoint", raw.get("roles", [""])[0]),
        terminal_nodes=list(raw.get("terminal_nodes", ["final_answer"])),
        edges=edges,
        orchestration=dict(raw.get("orchestration", {})),
        default_permissions=dict(raw.get("default_permissions", {})),
        raw=raw,
    )


def _agent_executable_payload(spec: LoadedAgentSpec) -> dict[str, Any]:
    raw = {key: value for key, value in spec.raw.items() if key != "prompt_file"}
    return {
        "role": spec.role,
        "canonical_role": spec.canonical_role,
        "description": spec.description,
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "permissions": spec.permissions,
        "prompt": spec.prompt,
        "raw": raw,
    }


def _architecture_executable_payload(spec: LoadedArchitectureSpec) -> dict[str, Any]:
    return {
        "architecture_id": spec.architecture_id,
        "name": spec.name,
        "family": spec.family,
        "roles": list(spec.roles),
        "canonical_roles": dict(spec.canonical_roles),
        "entrypoint": spec.entrypoint,
        "terminal_nodes": list(spec.terminal_nodes),
        "edges": [list(edge) for edge in spec.edges],
        "orchestration": dict(spec.orchestration),
        "default_permissions": dict(spec.default_permissions),
        "raw": spec.raw,
    }


def executable_config_bundle(raw: dict[str, Any], benchmark: BenchmarkSpec) -> dict[str, Any]:
    """Deterministic payload for the executable experiment specification.

    Paths are omitted when the loaded contents are already included. All loaded
    permission sets, agent specs (including prompt text), and architecture specs
    are hashed so a prompt or spec edit cannot reuse another treatment's IDs.
    """

    return {
        "experiment": raw,
        "permissions": benchmark.permissions,
        "agents": {
            name: _agent_executable_payload(spec)
            for name, spec in sorted(benchmark.agents.items())
        },
        "architectures": {
            name: _architecture_executable_payload(spec)
            for name, spec in sorted(benchmark.architectures.items())
        },
    }


def compute_config_hash(raw: dict[str, Any], benchmark: BenchmarkSpec) -> str:
    return stable_hash(executable_config_bundle(raw, benchmark))


def load_benchmark_spec(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> BenchmarkSpec:
    project_root = Path(project_root)
    specs_root = project_root / "configs" / "benchmark_specs"
    architecture_dir = specs_root / "architectures"
    agent_dir = specs_root / "agents"
    prompt_dir = specs_root / "prompts"
    permission_file = specs_root / "permissions" / "permission_sets.yaml"
    permissions = load_yaml(permission_file)
    agents = {
        path.stem: _load_agent(path, project_root, permissions)
        for path in sorted(agent_dir.glob("*.yaml"))
    }
    architectures = {
        path.stem: _load_architecture(path)
        for path in sorted(architecture_dir.glob("*.yaml"))
    }
    return BenchmarkSpec(
        project_root=project_root,
        architecture_dir=architecture_dir,
        agent_dir=agent_dir,
        prompt_dir=prompt_dir,
        permission_file=permission_file,
        permissions=permissions,
        agents=agents,
        architectures=architectures,
    )


def load_experiment_spec(path: str | Path, project_root: str | Path = DEFAULT_PROJECT_ROOT) -> ExperimentSpec:
    path = Path(path)
    project_root = Path(project_root)
    raw = load_yaml(path)
    benchmark = load_benchmark_spec(project_root)
    return ExperimentSpec(
        path=path,
        experiment_id=raw["id"],
        raw=raw,
        benchmark=benchmark,
        config_hash=compute_config_hash(raw, benchmark),
    )
