"""Run 1-2 MASContributionBench experiments with bounded subprocess parallelism.

Examples:
    python scripts/run_suite.py exp01_full_system exp02_single_agent_baseline --jobs 2 --print-plan
    python scripts/run_suite.py configs/experiments/exp01_full_system.yaml --max-tasks 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.utils.io import load_yaml  # noqa: E402


MAX_JOBS = 2
MAX_CONFIGS = 2
_RUN_EXPERIMENT_MOD = None


def _load_run_experiment():
    global _RUN_EXPERIMENT_MOD
    if _RUN_EXPERIMENT_MOD is None:
        path = Path(__file__).resolve().parent / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("mas_run_experiment_cli", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUN_EXPERIMENT_MOD = module
    return _RUN_EXPERIMENT_MOD


def resolve_suite_entry(token: str) -> Path:
    aliases = _load_run_experiment().EXPERIMENT_ALIASES
    if token in aliases:
        path = Path(aliases[token])
    else:
        path = Path(token)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        known = ", ".join(sorted(aliases))
        raise SystemExit(f"Unknown experiment or missing config '{token}'. Known aliases: {known}")
    return path


def _fresh_run_inherited() -> bool:
    if "MAS_FRESH_RUN" not in os.environ:
        return False
    return os.environ.get("MAS_FRESH_RUN", "").lower() in {"1", "true", "yes", "y"}


def _is_exp09(token: str, experiment_id: str) -> bool:
    blob = f"{token} {experiment_id}".lower()
    return "exp09" in blob or "contribution_predictor" in blob


def _is_generalization(token: str, experiment_id: str) -> bool:
    blob = f"{token} {experiment_id}".lower()
    return "exp08" in blob or "generalization" in blob


def _declared_output_paths(raw: dict[str, Any], project_root: Path) -> list[Path]:
    outputs = raw.get("outputs") or {}
    values: list[Any] = list(outputs.values()) if isinstance(outputs, dict) else []
    paths: list[Path] = []
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        paths.append(path.resolve())
    return paths


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_relative_to(left, right) or _is_relative_to(right, left)


def _overlap_hits(left_paths: list[Path], right_paths: list[Path]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for left in left_paths:
        for right in right_paths:
            if _paths_overlap(left, right):
                hits.append((str(left), str(right)))
    return hits


def _child_cache_dir(index: int, experiment_id: str) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in experiment_id)
    return str(PROJECT_ROOT / "data" / "cache" / "suite_llm" / f"{index}_{safe_id}")


def build_plan(tokens: list[str], *, jobs: int | None, max_tasks: int | None) -> dict[str, Any]:
    if not tokens:
        raise SystemExit("Provide 1 or 2 experiment aliases or config paths.")
    if len(tokens) > MAX_CONFIGS:
        raise SystemExit(f"Suite accepts at most {MAX_CONFIGS} configs, got {len(tokens)}.")
    if _fresh_run_inherited():
        raise SystemExit(
            "Inherited MAS_FRESH_RUN is rejected for suite runs. "
            "Unset it before launching scripts/run_suite.py."
        )

    resolved_jobs = MAX_JOBS if jobs is None else jobs
    if resolved_jobs < 1 or resolved_jobs > MAX_JOBS:
        raise SystemExit(f"--jobs must be between 1 and {MAX_JOBS}, got {resolved_jobs}.")

    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for token in tokens:
        path = resolve_suite_entry(token)
        if path in seen:
            raise SystemExit(f"Duplicate config is rejected: {path}")
        seen.add(path)
        raw = load_yaml(path)
        experiment_id = str(raw.get("id") or path.stem)
        if _is_exp09(token, experiment_id):
            raise SystemExit(
                f"{experiment_id} is a planned contribution-predictor design and has no runner. "
                "Suite dispatch of Exp09 is rejected."
            )
        entries.append(
            {
                "token": token,
                "config": str(path),
                "experiment_id": experiment_id,
                "outputs": dict(raw.get("outputs") or {}),
                "output_paths": _declared_output_paths(raw, PROJECT_ROOT),
            }
        )

    if len(entries) == 2 and any(_is_generalization(row["token"], row["experiment_id"]) for row in entries):
        raise SystemExit(
            "Pairs containing Exp08/generalization are rejected. "
            "Run generalization separately from scripts/run_experiment.py."
        )

    if len(entries) == 2:
        hits = _overlap_hits(entries[0]["output_paths"], entries[1]["output_paths"])
        if hits:
            sample = "; ".join(f"{left} overlaps {right}" for left, right in hits[:4])
            raise SystemExit(f"Overlapping declared output paths/dirs are rejected: {sample}")

    children = []
    for index, entry in enumerate(entries):
        cache_dir = _child_cache_dir(index, entry["experiment_id"])
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
            "--config",
            entry["config"],
        ]
        if max_tasks is not None:
            cmd.extend(["--max-tasks", str(max_tasks)])
        children.append(
            {
                "token": entry["token"],
                "config": entry["config"],
                "experiment_id": entry["experiment_id"],
                "outputs": entry["outputs"],
                "cache_dir": cache_dir,
                "cmd": cmd,
            }
        )

    return {
        "jobs": min(resolved_jobs, len(children)),
        "max_jobs": MAX_JOBS,
        "max_tasks": max_tasks,
        "experiments": children,
    }


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _popen(cmd: list[str], *, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)


def _wait_parallel(procs: list[subprocess.Popen]) -> int:
    remaining = list(procs)
    while remaining:
        for proc in list(remaining):
            code = proc.poll()
            if code is None:
                continue
            remaining.remove(proc)
            if code:
                for other in remaining:
                    _stop_process(other)
                return code
        if remaining:
            try:
                remaining[0].wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
    return 0


def execute_plan(plan: dict[str, Any], *, popen=_popen) -> int:
    children = list(plan["experiments"])
    max_parallel = plan["jobs"]
    if not children:
        return 0

    def child_env(cache_dir: str) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("MAS_FRESH_RUN", None)
        env["MAS_LLM_CACHE_DIR"] = cache_dir
        return env

    if max_parallel <= 1 or len(children) == 1:
        for child in children:
            proc = popen(child["cmd"], env=child_env(child["cache_dir"]))
            code = proc.wait()
            if code:
                return code
        return 0

    procs = [popen(child["cmd"], env=child_env(child["cache_dir"])) for child in children]
    return _wait_parallel(procs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 1-2 configured MASContributionBench experiments with at most two subprocesses."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="One or two experiment aliases or configs/experiments/*.yaml paths.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"Max concurrent children (1-{MAX_JOBS}). Default is the number of configs, capped at {MAX_JOBS}.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Forwarded to each child run_experiment.py.")
    parser.add_argument("--print-plan", action="store_true", help="Print the resolved plan and exit.")
    args = parser.parse_args(argv)

    plan = build_plan(args.experiments, jobs=args.jobs, max_tasks=args.max_tasks)
    if args.print_plan:
        printable = {
            "jobs": plan["jobs"],
            "max_tasks": plan["max_tasks"],
            "experiments": [
                {
                    "token": child["token"],
                    "config": child["config"],
                    "experiment_id": child["experiment_id"],
                    "outputs": child["outputs"],
                    "cache_dir": child["cache_dir"],
                    "cmd": child["cmd"],
                }
                for child in plan["experiments"]
            ],
        }
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return 0
    return execute_plan(plan)


if __name__ == "__main__":
    raise SystemExit(main())
