"""Run one MASContributionBench experiment config.

Examples:
    python scripts/run_experiment.py --config configs/experiments/exp01_full_system.yaml --max-tasks 2
    python scripts/run_experiment.py --config configs/experiments/exp03_loo_attribution.yaml --max-tasks 2
    python scripts/run_experiment.py --experiment exp01_full_system --max-tasks 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mas_contribution_bench.config import load_experiment_spec  # noqa: E402
from mas_contribution_bench.runners import (  # noqa: E402
    run_attribution,
    run_full_system,
    run_generalization,
    run_intervention,
    run_single_agent_baseline,
)


EXPERIMENT_ALIASES = {
    "exp01_full_system": "configs/experiments/exp01_full_system.yaml",
    "exp02_single_agent_baseline": "configs/experiments/exp02_single_agent_baseline.yaml",
    "exp03_loo_attribution": "configs/experiments/exp03_loo_attribution.yaml",
    "exp01_qwen_all_tasksets": "configs/experiments/exp01_qwen_all_tasksets.yaml",
    "exp02_qwen_all_tasksets_single_agent_baseline": "configs/experiments/exp02_qwen_all_tasksets_single_agent_baseline.yaml",
    "exp03_qwen_all_tasksets_loo_attribution": "configs/experiments/exp03_qwen_all_tasksets_loo_attribution.yaml",
    "exp04_shapley_attribution": "configs/experiments/exp04_shapley_attribution.yaml",
    "exp05_topology_intervention": "configs/experiments/exp05_topology_intervention.yaml",
    "exp06_role_intervention": "configs/experiments/exp06_role_intervention.yaml",
    "exp07_permission_intervention": "configs/experiments/exp07_permission_intervention.yaml",
    "exp08_generalization": "configs/experiments/exp08_generalization.yaml",
    "exp09_contribution_predictor": "configs/experiments/exp09_contribution_predictor.yaml",
}


def resolve_config(args: argparse.Namespace) -> Path:
    if args.config:
        path = Path(args.config)
    elif args.experiment:
        if args.experiment not in EXPERIMENT_ALIASES:
            known = ", ".join(sorted(EXPERIMENT_ALIASES))
            raise SystemExit(f"Unknown experiment '{args.experiment}'. Known: {known}")
        path = Path(EXPERIMENT_ALIASES[args.experiment])
    else:
        raise SystemExit("Provide --config or --experiment.")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")
    return path


UNIMPLEMENTED_EXPERIMENTS = {
    "exp09_contribution_predictor": (
        "exp09_contribution_predictor is a planned design and has no runner. "
        "Refusing to dispatch it in auto or forced mode; it must not fall through "
        "to run_full_system or reuse data/runs/full_system."
    ),
}


def refuse_unimplemented_experiment(experiment_id: str) -> None:
    experiment_id = str(experiment_id)
    message = UNIMPLEMENTED_EXPERIMENTS.get(experiment_id)
    if message is None and "contribution_predictor" in experiment_id:
        message = (
            f"{experiment_id} is a planned contribution-predictor design and has no runner. "
            "Refusing to dispatch it in auto or forced mode."
        )
    if message:
        raise SystemExit(message)


def infer_runner(experiment_id: str, mode: str | None):
    refuse_unimplemented_experiment(experiment_id)
    selected = mode or "auto"
    if selected == "full":
        return run_full_system
    if selected == "single":
        return run_single_agent_baseline
    if selected == "attribution":
        return run_attribution
    if selected == "intervention":
        return run_intervention
    if selected != "auto":
        raise SystemExit(f"Unknown mode: {selected}")
    if "single_agent" in experiment_id:
        return run_single_agent_baseline
    if "generalization" in experiment_id:
        return run_generalization
    if "intervention" in experiment_id:
        return run_intervention
    if any(key in experiment_id for key in ["loo", "shapley", "banzhaf", "myerson", "owen"]):
        return run_attribution
    return run_full_system


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured MASContributionBench experiment.")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/experiments/*.yaml.")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment id alias, e.g. exp01_full_system.")
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "single", "attribution", "intervention"],
        default="auto",
        help="Runner mode. Default infers from experiment id.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks for smoke tests.")
    parser.add_argument(
        "--execute-code",
        action="store_true",
        help="Run HumanEval/MBPP code tests through the configured sandbox.",
    )
    parser.add_argument(
        "--sandbox-backend",
        choices=["auto", "docker", "subprocess"],
        default=None,
        help="Sandbox backend for --execute-code. Use docker for formal experiments.",
    )
    parser.add_argument(
        "--model-backend",
        choices=["dry-run", "deepseek", "vllm", "qwen", "openai", "openai_compatible"],
        default=None,
        help="Model backend. Default is dry-run unless MAS_MODEL_BACKEND is set.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name, e.g. deepseek-chat.")
    parser.add_argument("--print-config", action="store_true", help="Print parsed experiment id and exit.")
    args = parser.parse_args()

    config_path = resolve_config(args)
    experiment = load_experiment_spec(config_path, PROJECT_ROOT)
    if args.print_config:
        print(json.dumps({"experiment_id": experiment.experiment_id, "config": str(config_path)}, indent=2))
        return 0

    runner = infer_runner(experiment.experiment_id, args.mode)
    if args.model_backend:
        os.environ["MAS_MODEL_BACKEND"] = args.model_backend
    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
        os.environ["DEEPSEEK_MODEL"] = args.model_name
        os.environ["OPENAI_MODEL"] = args.model_name
        os.environ["VLLM_MODEL"] = args.model_name
    if args.execute_code:
        os.environ["MAS_EXECUTE_CODE"] = "1"
    if args.sandbox_backend:
        os.environ["MAS_SANDBOX_BACKEND"] = args.sandbox_backend
    summary: dict[str, Any] = runner(config_path, max_tasks=args.max_tasks)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
