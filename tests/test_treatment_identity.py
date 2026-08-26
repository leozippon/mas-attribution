"""Regression tests for executable hashes, treatment IDs, and reuse/method safety."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.config.loaders import (
    BenchmarkSpec,
    ExperimentSpec,
    LoadedAgentSpec,
    LoadedArchitectureSpec,
    compute_config_hash,
    executable_config_bundle,
    load_experiment_spec,
)
from mas_contribution_bench.runners.common import (
    WIRED_ATTRIBUTION_METHODS,
    WIRED_INTERVENTION_METHODS,
    execution_fingerprint,
    execution_treatment,
    listed_attribution_methods,
    mas_run_id,
    validate_attribution_methods,
)
from mas_contribution_bench.runners.run_attribution import (
    _coalition_key,
    _loo_attribution_id,
    run_attribution,
    run_loo_attribution,
)
from mas_contribution_bench.runners.run_intervention import (
    _condition_attribution_id,
    _permission_coalition_key,
    _role_coalition_key,
    run_topology_intervention,
)
from mas_contribution_bench.utils.io import stable_hash

attribution_mod = importlib.import_module("mas_contribution_bench.runners.run_attribution")
intervention_mod = importlib.import_module("mas_contribution_bench.runners.run_intervention")


def _agent(*, prompt: str = "PROMPT", role: str = "coder") -> LoadedAgentSpec:
    return LoadedAgentSpec(
        role=role,
        canonical_role=role,
        description="desc",
        model="model",
        temperature=0.2,
        max_tokens=128,
        permissions={"code": True},
        prompt_file="configs/benchmark_specs/prompts/coder.md",
        prompt=prompt,
        raw={"role": role, "prompt_file": "configs/benchmark_specs/prompts/coder.md"},
    )


def _architecture(*, architecture_id: str = "A1", roles: list[str] | None = None) -> LoadedArchitectureSpec:
    role_list = list(roles or ["coder"])
    return LoadedArchitectureSpec(
        architecture_id=architecture_id,
        name=architecture_id,
        family="test",
        roles=role_list,
        canonical_roles={role: role for role in role_list},
        entrypoint=role_list[0],
        terminal_nodes=["final_answer"],
        edges=[],
        orchestration={},
        default_permissions={},
        raw={"id": architecture_id, "roles": role_list},
    )


def _benchmark(
    *,
    agents: dict[str, LoadedAgentSpec] | None = None,
    architectures: dict[str, LoadedArchitectureSpec] | None = None,
    permissions: dict[str, dict[str, object]] | None = None,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        project_root=Path("/tmp/mas-attribution-test"),
        architecture_dir=Path("/tmp/mas-attribution-test/arch"),
        agent_dir=Path("/tmp/mas-attribution-test/agents"),
        prompt_dir=Path("/tmp/mas-attribution-test/prompts"),
        permission_file=Path("/tmp/mas-attribution-test/permissions.yaml"),
        permissions=permissions or {"coder": {"code": True}},
        agents=agents or {"coder": _agent()},
        architectures=architectures or {"A1": _architecture()},
    )


def _experiment(
    *,
    raw: dict | None = None,
    config_hash: str = "cfghash",
    experiment_id: str = "exp_dummy",
) -> ExperimentSpec:
    return ExperimentSpec(
        path=Path("dummy.yaml"),
        experiment_id=experiment_id,
        raw=raw
        or {
            "id": experiment_id,
            "model": {"backend": "vllm", "name": "from-yaml"},
            "evaluation": {"execute_code": False, "sandbox_backend": "auto"},
        },
        benchmark=_benchmark(),
        config_hash=config_hash,
    )


def _write_min_project(root: Path, *, prompt: str, architecture_roles: list[str] | None = None) -> Path:
    architecture_dir = root / "configs" / "benchmark_specs" / "architectures"
    agent_dir = root / "configs" / "benchmark_specs" / "agents"
    prompt_dir = root / "configs" / "benchmark_specs" / "prompts"
    permission_dir = root / "configs" / "benchmark_specs" / "permissions"
    experiment_dir = root / "configs" / "experiments"
    for path in (architecture_dir, agent_dir, prompt_dir, permission_dir, experiment_dir):
        path.mkdir(parents=True, exist_ok=True)

    (permission_dir / "permission_sets.yaml").write_text("coder:\n  code: true\n", encoding="utf-8")
    (prompt_dir / "coder.md").write_text(prompt, encoding="utf-8")
    (agent_dir / "coder.yaml").write_text(
        "\n".join(
            [
                "role: coder",
                "prompt_file: configs/benchmark_specs/prompts/coder.md",
                "permissions:",
                "  inherit_from: coder",
                "",
            ]
        ),
        encoding="utf-8",
    )
    roles = architecture_roles or ["coder"]
    role_lines = "\n".join(f"- {role}" for role in roles)
    (architecture_dir / "A1.yaml").write_text(
        f"id: A1\nroles:\n{role_lines}\nedges: []\n",
        encoding="utf-8",
    )
    experiment_path = experiment_dir / "exp.yaml"
    experiment_path.write_text("id: exp_hash_test\n", encoding="utf-8")
    return experiment_path


def _write_temp_experiment(directory: Path, body: str) -> Path:
    path = directory / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class ExecutableConfigHashTests(unittest.TestCase):
    def test_prompt_and_spec_changes_change_hash_payload_and_result(self) -> None:
        raw = {"id": "exp_hash", "seeds": [0]}
        first = _benchmark(agents={"coder": _agent(prompt="alpha")})
        second = _benchmark(agents={"coder": _agent(prompt="beta")})
        payload_first = executable_config_bundle(raw, first)
        payload_second = executable_config_bundle(raw, second)
        self.assertEqual(payload_first["agents"]["coder"]["prompt"], "alpha")
        self.assertEqual(payload_second["agents"]["coder"]["prompt"], "beta")
        self.assertNotEqual(payload_first, payload_second)
        self.assertNotEqual(compute_config_hash(raw, first), compute_config_hash(raw, second))

        arch_changed = _benchmark(architectures={"A1": _architecture(roles=["coder", "planner"])})
        self.assertNotEqual(compute_config_hash(raw, first), compute_config_hash(raw, arch_changed))
        self.assertEqual(compute_config_hash(raw, first), stable_hash(payload_first))

    def test_load_experiment_spec_hash_tracks_loaded_prompt_content(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir)
            second_root = Path(second_dir)
            first_path = _write_min_project(first_root, prompt="prompt-one")
            second_path = _write_min_project(second_root, prompt="prompt-two")
            first = load_experiment_spec(first_path, first_root)
            second = load_experiment_spec(second_path, second_root)
            self.assertEqual(first.config_hash, compute_config_hash(first.raw, first.benchmark))
            self.assertNotEqual(first.config_hash, second.config_hash)
            self.assertIn("prompt-one", first.benchmark.agents["coder"].prompt)
            self.assertIn("prompt-two", second.benchmark.agents["coder"].prompt)


class RuntimeFingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_runtime_treatment_and_excludes_secrets(self) -> None:
        experiment = _experiment()
        env = {
            "MAS_MODEL_BACKEND": "vllm",
            "MODEL_NAME": "model-a",
            "MAS_EXECUTE_CODE": "0",
            "MAS_SANDBOX_BACKEND": "auto",
        }
        with patch.dict(os.environ, env, clear=False):
            treatment = execution_treatment(experiment, removal_protocol="null_agent_replacement")
        self.assertEqual(
            set(treatment),
            {
                "config_hash",
                "model_backend",
                "model_name",
                "execute_code",
                "sandbox_backend",
                "removal_protocol",
            },
        )
        self.assertEqual(treatment["model_backend"], "vllm")
        self.assertEqual(treatment["model_name"], "model-a")
        self.assertFalse(treatment["execute_code"])
        self.assertEqual(treatment["sandbox_backend"], "auto")
        self.assertEqual(treatment["removal_protocol"], "null_agent_replacement")
        serialized = stable_hash(treatment)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("base_url", str(treatment))

    def test_backend_name_and_protocol_change_ids(self) -> None:
        experiment = _experiment()
        base_env = {
            "MAS_MODEL_BACKEND": "vllm",
            "MODEL_NAME": "model-a",
            "MAS_EXECUTE_CODE": "0",
            "MAS_SANDBOX_BACKEND": "subprocess",
        }
        with patch.dict(os.environ, base_env, clear=False):
            fingerprint_a = execution_fingerprint(experiment, removal_protocol="null_agent_replacement")
            run_a = mas_run_id(experiment, "task-1", "A1", 0, removal_protocol="null_agent_replacement")
            coalition_a = _coalition_key(
                experiment.experiment_id,
                "task-1",
                "A1",
                0,
                {"coder"},
                fingerprint_a,
            )
            attr_a = _loo_attribution_id(
                experiment.experiment_id,
                "task-1",
                "A1",
                0,
                "coder",
                fingerprint_a,
            )

        renamed = dict(base_env)
        renamed["MODEL_NAME"] = "model-b"
        with patch.dict(os.environ, renamed, clear=False):
            fingerprint_name = execution_fingerprint(experiment, removal_protocol="null_agent_replacement")
            run_name = mas_run_id(experiment, "task-1", "A1", 0, removal_protocol="null_agent_replacement")

        backend = dict(base_env)
        backend["MAS_MODEL_BACKEND"] = "deepseek"
        backend["DEEPSEEK_MODEL"] = "deepseek-chat"
        with patch.dict(os.environ, backend, clear=False):
            fingerprint_backend = execution_fingerprint(experiment, removal_protocol="null_agent_replacement")

        with patch.dict(os.environ, base_env, clear=False):
            fingerprint_protocol = execution_fingerprint(experiment, removal_protocol="hard_removal")
            run_protocol = mas_run_id(experiment, "task-1", "A1", 0, removal_protocol="hard_removal")
            coalition_protocol = _coalition_key(
                experiment.experiment_id,
                "task-1",
                "A1",
                0,
                {"coder"},
                fingerprint_protocol,
            )
            attr_protocol = _loo_attribution_id(
                experiment.experiment_id,
                "task-1",
                "A1",
                0,
                "coder",
                fingerprint_protocol,
            )

        self.assertNotEqual(fingerprint_a, fingerprint_name)
        self.assertNotEqual(fingerprint_a, fingerprint_backend)
        self.assertNotEqual(fingerprint_a, fingerprint_protocol)
        self.assertNotEqual(run_a, run_name)
        self.assertNotEqual(run_a, run_protocol)
        self.assertNotEqual(coalition_a, coalition_protocol)
        self.assertNotEqual(attr_a, attr_protocol)

    def test_role_map_permission_and_condition_remain_in_identities(self) -> None:
        fingerprint = "fp"
        role_identity = _role_coalition_key(
            "exp06",
            "task-1",
            "A1",
            0,
            {"coder", "planner"},
            {"coder": "coder", "planner": "planner"},
            fingerprint,
        )
        role_swapped = _role_coalition_key(
            "exp06",
            "task-1",
            "A1",
            0,
            {"coder", "planner"},
            {"coder": "planner", "planner": "coder"},
            fingerprint,
        )
        self.assertNotEqual(role_identity, role_swapped)

        permission_on = _permission_coalition_key(
            "exp07",
            "task-1",
            "A1",
            0,
            {"coder"},
            "A1__mem__true",
            {"coder": {"read_memory": True}},
            fingerprint,
        )
        permission_off = _permission_coalition_key(
            "exp07",
            "task-1",
            "A1",
            0,
            {"coder"},
            "A1__mem__true",
            {"coder": {"read_memory": False}},
            fingerprint,
        )
        self.assertNotEqual(permission_on, permission_off)

        condition_a = _condition_attribution_id(
            "exp06",
            "task-1",
            "A1",
            0,
            "cond-a",
            "loo",
            "coder",
            fingerprint,
        )
        condition_b = _condition_attribution_id(
            "exp06",
            "task-1",
            "A1",
            0,
            "cond-b",
            "loo",
            "coder",
            fingerprint,
        )
        self.assertNotEqual(condition_a, condition_b)


class DonorReuseAndMethodTests(unittest.TestCase):
    def test_no_hardcoded_donor_path_remains_reachable(self) -> None:
        source = inspect.getsource(attribution_mod)
        self.assertNotIn("data/runs/full_system", source)
        self.assertNotIn("data/results/scores/full_system_scores.jsonl", source)
        self.assertNotIn("exp01_full_system", source)
        self.assertNotIn("exp01_full_system_cache", source)
        self.assertFalse(hasattr(attribution_mod, "_full_system_score_index"))

        for name in (
            "exp03_loo_attribution.yaml",
            "exp03_qwen_loo_attribution.yaml",
            "exp03_qwen_all_tasksets_loo_attribution.yaml",
            "exp03_qwen_all_tasksets_loo_attribution_smoke.yaml",
        ):
            text = (ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8")
            self.assertIn("reuse_full_system_runs: false", text)
            self.assertNotIn("reuse_full_system_runs: true", text)

    def test_reuse_true_fails_before_work(self) -> None:
        body = "\n".join(
            [
                "id: exp03_reuse_guard",
                "attribution:",
                "  method: loo",
                "  reuse_full_system_runs: true",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_temp_experiment(Path(tmp), body)
            with patch.object(
                attribution_mod,
                "select_tasks",
                side_effect=AssertionError("should not select tasks"),
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_loo_attribution(path)
            message = str(ctx.exception)
            self.assertIn("reuse_full_system_runs", message)
            self.assertIn("not implemented", message.lower())

            with patch.object(
                attribution_mod,
                "select_tasks",
                side_effect=AssertionError("should not select tasks"),
            ):
                with self.assertRaises(ValueError):
                    run_attribution(path)

    def test_unsupported_methods_are_rejected_instead_of_rewritten(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_attribution_methods(
                ["myerson"],
                WIRED_ATTRIBUTION_METHODS,
                context="attribution runner",
            )
        message = str(ctx.exception)
        self.assertIn("myerson", message.lower())
        self.assertIn("unwired", message.lower())

        with self.assertRaises(ValueError):
            validate_attribution_methods(
                ["owen"],
                WIRED_INTERVENTION_METHODS,
                context="intervention runner",
            )
        with self.assertRaises(ValueError):
            validate_attribution_methods(
                ["shapley_exact"],
                WIRED_ATTRIBUTION_METHODS,
                context="attribution runner",
            )
        with self.assertRaises(ValueError):
            validate_attribution_methods(
                ["banzhaf_sampled"],
                WIRED_INTERVENTION_METHODS,
                context="intervention runner",
            )

        preserved = listed_attribution_methods(
            {"methods": ["sampled_shapley", "sampled_banzhaf"]}
        )
        self.assertEqual(preserved, ["shapley_sampled", "banzhaf_sampled"])
        self.assertEqual(
            validate_attribution_methods(
                preserved,
                WIRED_ATTRIBUTION_METHODS,
                context="exp04",
            ),
            ["shapley_sampled", "banzhaf_sampled"],
        )

        body = "\n".join(
            [
                "id: exp04_method_guard",
                "attribution:",
                "  method: myerson",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_temp_experiment(Path(tmp), body)
            with patch.object(
                attribution_mod,
                "select_tasks",
                side_effect=AssertionError("should not select tasks"),
            ):
                with self.assertRaises(ValueError) as raised:
                    run_attribution(path)
            self.assertIn("myerson", str(raised.exception).lower())

            intervention_body = "\n".join(
                [
                    "id: exp05_method_guard",
                    "attribution:",
                    "  methods:",
                    "    - owen",
                    "",
                ]
            )
            intervention_dir = Path(tmp) / "intervention"
            intervention_dir.mkdir()
            intervention_path = _write_temp_experiment(intervention_dir, intervention_body)
            with patch.object(
                intervention_mod,
                "select_tasks",
                side_effect=AssertionError("should not select tasks"),
            ):
                with self.assertRaises(ValueError) as intervention_ctx:
                    run_topology_intervention(intervention_path)
            self.assertIn("owen", str(intervention_ctx.exception).lower())


class RunMetadataFingerprintTests(unittest.TestCase):
    def test_run_mas_once_persists_fingerprint_without_external_models(self) -> None:
        from mas_contribution_bench.runners.common import identity_role_map, run_mas_once

        experiment = load_experiment_spec(
            ROOT / "configs" / "experiments" / "exp04_shapley_attribution.yaml",
            ROOT,
        )
        task = {
            "task_id": "identity-task",
            "dataset": "gpqa_diamond",
            "prompt": "answer",
            "reference_solution": "ok",
            "evaluation": {"evaluator_type": "exact_match", "metric": "exact_match"},
        }
        env = {
            "MAS_MODEL_BACKEND": "dry-run",
            "MAS_EXECUTE_CODE": "0",
            "MAS_SANDBOX_BACKEND": "auto",
        }
        with patch.dict(os.environ, env, clear=False):
            run, _traces, _evaluation = run_mas_once(
                experiment,
                task,
                "A1_pev",
                0,
                removal_protocol="null_agent_replacement",
            )
            expected = mas_run_id(
                experiment,
                task["task_id"],
                "A1_pev",
                0,
                removed_agents=[],
                removal_protocol="null_agent_replacement",
                role_map_items=sorted(
                    identity_role_map(experiment.benchmark.architectures["A1_pev"].roles).items()
                ),
            )
        self.assertEqual(run.run_id, expected)
        self.assertIn("execution_fingerprint", run.metadata)
        self.assertEqual(
            run.metadata["execution_treatment"]["removal_protocol"],
            "null_agent_replacement",
        )
        self.assertEqual(run.metadata["execution_treatment"]["model_backend"], "dry-run")
        self.assertEqual(run.config_hash, experiment.config_hash)


if __name__ == "__main__":
    unittest.main()
