"""Regression tests for executable hashes, treatment IDs, and reuse/method safety."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
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
from mas_contribution_bench.data.schemas import EvaluationConfig, EvaluationRecord, EvaluatorType, FailureType
from mas_contribution_bench.runners.common import (
    WIRED_ATTRIBUTION_METHODS,
    WIRED_INTERVENTION_METHODS,
    execution_fingerprint,
    execution_treatment,
    identity_role_map,
    listed_attribution_methods,
    mas_run_id,
    require_evaluation_score,
    run_mas_once,
    validate_attribution_methods,
    validate_permission_toggles,
)
from mas_contribution_bench.runners.run_single_agent import _baseline_run_id, _run_baseline_once
from mas_contribution_bench.runners.run_attribution import (
    _coalition_key,
    _loo_attribution_id,
    run_attribution,
    run_coalition_attribution,
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

        text = (ROOT / "configs" / "experiments" / "exp03_loo_attribution.yaml").read_text(
            encoding="utf-8"
        )
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


class FailClosedEvaluationScoreTests(unittest.TestCase):
    def test_none_score_names_dataset_and_rejects_export_only(self) -> None:
        record = EvaluationRecord(
            run_id="r1",
            task_id="livecodebench/1",
            dataset="livecodebench",
            architecture_id="A1",
            score=None,
            metric="pass_at_1",
            evaluator=EvaluationConfig(evaluator_type=EvaluatorType.UNIT_TEST, metric="pass_at_1"),
        )
        with self.assertRaises(ValueError) as ctx:
            require_evaluation_score(record, dataset="livecodebench", task_id="livecodebench/1")
        message = str(ctx.exception)
        self.assertIn("livecodebench", message)
        self.assertIn("livecodebench/1", message)
        self.assertIn("Prediction/export-only", message)
        self.assertFalse(hasattr(attribution_mod, "_as_float"))
        self.assertEqual(
            require_evaluation_score({"score": 1.0, "dataset": "hle", "task_id": "hle/1"}),
            1.0,
        )
        scored = EvaluationRecord(
            run_id="r1",
            task_id="livecodebench/1",
            dataset="livecodebench",
            architecture_id="A1",
            score=1.0,
            metric="pass_at_1",
            evaluator=EvaluationConfig(evaluator_type=EvaluatorType.UNIT_TEST, metric="pass_at_1"),
        )
        self.assertEqual(require_evaluation_score(scored), 1.0)

    def test_loo_raises_before_using_none_as_zero(self) -> None:
        experiment = _experiment(
            raw={
                "id": "exp03_loo_attribution",
                "attribution": {"method": "loo"},
                "seeds": [0],
                "outputs": {},
            },
            experiment_id="exp03_loo_attribution",
        )
        task = {"task_id": "livecodebench/1", "dataset": "livecodebench"}
        evaluation = EvaluationRecord(
            run_id="run-none",
            task_id=task["task_id"],
            dataset=task["dataset"],
            architecture_id="A1",
            score=None,
            metric="pass_at_1",
            evaluator=EvaluationConfig(evaluator_type=EvaluatorType.UNIT_TEST, metric="pass_at_1"),
        )
        run = type("Run", (), {"run_id": "run-none"})()
        with patch.object(attribution_mod, "load_experiment", return_value=experiment):
            with patch.object(attribution_mod, "select_tasks", return_value=[task]):
                with patch.object(attribution_mod, "select_architectures", return_value=["A1"]):
                    with patch.object(
                        attribution_mod,
                        "run_mas_once",
                        return_value=(run, [], evaluation),
                    ):
                        with patch.object(
                            attribution_mod,
                            "append_jsonl",
                            side_effect=AssertionError("should not persist unscored evaluation"),
                        ):
                            with self.assertRaises(ValueError) as ctx:
                                run_loo_attribution("unused.yaml")
        message = str(ctx.exception)
        self.assertIn("livecodebench", message)
        self.assertIn("Prediction/export-only", message)

    def test_stale_cached_none_score_fails_closed_before_attribution(self) -> None:
        experiment = _experiment(
            raw={
                "id": "exp04_shapley_attribution",
                "attribution": {"methods": ["shapley_sampled"], "shapley": {"num_permutations": 1}},
                "seeds": [0],
                "outputs": {},
            },
            experiment_id="exp04_shapley_attribution",
        )
        task = {"task_id": "hle/1", "dataset": "hle"}
        roles = experiment.benchmark.architectures["A1"].roles
        fingerprint = execution_fingerprint(experiment, removal_protocol="null_agent_replacement")
        coalition_id = _coalition_key(
            experiment.experiment_id,
            task["task_id"],
            "A1",
            0,
            set(roles),
            fingerprint,
        )
        stale = {
            "coalition_id": coalition_id,
            "score": None,
            "dataset": "hle",
            "task_id": "hle/1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = replace(experiment, benchmark=replace(experiment.benchmark, project_root=root))
            coalition_path = root / f"data/results/attribution/{experiment.experiment_id}_coalitions.jsonl"
            attribution_path = root / f"data/results/attribution/{experiment.experiment_id}.jsonl"
            coalition_path.parent.mkdir(parents=True, exist_ok=True)
            coalition_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
            with patch.object(attribution_mod, "load_experiment", return_value=experiment):
                with patch.object(attribution_mod, "select_tasks", return_value=[task]):
                    with patch.object(attribution_mod, "select_architectures", return_value=["A1"]):
                        with patch.object(
                            attribution_mod,
                            "run_mas_once",
                            side_effect=AssertionError("should not run after stale cache"),
                        ):
                            with self.assertRaises(ValueError) as ctx:
                                run_coalition_attribution("unused.yaml")
            self.assertIn("hle", str(ctx.exception))
            self.assertIn("score is None", str(ctx.exception))
            self.assertFalse(attribution_path.exists())


class PermissionRuntimeAllowlistTests(unittest.TestCase):
    def test_runtime_overrides_reject_non_allowlisted_toggles_before_model(self) -> None:
        with patch(
            "mas_contribution_bench.runners.common.build_model_client",
            side_effect=AssertionError("should not build a model client"),
        ):
            with self.assertRaises(ValueError) as ctx:
                run_mas_once(
                    _experiment(),
                    {"task_id": "t1", "dataset": "hle", "prompt": "q"},
                    "A1",
                    0,
                    permission_overrides={"coder": {"read_memory": False}},
                )
        message = str(ctx.exception)
        self.assertIn("read_memory", message)
        self.assertIn("write_solution", message)
        validate_permission_toggles(["write_solution", "final_answer"], context="ok")


class BaselineTreatmentIdentityTests(unittest.TestCase):
    def test_precomputed_and_written_ids_match_and_cover_treatment(self) -> None:
        experiment = _experiment(experiment_id="exp02_single_agent_baseline")
        task = {"task_id": "task-1", "dataset": "hle", "prompt": "q"}
        spec = {"id": "single_agent_coder", "type": "single_agent", "roles": ["coder"], "sample": 0}
        env = {
            "MAS_MODEL_BACKEND": "vllm",
            "MODEL_NAME": "model-a",
            "MAS_EXECUTE_CODE": "0",
            "MAS_SANDBOX_BACKEND": "subprocess",
        }
        evaluation = EvaluationRecord(
            run_id="placeholder",
            task_id=task["task_id"],
            dataset=task["dataset"],
            architecture_id=spec["id"],
            score=1.0,
            metric="exact_match",
            passed=True,
            failure_type=FailureType.NONE,
            evaluator=EvaluationConfig(evaluator_type=EvaluatorType.EXACT_MATCH, metric="exact_match"),
        )
        state = {"task": task, "messages": [], "agent_outputs": {}, "final_answer": "ok"}
        with patch.dict(os.environ, env, clear=False):
            expected = _baseline_run_id(experiment, task, spec, 0, ["coder"])
            mas_expected = mas_run_id(
                experiment,
                task["task_id"],
                spec["id"],
                0,
                removed_agents=[],
                removal_protocol="none",
                role_map_items=sorted(identity_role_map(["coder"]).items()),
                condition_id="baseline",
            )
            self.assertEqual(expected, mas_expected)
            with patch(
                "mas_contribution_bench.runners.run_single_agent._invoke_roles",
                return_value=(state, "ok"),
            ):
                with patch(
                    "mas_contribution_bench.runners.run_single_agent.evaluate_task_output",
                    return_value=evaluation,
                ):
                    run, _traces, written_eval = _run_baseline_once(experiment, task, spec, 0)
            fingerprint = execution_fingerprint(experiment, removal_protocol="none")
            treatment = execution_treatment(experiment, removal_protocol="none")

        self.assertEqual(run.run_id, expected)
        self.assertEqual(run.metadata["execution_fingerprint"], fingerprint)
        self.assertEqual(run.metadata["execution_treatment"], treatment)
        self.assertEqual(written_eval.metadata["execution_fingerprint"], fingerprint)
        self.assertEqual(written_eval.metadata["execution_treatment"]["model_backend"], "vllm")
        self.assertEqual(written_eval.metadata["execution_treatment"]["sandbox_backend"], "subprocess")

        renamed = dict(env)
        renamed["MODEL_NAME"] = "model-b"
        with patch.dict(os.environ, renamed, clear=False):
            renamed_id = _baseline_run_id(experiment, task, spec, 0, ["coder"])
        self.assertNotEqual(expected, renamed_id)

        other_roles = _baseline_run_id(experiment, task, spec, 0, ["planner"])
        self.assertNotEqual(expected, other_roles)

        execute_env = dict(env)
        execute_env["MAS_EXECUTE_CODE"] = "1"
        with patch.dict(os.environ, execute_env, clear=False):
            execute_id = _baseline_run_id(experiment, task, spec, 0, ["coder"])
        self.assertNotEqual(expected, execute_id)


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
