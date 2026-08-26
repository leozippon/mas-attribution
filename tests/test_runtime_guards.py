"""Offline tests for prompt leakage, seed cache, task selection, and null terminals."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from types import SimpleNamespace

from mas_contribution_bench.agents.base import BaseAgent, DeepSeekModelClient, OpenAICompatibleModelClient
from mas_contribution_bench.config.loaders import BenchmarkSpec, ExperimentSpec
from mas_contribution_bench.graphs.architectures import controlled_architecture
from mas_contribution_bench.graphs.langgraph_builder import MASGraphBuilder
from mas_contribution_bench.runners.common import select_tasks
from mas_contribution_bench.runners.run_intervention import (
    _clone_architecture,
    _graph_feature_row,
    _inject_topology_variants,
    _permission_level_overrides,
    run_permission_intervention,
)


def _user_text(agent: BaseAgent, task: dict) -> str:
    messages = agent.build_messages({"task": task, "messages": []})
    return messages[1]["content"]


def _write_tasks(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task_payload(task_id: str, *, difficulty: str = "unknown") -> dict:
    return {
        "task_id": task_id,
        "dataset": "hle",
        "split": "test",
        "task_type": "general_mas",
        "prompt": f"question {task_id}",
        "evaluation": {"evaluator_type": "exact_match", "metric": "answer_accuracy"},
        "source": {"raw_dataset": "hle", "raw_task_id": task_id},
        "difficulty": {"level": difficulty, "source": "test"},
    }


def _benchmark(root: Path) -> BenchmarkSpec:
    return BenchmarkSpec(
        project_root=root,
        architecture_dir=root / "arch",
        agent_dir=root / "agents",
        prompt_dir=root / "prompts",
        permission_file=root / "permissions.yaml",
        permissions={},
        agents={},
        architectures={},
    )


class PromptLeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = BaseAgent("coder", "coder", "system", {"read_task": True})

    def test_tests_gold_and_arc_outputs_stay_out_of_messages(self) -> None:
        arc_prompt = json.dumps(
            {
                "train": [{"input": [[1]], "output": [[7]]}],
                "test": [{"input": [[2]], "output": [[99]]}],
            }
        )
        task = {
            "task_id": "arc_agi_2/leak",
            "dataset": "arc_agi_2",
            "prompt": arc_prompt,
            "tests": "SECRET_ASSERT foo() == 1",
            "reference_solution": "SECRET_GOLD",
            "metadata": {"Correct Answer": "SECRET_META", "expected_test_outputs": [[[99]]]},
            "output_format": "json_grid",
        }
        user = _user_text(self.agent, task)
        self.assertIn("[[1]]", user)
        self.assertIn("[[7]]", user)
        self.assertIn("[[2]]", user)
        self.assertIn("json_grid", user)
        self.assertNotIn("99", user)
        self.assertNotIn("SECRET_ASSERT", user)
        self.assertNotIn("SECRET_GOLD", user)
        self.assertNotIn("SECRET_META", user)
        self.assertNotIn("Visible tests/assertions", user)

        lcb = {
            "task_id": "livecodebench/1",
            "dataset": "livecodebench",
            "prompt": "Write foo",
            "tests": "assert False  # private",
            "reference_solution": "def foo():\n    return 1\n",
        }
        lcb_user = _user_text(self.agent, lcb)
        self.assertIn("Write foo", lcb_user)
        self.assertNotIn("assert False", lcb_user)
        self.assertNotIn("return 1", lcb_user)


class SeedCacheTests(unittest.TestCase):
    def test_request_payload_and_cache_key_include_seed(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        client = DeepSeekModelClient(api_key="dummy")
        payload_a = {
            "model": "m",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8,
            "seed": 1,
        }
        payload_b = dict(payload_a)
        payload_b["seed"] = 2
        self.assertNotEqual(client._cache_key(payload_a), client._cache_key(payload_b))

        class _Response:
            status_code = 200

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "MAS_LLM_CACHE_ENABLED": "1",
                "MAS_LLM_CACHE_DIR": tmp,
                "MAS_LLM_CACHE_FILE": "",
            }
            DeepSeekModelClient._shared_cache_indexes.clear()
            captured: list[dict] = []

            def _post(*_args, **kwargs):
                captured.append(kwargs["json"])
                return _Response()

            with patch.dict(os.environ, env, clear=False), patch("requests.post", side_effect=_post):
                vllm = OpenAICompatibleModelClient(
                    api_key="dummy",
                    base_url="http://127.0.0.1:9/v1",
                    default_model="qwen-local",
                )
                vllm.complete(messages, seed=3, model="qwen-local", temperature=0.0, max_tokens=8)
                vllm.complete(messages, seed=4, model="qwen-local", temperature=0.0, max_tokens=8)
            self.assertEqual([row["seed"] for row in captured], [3, 4])
            self.assertNotEqual(
                client._cache_key(captured[0]),
                client._cache_key(captured[1]),
            )
            DeepSeekModelClient._shared_cache_indexes.clear()


class TaskSelectionTests(unittest.TestCase):
    def _experiment(self, root: Path, dataset_spec: dict, sampling: dict | None = None) -> ExperimentSpec:
        return ExperimentSpec(
            path=root / "exp.yaml",
            experiment_id="sel",
            raw={
                "id": "sel",
                "datasets": [dataset_spec],
                "sampling": sampling or {},
                "seeds": [0, 1, 2],
            },
            benchmark=_benchmark(root),
            config_hash="x",
        )

    def test_seeded_random_is_stable_across_run_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_file = root / "tasks.jsonl"
            _write_tasks(task_file, [_task_payload(f"hle/{idx}") for idx in range(10)])
            spec = {
                "name": "hle",
                "task_file": "tasks.jsonl",
                "max_tasks": 3,
                "task_selection": "seeded_random",
                "selection_seed": 123,
            }
            first = [task["task_id"] for task in select_tasks(self._experiment(root, spec), seed=0)]
            second = [task["task_id"] for task in select_tasks(self._experiment(root, spec), seed=99)]
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertEqual(len(set(first)), 3)

    def test_unknown_strategy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_file = root / "tasks.jsonl"
            _write_tasks(task_file, [_task_payload("hle/0")])
            spec = {
                "name": "hle",
                "task_file": "tasks.jsonl",
                "max_tasks": 1,
                "task_selection": "coin_flip",
            }
            with self.assertRaises(ValueError) as ctx:
                select_tasks(self._experiment(root, spec))
            self.assertIn("coin_flip", str(ctx.exception))

    def test_deterministic_first_keeps_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tasks(root / "tasks.jsonl", [_task_payload(f"hle/{idx}") for idx in range(5)])
            spec = {
                "name": "hle",
                "task_file": "tasks.jsonl",
                "max_tasks": 2,
                "task_selection": "deterministic_first",
            }
            ids = [task["task_id"] for task in select_tasks(self._experiment(root, spec))]
            self.assertEqual(ids, ["hle/0", "hle/1"])


class NullTerminalTests(unittest.TestCase):
    def test_null_terminal_is_empty_when_fallback_disabled(self) -> None:
        architecture = controlled_architecture(
            {
                "roles": ["planner", "finalizer"],
                "edges": [["planner", "finalizer"], ["finalizer", "final_answer"]],
                "orchestration": {},
            }
        )
        agents = {
            "planner": BaseAgent("planner", "planner", "p", {}),
            "finalizer": BaseAgent("finalizer", "finalizer", "f", {}),
        }
        result = MASGraphBuilder(
            architecture,
            agents,
            removed_agents={"finalizer"},
            null_replacement=True,
        ).invoke({"task": {"task_id": "t", "dataset": "hle", "prompt": "q"}, "messages": []})
        self.assertEqual(result.final_answer, "")
        self.assertNotIn("null agent replacement", result.final_answer)

    def test_explicit_fallback_still_uses_upstream_when_enabled(self) -> None:
        architecture = controlled_architecture(
            {
                "roles": ["planner", "finalizer"],
                "edges": [["planner", "finalizer"], ["finalizer", "final_answer"]],
                "orchestration": {"fallback_final_answer": {"enabled": True}},
            }
        )
        planner = BaseAgent("planner", "planner", "p", {})
        planner.invoke = lambda state: type(  # type: ignore[method-assign]
            "Out",
            (),
            {
                "agent_id": "planner",
                "role": "planner",
                "content": "PLAN_ARTIFACT",
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_calls": 0,
                "metadata": {},
            },
        )()
        result = MASGraphBuilder(
            architecture,
            {"planner": planner, "finalizer": BaseAgent("finalizer", "finalizer", "f", {})},
            removed_agents={"finalizer"},
            null_replacement=True,
        ).invoke({"task": {"task_id": "t", "dataset": "hle", "prompt": "q"}, "messages": []})
        self.assertEqual(result.final_answer, "PLAN_ARTIFACT")


class ControlledTopologyFallbackTests(unittest.TestCase):
    def test_controlled_topology_variants_do_not_acquire_fallback(self) -> None:
        variant = {
            "id": "controlled_chain",
            "template": "chain",
            "description": "linear",
            "edges": [["planner", "finalizer"], ["finalizer", "final_answer"]],
        }
        roles = ["planner", "finalizer"]
        cloned = _clone_architecture(
            controlled_architecture({"roles": roles, "edges": []}, architecture_id="A1"),
            variant,
            roles,
        )
        self.assertNotIn("fallback_final_answer", cloned.orchestration)
        self.assertFalse(cloned.orchestration.get("fallback_final_answer", {}).get("enabled", False))

        row = _graph_feature_row("controlled_chain", variant, roles)
        self.assertNotIn("fallback_final_answer_policy", row)
        self.assertNotIn("fallback", json.dumps(row).lower())

        experiment = SimpleNamespace(
            raw={
                "controlled_role_set": roles,
                "topology_variants": [variant],
            },
            benchmark=SimpleNamespace(
                architectures={
                    "A1": controlled_architecture({"roles": roles, "edges": []}, architecture_id="A1")
                }
            ),
        )
        injected = _inject_topology_variants(experiment)
        self.assertEqual(injected[0][0], "controlled_chain")
        injected_arch = experiment.benchmark.architectures["controlled_chain"]
        self.assertNotIn("fallback_final_answer", injected_arch.orchestration)
        self.assertEqual(cloned.orchestration.get("max_rounds"), 1)
        self.assertNotEqual(cloned.orchestration.get("execution_mode"), "debate")

        star_variant = {
            "id": "controlled_star",
            "template": "star",
            "description": "star",
            "edges": [["finalizer", "planner"], ["planner", "finalizer"], ["finalizer", "final_answer"]],
        }
        cloned_star = _clone_architecture(
            controlled_architecture({"roles": ["planner", "finalizer"], "edges": []}, architecture_id="A1"),
            star_variant,
            ["planner", "finalizer"],
        )
        self.assertEqual(cloned_star.orchestration.get("max_rounds"), 1)
        self.assertEqual(cloned_star.orchestration.get("execution_mode"), "single_pass")
        self.assertNotEqual(cloned_star.orchestration.get("execution_mode"), "debate")

        agents = {
            "planner": BaseAgent("planner", "planner", "p", {}),
            "finalizer": BaseAgent("finalizer", "finalizer", "f", {}),
        }
        result = MASGraphBuilder(
            injected_arch,
            agents,
            removed_agents={"finalizer"},
            null_replacement=True,
        ).invoke({"task": {"task_id": "t", "dataset": "hle", "prompt": "q"}, "messages": []})
        self.assertEqual(result.final_answer, "")


class PermissionAllowlistTests(unittest.TestCase):
    def test_allowlisted_toggles_are_accepted(self) -> None:
        self.assertEqual(
            _permission_level_overrides({"id": "write", "toggle": "write_solution"}, False),
            {"write_solution": False},
        )
        self.assertEqual(
            _permission_level_overrides({"id": "final", "toggle": "final_answer"}, True),
            {"final_answer": True},
        )

    def test_non_allowlisted_toggles_raise(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _permission_level_overrides(
                {"id": "mem", "toggles": ["read_memory", "write_memory"]},
                "none",
            )
        self.assertIn("read_memory", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _permission_level_overrides(
                {"id": "tools", "toggles": ["call_tools", "run_tests"]},
                False,
            )
        message = str(ctx.exception)
        self.assertIn("call_tools", message)
        self.assertIn("write_solution", message)

    def test_configured_exp07_toggles_fail_before_task_work(self) -> None:
        body = "\n".join(
            [
                "id: exp07_permission_intervention",
                "permission_interventions:",
                "- id: verifier_test_access",
                "  role: verifier",
                "  toggles:",
                "  - call_tools",
                "  - run_tests",
                "  levels:",
                "  - false",
                "  - true",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exp07.yaml"
            path.write_text(body, encoding="utf-8")
            with patch(
                "mas_contribution_bench.runners.run_intervention.select_tasks",
                side_effect=AssertionError("should not select tasks"),
            ):
                with self.assertRaises(ValueError) as ctx:
                    run_permission_intervention(path)
            message = str(ctx.exception)
            self.assertIn("call_tools", message)
            self.assertIn("write_solution", message)


class LlmErrorFallbackTests(unittest.TestCase):
    def test_vllm_timeout_raises_by_default(self) -> None:
        client = OpenAICompatibleModelClient(
            api_key="dummy",
            base_url="http://127.0.0.1:9/v1",
            default_model="qwen-local",
            max_retries=0,
            retry_backoff_seconds=0,
        )
        env = {"MAS_LLM_CACHE_ENABLED": "0"}
        with patch.dict(os.environ, env, clear=True):
            self.assertNotIn("MAS_LLM_ERROR_FALLBACK", os.environ)
            self.assertFalse(client._llm_error_fallback_enabled())
            with patch("requests.post", side_effect=requests.Timeout("timed out")):
                with self.assertRaises(RuntimeError) as ctx:
                    client.complete([{"role": "user", "content": "hi"}], max_tokens=8)
        self.assertIn("failed", str(ctx.exception).lower())

    def test_timeout_fallback_only_when_opted_in(self) -> None:
        client = OpenAICompatibleModelClient(
            api_key="dummy",
            base_url="http://127.0.0.1:9/v1",
            default_model="qwen-local",
            max_retries=0,
            retry_backoff_seconds=0,
        )
        env = {"MAS_LLM_ERROR_FALLBACK": "1", "MAS_LLM_CACHE_ENABLED": "0"}
        with patch.dict(os.environ, env, clear=False):
            self.assertTrue(client._llm_error_fallback_enabled())
            with patch("requests.post", side_effect=requests.Timeout("timed out")):
                content = client.complete([{"role": "user", "content": "hi"}], max_tokens=8)
        payload = json.loads(content)
        self.assertIn("llm_request_failed", payload["failure_modes"])


if __name__ == "__main__":
    unittest.main()
