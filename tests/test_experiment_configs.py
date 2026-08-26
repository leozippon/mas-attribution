"""Offline contract tests for the canonical Exp01-Exp09 suite."""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
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
    load_experiment_spec,
)
from mas_contribution_bench.runners.common import build_model_client
from mas_contribution_bench.utils.io import load_yaml


CANONICAL_IDS = (
    "exp01_full_system",
    "exp02_single_agent_baseline",
    "exp03_loo_attribution",
    "exp04_shapley_attribution",
    "exp05_topology_intervention",
    "exp06_role_intervention",
    "exp07_permission_intervention",
    "exp08_generalization",
    "exp09_contribution_predictor",
)
ALL_SEVEN = (
    "aime_2026",
    "gpqa_diamond",
    "hle",
    "arc_agi_2",
    "ifbench",
    "livecodebench",
    "swebench_verified",
)
LOCAL_FIVE = ALL_SEVEN[:5]
BANNED_TOKENS = ("humaneval", "mbpp", "deepseek")
BANNED_EXP08_AXES = (
    "single_agent_success_bin",
    "full_mas_success_bin",
    "dominant_failure_type",
)


def _load_run_experiment():
    path = ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_suite():
    path = ROOT / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_names(raw: dict) -> list[str]:
    return [str(item.get("name")) for item in raw.get("datasets") or []]


def _methods(raw: dict) -> list[str]:
    attribution = raw.get("attribution") or {}
    if "methods" in attribution:
        return [str(item) for item in attribution.get("methods") or []]
    if attribution.get("method"):
        return [str(attribution.get("method"))]
    return []


class CanonicalExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_exp = _load_run_experiment()
        cls.run_suite = _load_run_suite()
        cls.exp_dir = ROOT / "configs" / "experiments"
        cls.raw_by_id = {}
        for path in sorted(cls.exp_dir.glob("*.yaml")):
            raw = load_yaml(path)
            cls.raw_by_id[str(raw["id"])] = (path, raw)

    def test_exact_nine_files_and_aliases(self) -> None:
        names = tuple(sorted(path.name for path in self.exp_dir.glob("*.yaml")))
        expected_files = tuple(f"{item}.yaml" for item in CANONICAL_IDS)
        self.assertEqual(names, expected_files)
        self.assertEqual(tuple(self.run_exp.EXPERIMENT_ALIASES), CANONICAL_IDS)
        for experiment_id, relative in self.run_exp.EXPERIMENT_ALIASES.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(path.name, f"{experiment_id}.yaml")
            self.assertEqual(load_yaml(path)["id"], experiment_id)

    def test_dataset_sets_order_max_tasks_and_seed(self) -> None:
        for experiment_id in CANONICAL_IDS[:7]:
            _path, raw = self.raw_by_id[experiment_id]
            expected = ALL_SEVEN if experiment_id in {"exp01_full_system", "exp02_single_agent_baseline"} else LOCAL_FIVE
            names = _dataset_names(raw)
            self.assertEqual(names, list(expected), experiment_id)
            sampling = raw.get("sampling") or {}
            self.assertEqual(sampling.get("selection_seed"), 0, experiment_id)
            self.assertEqual(sampling.get("task_selection"), "seeded_random", experiment_id)
            for dataset in raw.get("datasets") or []:
                self.assertEqual(dataset.get("max_tasks"), 30, dataset)
                self.assertEqual(dataset.get("task_selection"), "seeded_random", dataset)

        exp08 = self.raw_by_id["exp08_generalization"][1]
        self.assertEqual(_dataset_names(exp08), list(LOCAL_FIVE))
        self.assertEqual((exp08.get("sampling") or {}).get("selection_seed"), 0)

    def test_no_humaneval_mbpp_or_deepseek_in_canonical_configs(self) -> None:
        for experiment_id, (path, _raw) in self.raw_by_id.items():
            text = path.read_text(encoding="utf-8").lower()
            for token in BANNED_TOKENS:
                self.assertNotIn(token, text, f"{experiment_id} contains {token}")

    def test_qwen_endpoint_metadata_for_model_generating_configs(self) -> None:
        for experiment_id in CANONICAL_IDS[:7]:
            raw = self.raw_by_id[experiment_id][1]
            model = raw.get("model") or {}
            self.assertEqual(model.get("backend"), "vllm", experiment_id)
            self.assertEqual(model.get("name"), "qwen-local", experiment_id)
            self.assertEqual(model.get("served_model_name"), "qwen-local", experiment_id)
            self.assertEqual(model.get("base_url"), "http://127.0.0.1:8000/v1", experiment_id)
            self.assertEqual(model.get("api_key"), "dummy", experiment_id)
            self.assertEqual(model.get("weights_path"), "/Data/public/Qwen3.8-27B-FP8", experiment_id)
            self.assertEqual(model.get("context_length"), 262144, experiment_id)
            self.assertEqual(model.get("tensor_parallel_size"), 2, experiment_id)
            self.assertEqual(model.get("max_num_seqs"), 2, experiment_id)
            self.assertEqual(model.get("managed_by"), "zhangqy", experiment_id)
            self.assertTrue(model.get("managed_externally"), experiment_id)
            self.assertEqual(model.get("max_tokens"), 2048, experiment_id)
            blob = " ".join(str(value) for value in model.values()).lower()
            self.assertNotIn("vllm serve", blob)
            self.assertNotIn("launch", blob)

    def test_dead_config_fields_are_absent(self) -> None:
        exp01 = self.raw_by_id["exp01_full_system"][1]
        self.assertNotIn("compatibility_policy", exp01.get("architectures") or {})
        self.assertNotIn("feature_file", exp01.get("outputs") or {})
        for dataset in exp01.get("datasets") or []:
            self.assertNotIn("attribution_eligible", dataset)

        exp02 = self.raw_by_id["exp02_single_agent_baseline"][1]
        self.assertNotIn("single_agent_metadata_file", exp02.get("outputs") or {})
        self.assertNotIn("difficulty_file", exp02.get("outputs") or {})
        for dataset in exp02.get("datasets") or []:
            self.assertNotIn("attribution_eligible", dataset)

        exp03 = self.raw_by_id["exp03_loo_attribution"][1]
        self.assertNotIn("ranking_file", exp03.get("outputs") or {})

        exp04 = self.raw_by_id["exp04_shapley_attribution"][1]
        self.assertNotIn("analysis", exp04)
        self.assertNotIn("comparison_file", exp04.get("outputs") or {})

        exp05 = self.raw_by_id["exp05_topology_intervention"][1]
        self.assertNotIn("analysis", exp05)
        self.assertNotIn("compare_with", str(exp05))
        star = next(item for item in exp05.get("topology_variants") or [] if item.get("id") == "controlled_star")
        star_text = str(star.get("description") or "").lower()
        self.assertIn("not hub dispatch", star_text)
        self.assertIn("not", star_text)
        self.assertNotIn("communicate through", star_text)
        self.assertIn("single-pass", star_text)
        self.assertIn("finalizer last", star_text)

        exp06 = self.raw_by_id["exp06_role_intervention"][1]
        self.assertNotIn("effect_targets", exp06.get("analysis") or {})
        exp07 = self.raw_by_id["exp07_permission_intervention"][1]
        self.assertNotIn("effect_targets", exp07.get("analysis") or {})

        exp08 = self.raw_by_id["exp08_generalization"][1]
        self.assertNotIn("research_questions", exp08)
        self.assertNotIn("hypotheses", exp08)

    def test_exp05_variants_loo_and_no_fallback_or_debate(self) -> None:
        raw = self.raw_by_id["exp05_topology_intervention"][1]
        variant_ids = [item["id"] for item in raw.get("topology_variants") or []]
        self.assertEqual(variant_ids, ["controlled_chain", "controlled_star", "controlled_dag"])
        self.assertEqual(_methods(raw), ["loo"])
        blob = str(raw).lower()
        self.assertNotIn("controlled_debate", blob)
        self.assertNotIn("fallback_final_answer", blob)
        self.assertNotIn("paired_test", blob)
        self.assertNotIn("mixed_effects", blob)

    def test_exp06_architectures_swap_and_loo(self) -> None:
        raw = self.raw_by_id["exp06_role_intervention"][1]
        self.assertEqual(raw.get("base_architectures"), ["A2_chain", "A3_dag", "A5_star"])
        swaps = raw.get("role_swaps") or []
        self.assertEqual([item.get("id") for item in swaps], ["planner_coder_swap"])
        self.assertEqual(swaps[0].get("swap"), ["planner", "coder"])
        self.assertIn("same_position_permissions", swaps[0].get("controls") or [])
        self.assertEqual(_methods(raw), ["loo"])
        self.assertNotIn("coder_verifier_swap", str(raw))
        self.assertNotIn("mixed_effects", str(raw).lower())
        description = " ".join(
            [
                str(raw.get("purpose") or ""),
                str(swaps[0].get("description") or ""),
                str((raw.get("analysis") or {}).get("notes") or ""),
            ]
        ).lower()
        self.assertIn("position", description)
        self.assertIn("permission", description)

    def test_exp07_architectures_permissions_strict_loo(self) -> None:
        raw = self.raw_by_id["exp07_permission_intervention"][1]
        self.assertEqual(raw.get("base_architectures"), ["A2_chain", "A3_dag"])
        interventions = raw.get("permission_interventions") or []
        self.assertEqual(
            [(item.get("id"), item.get("toggle")) for item in interventions],
            [
                ("coder_write_access", "write_solution"),
                ("finalizer_final_authority", "final_answer"),
            ],
        )
        for item in interventions:
            self.assertNotIn("toggles", item)
        self.assertEqual((raw.get("permission_enforcement") or {}).get("mode"), "strict")
        self.assertEqual(_methods(raw), ["loo"])
        blob = str(raw).lower()
        self.assertNotIn("call_tools", blob)
        self.assertNotIn("run_tests", blob)
        self.assertNotIn("mixed_effects", blob)

    def test_exp08_descriptive_and_no_score_context(self) -> None:
        raw = self.raw_by_id["exp08_generalization"][1]
        analysis = raw.get("analysis") or {}
        inputs = raw.get("inputs") or {}
        self.assertTrue(analysis.get("analysis_only"))
        self.assertTrue(analysis.get("descriptive_only"))
        self.assertFalse(analysis.get("causal_claims"))
        self.assertFalse(analysis.get("inferential_tests"))
        self.assertEqual((raw.get("model") or {}).get("backend"), "analysis_only")
        self.assertNotIn("score_files", inputs)
        self.assertNotIn("task_metadata_file", inputs)
        self.assertNotIn("task_metadata_file", analysis)
        for axis in BANNED_EXP08_AXES:
            self.assertNotIn(axis, analysis.get("task_condition_axes") or [])
        self.assertEqual(
            inputs.get("attribution_files"),
            [
                "data/results/exp03/loo_attribution.jsonl",
                "data/results/exp04/shapley_banzhaf_attribution.jsonl",
                "data/results/exp05/attribution.jsonl",
                "data/results/exp06/attribution.jsonl",
                "data/results/exp07/attribution.jsonl",
            ],
        )
        self.assertEqual(
            inputs.get("trace_files"),
            [
                "data/traces/exp03/exp03_loo_attribution_traces.jsonl",
                "data/traces/exp04/exp04_shapley_attribution_traces.jsonl",
                "data/traces/exp05/exp05_topology_intervention_traces.jsonl",
                "data/traces/exp06/exp06_role_intervention_traces.jsonl",
                "data/traces/exp07/exp07_permission_intervention_traces.jsonl",
            ],
        )

    def test_exp09_is_refused_and_not_executable(self) -> None:
        raw = self.raw_by_id["exp09_contribution_predictor"][1]
        self.assertEqual(raw.get("status"), "unimplemented")
        self.assertNotIn("model", raw)
        self.assertNotIn("models", raw)
        self.assertNotIn("splits", raw)
        for mode in ("auto", "full", "single", "attribution", "intervention"):
            with self.assertRaises(SystemExit) as ctx:
                self.run_exp.infer_runner("exp09_contribution_predictor", mode)
            self.assertIn("no runner", str(ctx.exception))

    def test_outputs_are_disjoint_for_suite_pairs(self) -> None:
        pairable = [item for item in CANONICAL_IDS if item not in {"exp08_generalization", "exp09_contribution_predictor"}]
        for left, right in itertools.combinations(pairable, 2):
            plan = self.run_suite.build_plan([left, right], jobs=2, max_tasks=None)
            self.assertEqual(plan["jobs"], 2)
            ids = [child["experiment_id"] for child in plan["experiments"]]
            self.assertEqual(ids, [left, right])
        with self.assertRaises(SystemExit):
            self.run_suite.build_plan(["exp01_full_system", "exp08_generalization"], jobs=2, max_tasks=None)
        with self.assertRaises(SystemExit):
            self.run_suite.build_plan(["exp09_contribution_predictor"], jobs=1, max_tasks=None)
        for experiment_id, (_path, raw) in self.raw_by_id.items():
            outputs = raw.get("outputs") or {}
            if "score_file" in outputs and "evaluation_file" in outputs:
                self.assertNotEqual(
                    outputs["score_file"],
                    outputs["evaluation_file"],
                    experiment_id,
                )

    def test_a1_to_a4_recommended_datasets_are_canonical_seven(self) -> None:
        architecture_dir = ROOT / "configs" / "benchmark_specs" / "architectures"
        for name in ("A1_pev.yaml", "A2_chain.yaml", "A3_dag.yaml", "A4_metagpt_lite.yaml"):
            raw = load_yaml(architecture_dir / name)
            self.assertEqual(raw.get("recommended_datasets"), list(ALL_SEVEN), name)

    def test_a5_a6_a7_are_single_pass(self) -> None:
        architecture_dir = ROOT / "configs" / "benchmark_specs" / "architectures"
        for name in ("A5_star.yaml", "A6_debate.yaml", "A7_graph.yaml"):
            raw = load_yaml(architecture_dir / name)
            orchestration = raw.get("orchestration") or {}
            self.assertEqual(orchestration.get("max_rounds"), 1, name)
            blob = " ".join(
                [
                    str(raw.get("description") or ""),
                    str(orchestration.get("notes") or ""),
                    str(orchestration.get("routing_policy") or ""),
                ]
            ).lower()
            self.assertIn("single", blob)
            self.assertNotIn("multi-round", blob)


class ModelApiKeyHonorTests(unittest.TestCase):
    def test_build_model_client_honors_config_api_key(self) -> None:
        experiment = ExperimentSpec(
            path=ROOT / "configs" / "experiments" / "exp01_full_system.yaml",
            experiment_id="exp01_full_system",
            raw={
                "model": {
                    "backend": "vllm",
                    "name": "qwen-local",
                    "api_key": "from-config",
                    "base_url": "http://127.0.0.1:8000/v1",
                }
            },
            benchmark=BenchmarkSpec(
                project_root=ROOT,
                architecture_dir=ROOT / "configs" / "benchmark_specs" / "architectures",
                agent_dir=ROOT / "configs" / "benchmark_specs" / "agents",
                prompt_dir=ROOT / "configs" / "benchmark_specs" / "prompts",
                permission_file=ROOT / "configs" / "benchmark_specs" / "permissions" / "permission_sets.yaml",
                permissions={},
                agents={},
                architectures={},
            ),
            config_hash="test",
        )
        env = {
            "MAS_MODEL_BACKEND": "",
            "OPENAI_API_KEY": "",
            "VLLM_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            client = build_model_client(experiment)
        self.assertEqual(getattr(client, "api_key"), "from-config")


class PrintConfigSmokeTests(unittest.TestCase):
    def test_canonical_configs_load(self) -> None:
        for experiment_id in CANONICAL_IDS:
            spec = load_experiment_spec(
                ROOT / "configs" / "experiments" / f"{experiment_id}.yaml",
                ROOT,
            )
            self.assertEqual(spec.experiment_id, experiment_id)


if __name__ == "__main__":
    unittest.main()
