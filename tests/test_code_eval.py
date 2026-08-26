"""Regression tests for unit-test vs text evaluation guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unittest.mock import patch

from mas_contribution_bench.data.schemas import EvaluatorType, FailureType
from mas_contribution_bench.evaluation import official_task_eval
from mas_contribution_bench.evaluation.code_eval import evaluate_task_output
from mas_contribution_bench.evaluation.official_task_eval import ifbench_eval


class _MockIfbenchInstruction:
    def __init__(self, instruction_id: str):
        self.instruction_id = instruction_id
        self.kwargs: dict = {}

    def build_description(self, **kwargs):
        self.kwargs = dict(kwargs)

    def get_instruction_args(self):
        if self.instruction_id == "combination:repeat_prompt":
            return {"prompt": None}
        return dict(self.kwargs)

    def check_following(self, value: str) -> bool:
        if self.instruction_id == "punctuation:no_comma":
            return "," not in value and "，" not in value
        if self.instruction_id == "count:numbers":
            target = int(self.kwargs.get("num_numbers") or 0)
            return len([token for token in value.split() if token.isdigit()]) >= target
        if self.instruction_id == "combination:repeat_prompt":
            return self.kwargs == {"prompt": value}
        return False


class _MockIfbenchRegistry:
    INSTRUCTION_DICT = {
        "punctuation:no_comma": _MockIfbenchInstruction,
        "count:numbers": _MockIfbenchInstruction,
        "combination:repeat_prompt": _MockIfbenchInstruction,
    }


class EvaluateTaskOutputGuardTests(unittest.TestCase):
    def test_unit_test_without_execution_raises_before_record(self) -> None:
        task = {
            "task_id": "humaneval/0",
            "dataset": "humaneval",
            "reference_solution": "def add(a, b):\n    return a + b\n",
            "evaluation": {"evaluator_type": "unit_test", "metric": "pass_at_1"},
        }
        with self.assertRaises(ValueError) as ctx:
            evaluate_task_output(
                "run-1",
                task,
                "A1_pev",
                task["reference_solution"],
                cost={},
                execute_code=False,
            )
        message = str(ctx.exception)
        self.assertIn("unit_test", message)
        self.assertIn("execute-code", message)
        self.assertNotIn("EvaluationRecord(", message)

        enum_task = dict(task)
        enum_task["evaluation"] = {
            "evaluator_type": EvaluatorType.UNIT_TEST,
            "metric": "pass_at_1",
        }
        with self.assertRaises(ValueError):
            evaluate_task_output(
                "run-1b",
                enum_task,
                "A1_pev",
                "not even the reference",
                cost={},
                execute_code=False,
            )

    def test_text_exact_match_still_scores_without_execution(self) -> None:
        task = {
            "task_id": "gpqa/1",
            "dataset": "gpqa_diamond",
            "reference_solution": "Paris",
            "evaluation": {"evaluator_type": "exact_match", "metric": "exact_match"},
        }
        passed = evaluate_task_output(
            "run-2",
            task,
            "A1_pev",
            "Paris",
            cost={},
            execute_code=False,
        )
        self.assertEqual(passed.score, 1.0)
        self.assertTrue(passed.passed)
        self.assertEqual(passed.metric, "exact_match")
        self.assertEqual(passed.failure_type, FailureType.NONE)

        failed = evaluate_task_output(
            "run-3",
            task,
            "A1_pev",
            "London",
            cost={},
            execute_code=False,
        )
        self.assertEqual(failed.score, 0.0)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.metric, "exact_match")

    def test_official_answer_datasets_score_without_code_execution(self) -> None:
        cases = [
            (
                {
                    "task_id": "aime_2026/1",
                    "dataset": "aime_2026",
                    "reference_solution": "42",
                    "evaluation": {"evaluator_type": "exact_match", "metric": "exact_match"},
                },
                "The integer is 42",
                1.0,
            ),
            (
                {
                    "task_id": "gpqa_diamond/1",
                    "dataset": "gpqa_diamond",
                    "reference_solution": None,
                    "metadata": {"Correct Answer": "Helium"},
                    "evaluation": {"evaluator_type": "exact_match", "metric": "multiple_choice_accuracy"},
                },
                "Helium",
                1.0,
            ),
            (
                {
                    "task_id": "hle/1",
                    "dataset": "hle",
                    "reference_solution": "Mitochondria",
                    "evaluation": {"evaluator_type": "exact_match", "metric": "answer_accuracy"},
                },
                "mitochondria",
                1.0,
            ),
            (
                {
                    "task_id": "arc_agi_2/1",
                    "dataset": "arc_agi_2",
                    "prompt": '{"train": [], "test": [{"input": [[1]]}]}',
                    "metadata": {"expected_test_outputs": [[[2, 2]]]},
                    "evaluation": {"evaluator_type": "exact_match", "metric": "grid_exact_match"},
                },
                "[[2, 2]]",
                1.0,
            ),
            (
                {
                    "task_id": "ifbench/1",
                    "dataset": "ifbench",
                    "prompt": "Write a reply.",
                    "metadata": {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
                    "evaluation": {"evaluator_type": "custom", "metric": "instruction_following_score"},
                },
                "hello world",
                1.0,
            ),
        ]
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            for task, prediction, expected_score in cases:
                with self.subTest(dataset=task["dataset"]):
                    record = evaluate_task_output(
                        f"run-{task['dataset']}",
                        task,
                        "A1_pev",
                        prediction,
                        cost={},
                        execute_code=False,
                    )
                    self.assertEqual(record.score, expected_score)
                    self.assertTrue(record.passed)
                    record_exec = evaluate_task_output(
                        f"run-{task['dataset']}-exec",
                        task,
                        "A1_pev",
                        prediction,
                        cost={},
                        execute_code=True,
                    )
                    self.assertEqual(record_exec.score, expected_score)
                    self.assertTrue(record_exec.passed)

    def test_arc_falls_back_to_legacy_prompt_outputs(self) -> None:
        task = {
            "task_id": "arc_agi_2/legacy",
            "dataset": "arc_agi_2",
            "prompt": '{"train": [], "test": [{"input": [[0]], "output": [[9]]}]}',
            "evaluation": {"evaluator_type": "exact_match", "metric": "grid_exact_match"},
        }
        record = evaluate_task_output("run-arc-legacy", task, "A1_pev", "[[9]]", cost={}, execute_code=False)
        self.assertEqual(record.score, 1.0)
        self.assertTrue(record.passed)

    def test_livecodebench_and_swebench_verified_are_unscored(self) -> None:
        for dataset, evaluator_type, metric in (
            ("livecodebench", "unit_test", "pass_at_1"),
            ("swebench_verified", "official_grader", "resolved"),
        ):
            with self.subTest(dataset=dataset):
                task = {
                    "task_id": f"{dataset}/1",
                    "dataset": dataset,
                    "reference_solution": "gold-should-not-match",
                    "evaluation": {"evaluator_type": evaluator_type, "metric": metric},
                }
                record = evaluate_task_output(
                    f"run-{dataset}",
                    task,
                    "A1_pev",
                    "gold-should-not-match",
                    cost={},
                    execute_code=False,
                )
                self.assertIsNone(record.score)
                self.assertIsNone(record.passed)
                self.assertTrue(record.metadata.get("unscored"))
                self.assertEqual(record.metadata.get("scoring"), "not_integrated")
                self.assertEqual(record.raw_evaluator_output.get("status"), "unscored")


class OfficialIfbenchVerifierTests(unittest.TestCase):
    def _task(self, instruction_ids, kwargs=None, prompt="Write a reply."):
        return {
            "task_id": "ifbench/1",
            "dataset": "ifbench",
            "prompt": prompt,
            "metadata": {
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs if kwargs is not None else [{} for _ in instruction_ids],
            },
            "evaluation": {"evaluator_type": "custom", "metric": "instruction_following_score"},
        }

    def test_empty_output_does_not_import_verifier(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            side_effect=AssertionError("should not import verifier"),
        ):
            score, passed, failure, raw = ifbench_eval(self._task(["count:numbers"]), "")
        self.assertEqual(score, 0.0)
        self.assertFalse(passed)
        self.assertEqual(failure, FailureType.EMPTY_OUTPUT)
        self.assertEqual(raw.get("evaluator"), "allenai_ifbench_strict")

    def test_missing_package_raises(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            side_effect=RuntimeError(official_task_eval.IFBENCH_PACKAGE_ERROR),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ifbench_eval(self._task(["punctuation:no_comma"]), "hello world")
        self.assertIn("ifbench==0.2.0", str(ctx.exception))

    def test_missing_instruction_ids_raise(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            with self.assertRaises(ValueError) as ctx:
                ifbench_eval(self._task([]), "hello world")
        self.assertIn("instruction IDs", str(ctx.exception))

    def test_unknown_instruction_id_raises(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            with self.assertRaises(ValueError) as ctx:
                ifbench_eval(self._task(["count:numbers", "not:a_real_id"]), "1 2")
        self.assertIn("not:a_real_id", str(ctx.exception))
        self.assertIn("Unknown IFBench instruction ID", str(ctx.exception))

    def test_kwargs_count_mismatch_raises(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            with self.assertRaises(ValueError) as ctx:
                ifbench_eval(self._task(["count:numbers", "punctuation:no_comma"], [{}]), "1 2")
        self.assertIn("one kwargs object per instruction ID", str(ctx.exception))

    def test_prompt_argument_is_rebuilt_like_official_verifier(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            score, passed, _failure, _raw = ifbench_eval(
                self._task(["combination:repeat_prompt"], [{"ignored": "value"}], prompt="repeat me"),
                "repeat me",
            )
        self.assertEqual(score, 1.0)
        self.assertTrue(passed)

    def test_strict_prompt_level_score_and_instruction_rate(self) -> None:
        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            score, passed, failure, raw = ifbench_eval(
                self._task(["punctuation:no_comma", "count:numbers"], [{}, {"num_numbers": 2}]),
                "hello 1 2",
            )
        self.assertEqual(score, 1.0)
        self.assertTrue(passed)
        self.assertEqual(failure, FailureType.NONE)
        self.assertEqual(raw.get("verification"), "official AllenAI IFBench strict verification")
        self.assertEqual(raw.get("instruction_level_rate"), 1.0)
        self.assertEqual(len(raw.get("details") or []), 2)

        with patch.object(
            official_task_eval,
            "load_ifbench_instructions_registry",
            return_value=_MockIfbenchRegistry,
        ):
            score, passed, failure, raw = ifbench_eval(
                self._task(["punctuation:no_comma", "count:numbers"], [{}, {"num_numbers": 2}]),
                "hello, world",
            )
        self.assertEqual(score, 0.0)
        self.assertFalse(passed)
        self.assertEqual(raw.get("instruction_level_rate"), 0.0)


if __name__ == "__main__":
    unittest.main()
