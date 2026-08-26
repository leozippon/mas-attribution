"""Regression tests for unit-test vs text evaluation guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.data.schemas import EvaluatorType, FailureType
from mas_contribution_bench.evaluation.code_eval import evaluate_task_output


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


if __name__ == "__main__":
    unittest.main()
