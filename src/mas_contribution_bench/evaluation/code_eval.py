"""Code-task evaluators for HumanEval and MBPP."""

from __future__ import annotations

import json
import re
import string
from typing import Any

from mas_contribution_bench.data.schemas import CostInfo, EvaluationRecord, EvaluatorType, FailureType
from mas_contribution_bench.evaluation.humaneval_eval import evaluate_humaneval
from mas_contribution_bench.evaluation.mbpp_eval import evaluate_mbpp
from mas_contribution_bench.evaluation.official_task_eval import evaluate_official_answer_task
from mas_contribution_bench.evaluation.sandbox import SandboxResult


def _extract_artifact_or_answer(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[len("json") :].strip()
            if candidate.startswith("python"):
                candidate = candidate[len("python") :].strip()
            extracted = _extract_artifact_or_answer(candidate)
            if extracted:
                return extracted
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "final_answer", "artifact", "solution", "prediction", "choice"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return stripped


def _normalize_answer(text: str | None) -> str:
    value = _extract_artifact_or_answer(text).lower().strip()
    value = value.replace("\u2212", "-").replace("−", "-")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(string.whitespace + string.punctuation)
    return value


def _extract_integer_answer(text: str | None) -> str:
    value = _extract_artifact_or_answer(text)
    matches = re.findall(r"(?<![\d.])-?\d+(?![\d.])", value)
    return matches[-1] if matches else _normalize_answer(value)


def score_text_answer(final_answer: str | None, reference: str | None = None) -> tuple[float, bool, FailureType]:
    if not final_answer or not final_answer.strip():
        return 0.0, False, FailureType.EMPTY_OUTPUT
    if reference and _normalize_answer(final_answer) == _normalize_answer(reference):
        return 1.0, True, FailureType.NONE
    return 0.0, False, FailureType.UNKNOWN


def score_multiple_choice_or_exact(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    dataset = str(task.get("dataset", "")).lower()
    metadata = task.get("metadata") or {}
    reference = task.get("reference_solution")
    if dataset == "gpqa_diamond" and not reference:
        reference = metadata.get("Correct Answer") or metadata.get("Pre-Revision Correct Answer")
    if dataset == "aime_2026" and reference:
        predicted = _extract_integer_answer(prediction)
        passed = predicted == str(reference).strip()
        return (1.0 if passed else 0.0), passed, (FailureType.NONE if passed else FailureType.UNKNOWN), {
            "evaluator": "normalized_integer_exact_match",
            "predicted": predicted,
            "reference": str(reference),
        }
    if reference:
        score, passed, failure_type = score_text_answer(prediction, str(reference))
        return score, passed, failure_type, {
            "evaluator": "normalized_exact_match",
            "reference": str(reference),
        }
    return 0.0, False, FailureType.UNKNOWN, {"evaluator": "text_fallback", "reason": "missing_reference"}


def failure_type_from_sandbox(result: SandboxResult) -> FailureType:
    if result.passed:
        return FailureType.NONE
    if result.timed_out:
        return FailureType.TIMEOUT
    stderr = result.stderr or ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return FailureType.SYNTAX_ERROR
    if "AssertionError" in stderr:
        return FailureType.TEST_FAILURE
    if "Traceback" in stderr:
        return FailureType.RUNTIME_ERROR
    return FailureType.TEST_FAILURE


def evaluate_code_prediction(
    task: dict[str, Any],
    prediction: str | None,
    *,
    sandbox_backend: str = "auto",
) -> tuple[float, bool, FailureType, dict[str, Any]]:
    dataset = str(task.get("dataset", "")).lower()
    timeout = (task.get("evaluation") or {}).get("timeout_seconds")
    try:
        if dataset == "humaneval":
            result = evaluate_humaneval(
                task,
                prediction,
                timeout_seconds=timeout,
                sandbox_backend=sandbox_backend,
            )
        elif dataset == "mbpp":
            result = evaluate_mbpp(
                task,
                prediction,
                timeout_seconds=timeout,
                sandbox_backend=sandbox_backend,
            )
        else:
            official_result = evaluate_official_answer_task(task, prediction)
            if official_result is not None:
                return official_result
            return score_multiple_choice_or_exact(task, prediction)
    except ValueError as exc:
        return 0.0, False, FailureType.INVALID_FORMAT, {"error": str(exc)}
    except Exception as exc:
        return 0.0, False, FailureType.EVALUATOR_ERROR, {"error": repr(exc)}
    failure_type = failure_type_from_sandbox(result)
    return (
        1.0 if result.passed else 0.0,
        result.passed,
        failure_type,
        {
            "backend": result.backend,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "duration_seconds": result.duration_seconds,
            "metadata": result.metadata,
        },
    )


def _evaluation_field(task: dict[str, Any], name: str) -> Any:
    evaluation = task.get("evaluation") or {}
    if isinstance(evaluation, dict):
        return evaluation.get(name)
    return getattr(evaluation, name, None)


def _evaluator_type_name(task: dict[str, Any]) -> str:
    raw = _evaluation_field(task, "evaluator_type")
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def evaluate_task_output(
    run_id: str,
    task: dict[str, Any],
    architecture_id: str,
    final_answer: str | None,
    cost=None,
    execute_code: bool = False,
    sandbox_backend: str = "auto",
) -> EvaluationRecord:
    evaluator_type = _evaluator_type_name(task)
    dataset = str(task.get("dataset", "")).lower()
    official = evaluate_official_answer_task(task, final_answer) if dataset in {
        "aime_2026",
        "gpqa_diamond",
        "hle",
        "arc_agi_2",
        "ifbench",
    } else None
    extra_metadata: dict[str, Any] = {}
    if official is not None:
        score, passed, failure_type, raw_output = official
    elif dataset in {"livecodebench", "swebench_verified"}:
        score, passed, failure_type = None, None, FailureType.NONE
        raw_output = {
            "evaluator": "not_integrated",
            "status": "unscored",
            "reason": "official harness is not integrated in evaluate_task_output",
        }
        extra_metadata = {"scoring": "not_integrated", "unscored": True}
    elif evaluator_type == EvaluatorType.UNIT_TEST.value and not execute_code:
        raise ValueError(
            "Refusing to emit EvaluationRecord: evaluator_type='unit_test' requires code "
            f"execution (metric={_evaluation_field(task, 'metric')!r}). Enable --execute-code; "
            "formal runs should use Docker. Text equality is not pass@1 and will not be used "
            "as a silent fallback."
        )
    elif execute_code:
        score, passed, failure_type, raw_output = evaluate_code_prediction(
            task,
            final_answer,
            sandbox_backend=sandbox_backend,
        )
    else:
        score, passed, failure_type = score_text_answer(final_answer, task.get("reference_solution"))
        raw_output = {"safe_mode": True, "code_execution": False}
    metric = _evaluation_field(task, "metric") or "pass_at_1"
    return EvaluationRecord(
        run_id=run_id,
        task_id=task["task_id"],
        dataset=task["dataset"],
        architecture_id=architecture_id,
        final_answer=final_answer,
        score=score,
        metric=metric,
        passed=passed,
        failure_type=failure_type,
        evaluator=task["evaluation"],
        cost=CostInfo.model_validate(cost or {}),
        raw_evaluator_output=raw_output,
        metadata=extra_metadata,
    )
