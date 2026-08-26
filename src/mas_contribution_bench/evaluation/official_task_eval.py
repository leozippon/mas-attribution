"""Official-style evaluators for deterministic non-code benchmark tasks.

These functions cover benchmarks whose official score is exact answer,
exact grid match, or AllenAI IFBench strict instruction following.
Benchmarks that need a workspace or an interactive environment are handled by
external harness scripts.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

from mas_contribution_bench.data.external import extract_arc_expected_outputs, resolve_gpqa_gold
from mas_contribution_bench.data.schemas import FailureType


def unwrap_answer(text: str | None) -> str:
    """Extract the most likely final answer artifact from a model response."""
    if not text:
        return ""
    stripped = str(text).strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            candidate = re.sub(r"^(json|python|text|answer|csv|yaml)\s*", "", candidate, flags=re.I).strip()
            nested = unwrap_answer(candidate)
            if nested:
                return nested
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in (
                "answer",
                "final_answer",
                "prediction",
                "choice",
                "artifact",
                "solution",
                "output",
                "outputs",
            ):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return unwrap_answer(value)
                if value is not None and key in {"answer", "prediction", "output", "outputs"}:
                    return json.dumps(value, ensure_ascii=False)
    return stripped


def normalize_text(text: str | None) -> str:
    value = unwrap_answer(text).lower().strip()
    value = value.replace("\u2212", "-").replace("−", "-")
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"\s+", " ", value)
    return value.strip(string.whitespace + string.punctuation)


def extract_integer(text: str | None) -> str:
    value = unwrap_answer(text)
    matches = re.findall(r"(?<![\d.])-?\d+(?![\d.])", value)
    return matches[-1] if matches else normalize_text(value)


def exact_answer_eval(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    dataset = str(task.get("dataset", "")).lower()
    reference = task.get("reference_solution")
    if dataset == "gpqa_diamond" and (reference is None or str(reference).strip() == ""):
        reference = resolve_gpqa_gold(task)
    if reference is None or str(reference).strip() == "":
        return 0.0, False, FailureType.UNKNOWN, {"evaluator": "official_exact_match", "reason": "missing_reference"}
    if not prediction or not str(prediction).strip():
        return 0.0, False, FailureType.EMPTY_OUTPUT, {"evaluator": "official_exact_match", "reference": str(reference)}
    if dataset == "aime_2026":
        predicted = extract_integer(prediction)
        passed = predicted == str(reference).strip()
        return (1.0 if passed else 0.0), passed, (FailureType.NONE if passed else FailureType.TEST_FAILURE), {
            "evaluator": "official_integer_exact_match",
            "predicted": predicted,
            "reference": str(reference),
        }
    predicted_norm = normalize_text(prediction)
    reference_norm = normalize_text(str(reference))
    passed = predicted_norm == reference_norm
    return (1.0 if passed else 0.0), passed, (FailureType.NONE if passed else FailureType.TEST_FAILURE), {
        "evaluator": "official_normalized_exact_match",
        "predicted": predicted_norm,
        "reference": reference_norm,
    }


def _load_jsonish(text: str | None) -> Any:
    value = unwrap_answer(text)
    candidates = [value]
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", value)
    if match:
        candidates.append(match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _extract_arc_expected(task: dict[str, Any]) -> list[Any]:
    dedicated = extract_arc_expected_outputs(task)
    if dedicated:
        return dedicated
    raw = _load_jsonish(task.get("prompt"))
    if not isinstance(raw, dict):
        return []
    expected = []
    for item in raw.get("test") or []:
        if isinstance(item, dict) and "output" in item:
            expected.append(item["output"])
    return expected


def _extract_arc_prediction(prediction: str | None) -> list[Any]:
    raw = _load_jsonish(prediction)
    if isinstance(raw, dict):
        for key in ("outputs", "output", "answer", "prediction"):
            if key in raw:
                raw = raw[key]
                break
    if raw is None:
        return []
    if isinstance(raw, list):
        if raw and all(isinstance(row, list) and (not row or isinstance(row[0], int)) for row in raw):
            return [raw]
        return raw
    return []


def arc_agi_2_eval(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    expected = _extract_arc_expected(task)
    predicted = _extract_arc_prediction(prediction)
    if not expected:
        return 0.0, False, FailureType.UNKNOWN, {"evaluator": "arc_agi_2_exact_grid", "reason": "missing_expected_outputs"}
    if not predicted:
        return 0.0, False, FailureType.INVALID_FORMAT, {"evaluator": "arc_agi_2_exact_grid", "reason": "prediction_not_json_grid"}
    checks = [idx < len(predicted) and predicted[idx] == answer for idx, answer in enumerate(expected)]
    passed = bool(checks) and all(checks) and len(predicted) >= len(expected)
    return (1.0 if passed else 0.0), passed, (FailureType.NONE if passed else FailureType.TEST_FAILURE), {
        "evaluator": "arc_agi_2_exact_grid",
        "num_expected_outputs": len(expected),
        "num_predicted_outputs": len(predicted),
        "per_output_exact": checks,
    }


IFBENCH_PACKAGE_ERROR = (
    "Official AllenAI IFBench verifier is unavailable. "
    "Install the pinned project dependency ifbench==0.2.0."
)


def load_ifbench_instructions_registry() -> Any:
    """Import AllenAI IFBench's official instruction registry.

    Import is deferred so offline tests can mock the registry without installing
    the package. Runtime IFBench evaluation must call this after empty-output
    handling.
    """
    try:
        from ifbench import instructions_registry  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(IFBENCH_PACKAGE_ERROR) from exc
    if not hasattr(instructions_registry, "INSTRUCTION_DICT"):
        raise RuntimeError(
            "Official AllenAI IFBench verifier is invalid: "
            "instructions_registry.INSTRUCTION_DICT is missing."
        )
    return instructions_registry


def _ifbench_kwargs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if value is not None}


def _ifbench_instruction_ids(task: dict[str, Any]) -> list[str]:
    metadata_raw = task.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    raw_ids = metadata.get("instruction_id_list")
    if raw_ids is None:
        raw_ids = task.get("instruction_id_list")
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        return [raw_ids] if raw_ids.strip() else []
    if isinstance(raw_ids, (list, tuple)):
        return [str(item) for item in raw_ids if item not in (None, "")]
    return []


def _ifbench_kwargs_list(task: dict[str, Any], count: int) -> list[dict[str, Any]]:
    metadata_raw = task.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    raw = metadata.get("kwargs")
    if raw is None:
        raw = task.get("kwargs")
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    if len(items) != count or any(not isinstance(item, dict) for item in items):
        raise ValueError(
            "IFBench task metadata must provide one kwargs object per instruction ID; "
            f"got {len(items)} kwargs entries for {count} instruction IDs."
        )
    return [_ifbench_kwargs(item) for item in items]


def _check_ifbench_instruction(
    registry: Any,
    instruction_id: str,
    kwargs: dict[str, Any],
    answer: str,
    prompt: str,
) -> dict[str, Any]:
    registry_dict = registry.INSTRUCTION_DICT
    if instruction_id not in registry_dict:
        raise ValueError(
            f"Unknown IFBench instruction ID {instruction_id!r}. "
            "Official AllenAI IFBench verifier has no checker for this ID; "
            "unsupported rules are not scored as zero."
        )
    instruction_cls = registry_dict[instruction_id]
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    passed = bool(instruction.check_following(answer))
    return {"instruction_id": instruction_id, "passed": passed, "kwargs": kwargs}


def ifbench_eval(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    answer = unwrap_answer(prediction)
    if not str(answer).strip():
        return 0.0, False, FailureType.EMPTY_OUTPUT, {
            "evaluator": "allenai_ifbench_strict",
            "verification": "official AllenAI IFBench strict verification",
            "reason": "empty_output",
        }
    registry = load_ifbench_instructions_registry()
    ids = _ifbench_instruction_ids(task)
    if not ids:
        raise ValueError(
            "IFBench task metadata has no instruction IDs; "
            "official AllenAI IFBench strict verification cannot run."
        )
    kwargs_list = _ifbench_kwargs_list(task, len(ids))
    prompt = str(task.get("prompt") or "")
    details = [
        _check_ifbench_instruction(registry, str(instruction_id), kwargs_list[idx], answer, prompt)
        for idx, instruction_id in enumerate(ids)
    ]
    followed = sum(1 for item in details if item["passed"])
    total = len(details)
    instruction_level_rate = followed / total
    passed = followed == total
    score = 1.0 if passed else 0.0
    return score, passed, (FailureType.NONE if passed else FailureType.TEST_FAILURE), {
        "evaluator": "allenai_ifbench_strict",
        "verification": "official AllenAI IFBench strict verification",
        "strict": True,
        "prompt_level_passed": passed,
        "instruction_level_rate": instruction_level_rate,
        "followed_instructions": followed,
        "total_instructions": total,
        "details": details,
    }


def evaluate_official_answer_task(
    task: dict[str, Any],
    prediction: str | None,
) -> tuple[float, bool, FailureType, dict[str, Any]] | None:
    dataset = str(task.get("dataset", "")).lower()
    if dataset in {"aime_2026", "gpqa_diamond", "hle"}:
        return exact_answer_eval(task, prediction)
    if dataset == "arc_agi_2":
        return arc_agi_2_eval(task, prediction)
    if dataset == "ifbench":
        return ifbench_eval(task, prediction)
    return None
