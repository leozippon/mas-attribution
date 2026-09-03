"""Official-style evaluators for deterministic non-code benchmark tasks.

These functions cover benchmarks whose official score is exact answer,
exact grid match, or AllenAI IFBench strict instruction following.
Benchmarks that need a workspace or an interactive environment are handled by
external harness scripts.
"""

from __future__ import annotations

import ast
import json
import os
import re
import string
import subprocess
import sys
from typing import Any

from mas_contribution_bench.data.external import extract_arc_expected_outputs, resolve_gpqa_gold
from mas_contribution_bench.data.schemas import FailureType


_IFBENCH_UNAVAILABLE_REASON: str | None = None


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
    candidates.extend(_balanced_json_candidates(value))
    for field in ("artifact", "answer", "prediction", "output", "outputs", "final_answer"):
        field_match = re.search(
            rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"',
            value,
            flags=re.S,
        )
        if field_match:
            try:
                candidates.append(json.loads(f'"{field_match.group(1)}"'))
            except Exception:
                candidates.append(field_match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            continue
    return None


def _balanced_json_candidates(text: str | None) -> list[str]:
    """Return balanced JSON/Python literal substrings from noisy model output."""
    if not text:
        return []
    value = str(text)
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    quote = ""
    escape = False
    pairs = {"{": "}", "[": "]"}
    closers = set(pairs.values())

    for idx, ch in enumerate(value):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote = ch
            continue

        if ch in pairs:
            if not stack:
                start = idx
            stack.append(pairs[ch])
            continue

        if ch in closers and stack:
            expected = stack.pop()
            if ch != expected:
                stack = []
                start = None
                continue
            if not stack and start is not None:
                candidates.append(value[start : idx + 1])
                start = None
    return candidates


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


def _coerce_arc_cell(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 9:
        return value
    if isinstance(value, float) and value.is_integer() and 0 <= int(value) <= 9:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]", value.strip()):
        return int(value.strip())
    return None


def _normalize_arc_grid(raw: Any) -> list[list[int]] | None:
    if not isinstance(raw, list) or not raw:
        return None
    normalized: list[list[int]] = []
    width: int | None = None
    for row in raw:
        if not isinstance(row, list) or not row:
            return None
        normalized_row: list[int] = []
        for cell in row:
            coerced = _coerce_arc_cell(cell)
            if coerced is None:
                return None
            normalized_row.append(coerced)
        if width is None:
            width = len(normalized_row)
        elif len(normalized_row) != width:
            return None
        normalized.append(normalized_row)
    height = len(normalized)
    if width is None or height > 30 or width > 30:
        return None
    return normalized


def _normalize_arc_outputs(raw: Any, expected_count: int | None = None) -> list[list[list[int]]]:
    if isinstance(raw, dict):
        for key in ("outputs", "output", "answer", "prediction", "artifact", "final_answer", "solution"):
            if key in raw:
                return _normalize_arc_outputs(raw[key], expected_count=expected_count)
        return []

    if isinstance(raw, str):
        nested = _load_jsonish(raw)
        if nested is None or nested == raw:
            return []
        return _normalize_arc_outputs(nested, expected_count=expected_count)

    grid = _normalize_arc_grid(raw)
    if grid is not None:
        return [grid]

    if isinstance(raw, list):
        outputs: list[list[list[int]]] = []
        for item in raw:
            normalized = _normalize_arc_grid(item)
            if normalized is None:
                return []
            outputs.append(normalized)
        if expected_count is not None and len(outputs) != expected_count:
            return outputs
        return outputs

    return []


def _extract_arc_prediction(prediction: str | None, expected_count: int | None = None) -> list[Any]:
    raw = _load_jsonish(prediction)
    outputs = _normalize_arc_outputs(raw, expected_count=expected_count)
    if outputs:
        return outputs

    # If the whole response is not parseable JSON but contains a complete
    # artifact/output array inside prose or a partly malformed JSON object, try
    # each balanced array/object candidate independently.
    for candidate in _balanced_json_candidates(unwrap_answer(prediction)):
        try:
            parsed = json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                continue
        outputs = _normalize_arc_outputs(parsed, expected_count=expected_count)
        if outputs:
            return outputs

    # Last resort: search specifically after ARC-ish field names so a complete
    # artifact array can still be recovered from a malformed surrounding object.
    value = unwrap_answer(prediction)
    for field in ("artifact", "outputs", "output", "answer", "prediction", "final_answer"):
        marker = re.search(rf'["\']?{field}["\']?\s*:', value)
        if not marker:
            continue
        tail = value[marker.end() :]
        outputs = _normalize_arc_outputs(_load_jsonish(tail), expected_count=expected_count)
        if outputs:
            return outputs

    return []


def extract_arc_prediction_json(prediction: str | None, expected_count: int | None = None) -> str | None:
    """Extract a strict ARC-AGI-2 JSON grid/list-of-grids from a model response."""
    outputs = _extract_arc_prediction(prediction, expected_count=expected_count)
    if not outputs:
        return None
    if expected_count == 1 or len(outputs) == 1:
        payload: Any = outputs[0]
    else:
        payload = outputs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def arc_agi_2_eval(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    expected = _extract_arc_expected(task)
    predicted = _extract_arc_prediction(prediction, expected_count=len(expected) if expected else None)
    if not expected:
        return 0.0, False, FailureType.UNKNOWN, {"evaluator": "arc_agi_2_exact_grid", "reason": "missing_expected_outputs"}
    if not predicted:
        return 0.0, False, FailureType.INVALID_FORMAT, {
            "evaluator": "arc_agi_2_exact_grid",
            "reason": "prediction_not_json_grid",
            "prediction_preview": str(prediction or "")[:500],
        }
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
    global _IFBENCH_UNAVAILABLE_REASON
    if _IFBENCH_UNAVAILABLE_REASON:
        raise RuntimeError(_IFBENCH_UNAVAILABLE_REASON)

    timeout = float(os.getenv("MAS_IFBENCH_IMPORT_TIMEOUT", "30"))
    probe = (
        "from ifbench import instructions_registry; "
        "print(len(getattr(instructions_registry, 'INSTRUCTION_DICT', {})))"
    )
    try:
        probe_result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        reason = (
            f"{IFBENCH_PACKAGE_ERROR} Import timed out after {timeout:.1f}s; "
            "pre-download NLTK resources or set MAS_IFBENCH_IMPORT_TIMEOUT higher."
        )
        raise RuntimeError(reason) from exc

    if probe_result.returncode != 0:
        reason = (
            f"{IFBENCH_PACKAGE_ERROR} Import probe failed: "
            f"{(probe_result.stderr or probe_result.stdout)[-1000:]}"
        )
        raise RuntimeError(reason)

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
