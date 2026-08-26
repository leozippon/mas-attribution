"""Official-style evaluators for deterministic non-code benchmark tasks.

These functions cover benchmarks whose official score is exact answer,
exact grid match, or rule-based instruction following. Benchmarks that need a
workspace or an interactive environment are handled by external harness scripts.
"""

from __future__ import annotations

import json
import math
import re
import string
from typing import Any

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
    metadata = task.get("metadata") or {}
    reference = task.get("reference_solution")
    if dataset == "gpqa_diamond" and not reference:
        reference = metadata.get("Correct Answer") or metadata.get("Pre-Revision Correct Answer")
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


def _compare_count(count: int, relation: str, target: int) -> bool:
    rel = relation.lower().strip()
    if rel == "at least":
        return count >= target
    if rel in {"more than", "greater than"}:
        return count > target
    if rel == "at most":
        return count <= target
    if rel in {"less than", "fewer than"}:
        return count < target
    if rel in {"exactly", "equal to", "equals"}:
        return count == target
    return count >= target


def _sentences(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?。！？]+", answer) if s.strip()]


def _paragraphs(answer: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
    return parts or ([answer.strip()] if answer.strip() else [])


def _ifeval_one(instruction_id: str, kwargs: dict[str, Any], answer: str, prompt: str) -> tuple[bool | None, dict[str, Any]]:
    if instruction_id == "punctuation:no_comma":
        return ("," not in answer and "，" not in answer), {}
    if instruction_id == "length_constraints:number_words":
        words = re.findall(r"\b\w+\b", answer)
        target = int(kwargs.get("num_words", 0) or 0)
        return _compare_count(len(words), str(kwargs.get("relation", "at least")), target), {"word_count": len(words)}
    if instruction_id == "length_constraints:number_sentences":
        count = len(_sentences(answer))
        target = int(kwargs.get("num_sentences", 0) or 0)
        return _compare_count(count, str(kwargs.get("relation", "at least")), target), {"sentence_count": count}
    if instruction_id == "length_constraints:number_paragraphs":
        count = len(_paragraphs(answer))
        target = int(kwargs.get("num_paragraphs", 0) or 0)
        return count == target, {"paragraph_count": count}
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = _paragraphs(answer)
        nth = int(kwargs.get("nth_paragraph", 1) or 1) - 1
        expected = str(kwargs.get("first_word", "")).lower()
        actual = ""
        if 0 <= nth < len(paragraphs):
            match = re.search(r"\b\w+\b", paragraphs[nth])
            actual = match.group(0).lower() if match else ""
        return actual == expected, {"actual_first_word": actual, "expected_first_word": expected}
    if instruction_id == "keywords:forbidden_words":
        forbidden = [str(w).lower() for w in kwargs.get("forbidden_words", [])]
        lower = answer.lower()
        hits = [word for word in forbidden if re.search(rf"\b{re.escape(word)}\b", lower)]
        return not hits, {"forbidden_hits": hits}
    if instruction_id == "keywords:existence":
        keywords = [str(w).lower() for w in kwargs.get("keywords", [])]
        lower = answer.lower()
        missing = [word for word in keywords if word not in lower]
        return not missing, {"missing_keywords": missing}
    if instruction_id == "keywords:frequency":
        keyword = str(kwargs.get("keyword", "")).lower()
        target = int(kwargs.get("frequency", 0) or 0)
        count = len(re.findall(rf"\b{re.escape(keyword)}\b", answer.lower())) if keyword else 0
        return _compare_count(count, str(kwargs.get("relation", "at least")), target), {"keyword_count": count}
    if instruction_id == "keywords:letter_frequency":
        letter = str(kwargs.get("letter", ""))
        target = int(kwargs.get("let_frequency", 0) or 0)
        count = answer.count(letter) if letter else 0
        return _compare_count(count, str(kwargs.get("let_relation", "at least")), target), {"letter_count": count}
    if instruction_id == "detectable_format:number_highlighted_sections":
        target = int(kwargs.get("num_highlights", 0) or 0)
        count = len(re.findall(r"(?<!\*)\*[^*\n][^*]*\*(?!\*)", answer))
        return count >= target, {"highlight_count": count}
    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>\n]+>>", answer) or re.search(r"^\s*#\s+.+", answer, flags=re.M)), {}
    if instruction_id == "detectable_format:number_bullet_lists":
        target = int(kwargs.get("num_bullets", 0) or 0)
        count = len(re.findall(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", answer))
        return count >= target, {"bullet_count": count}
    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(unwrap_answer(answer))
            return True, {}
        except Exception as exc:
            return False, {"json_error": str(exc)}
    if instruction_id == "detectable_format:multiple_sections":
        target = int(kwargs.get("num_sections", 0) or 0)
        splitter = str(kwargs.get("section_spliter", "PARAGRAPH"))
        count = len(_paragraphs(answer)) if splitter.upper() == "PARAGRAPH" else len(re.split(re.escape(splitter), answer))
        return count >= target, {"section_count": count}
    if instruction_id == "detectable_format:constrained_response":
        normalized = normalize_text(answer)
        return normalized in {"yes", "no", "maybe", "true", "false"}, {"normalized": normalized}
    if instruction_id == "detectable_content:number_placeholders":
        target = int(kwargs.get("num_placeholders", 0) or 0)
        count = len(re.findall(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}", answer))
        return count >= target, {"placeholder_count": count}
    if instruction_id == "detectable_content:postscript":
        marker = str(kwargs.get("postscript_marker", "P.S."))
        return marker.lower() in answer.lower(), {"marker": marker}
    if instruction_id == "startend:end_checker":
        phrase = str(kwargs.get("end_phrase", "")).strip()
        return answer.rstrip().endswith(phrase), {"end_phrase": phrase}
    if instruction_id == "startend:quotation":
        stripped = answer.strip()
        return len(stripped) >= 2 and stripped[0] in {'"', "'", "“"} and stripped[-1] in {'"', "'", "”"}, {}
    if instruction_id == "change_case:english_lowercase":
        letters = re.findall(r"[A-Za-z]", answer)
        return bool(letters) and all(ch.islower() for ch in letters), {}
    if instruction_id == "change_case:english_capital":
        letters = re.findall(r"[A-Za-z]", answer)
        return bool(letters) and all(not ch.islower() for ch in letters), {}
    if instruction_id == "change_case:capital_word_frequency":
        words = re.findall(r"\b[A-Z]{2,}\b", answer)
        target = int(kwargs.get("capital_frequency", 0) or 0)
        return _compare_count(len(words), str(kwargs.get("capital_relation", "at most")), target), {"capital_word_count": len(words)}
    if instruction_id == "combination:repeat_prompt":
        required = str(kwargs.get("prompt_to_repeat", prompt)).strip()
        return normalize_text(answer) == normalize_text(required), {}
    if instruction_id == "combination:two_responses":
        markers = len(re.findall(r"(?im)response\s*[12]|answer\s*[12]|^\s*[12][.)]", answer))
        return markers >= 2, {"response_markers": markers}
    if instruction_id == "language:response_language":
        return None, {"reason": "language_detection_requires_official_checker", "language": kwargs.get("language")}
    return None, {"reason": "unsupported_instruction"}


def ifbench_eval(task: dict[str, Any], prediction: str | None) -> tuple[float, bool, FailureType, dict[str, Any]]:
    answer = unwrap_answer(prediction)
    if not answer.strip():
        return 0.0, False, FailureType.EMPTY_OUTPUT, {"evaluator": "ifeval_official_compatible_rule_engine"}
    metadata = task.get("metadata") or {}
    ids = list(metadata.get("instruction_id_list") or [])
    kwargs_list = list(metadata.get("kwargs") or [])
    prompt = str(task.get("prompt") or "")
    details: list[dict[str, Any]] = []
    supported = 0
    passed_count = 0
    for idx, instruction_id in enumerate(ids):
        kwargs = kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}
        passed, extra = _ifeval_one(str(instruction_id), kwargs, answer, prompt)
        details.append({"instruction_id": instruction_id, "passed": passed, **extra})
        if passed is not None:
            supported += 1
            passed_count += int(bool(passed))
    if supported == 0:
        return 0.0, False, FailureType.UNKNOWN, {
            "evaluator": "ifeval_official_compatible_rule_engine",
            "supported_rules": 0,
            "details": details,
        }
    score = passed_count / supported
    passed = math.isclose(score, 1.0)
    return score, passed, (FailureType.NONE if passed else FailureType.TEST_FAILURE), {
        "evaluator": "ifeval_official_compatible_rule_engine",
        "supported_rules": supported,
        "total_rules": len(ids),
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
