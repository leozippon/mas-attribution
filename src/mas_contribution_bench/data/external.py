"""Reusable descriptors and normalization for external benchmarks.

Download/retry stays in ``scripts/prepare_external_benchmarks.py``. This module
owns source descriptors, record inference, ARC public-view sanitization, and
TaskRecord conversion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    pd = None

from .schemas import (
    ContributionMetadata,
    EvaluationConfig,
    EvaluatorType,
    MASMetadata,
    SourceInfo,
    TaskDifficulty,
    TaskRecord,
    TaskType,
)


CONVERSION_VERSION = "external_v1"

DEFAULT_EXTERNAL_DATASETS: tuple[str, ...] = (
    "aime_2026",
    "gpqa_diamond",
    "hle",
    "arc_agi_2",
    "ifbench",
    "livecodebench",
    "swebench_verified",
)

GPQA_GOLD_KEYS: tuple[str, ...] = (
    "Correct Answer",
    "correct_answer",
    "CorrectAnswer",
    "Pre-Revision Correct Answer",
    "pre_revision_correct_answer",
    "gold_answer",
    "gold",
    "answer",
)

ARC_EXPECTED_METADATA_KEYS: tuple[str, ...] = (
    "expected_test_outputs",
    "expected_outputs",
    "test_outputs",
)

LOCAL_ANSWER_DATASETS: frozenset[str] = frozenset(
    {"aime_2026", "gpqa_diamond", "hle", "arc_agi_2"}
)

SIDECAR_FILENAMES: frozenset[str] = frozenset(
    {
        "dataset_info.json",
        "dataset_infos.json",
        "config.json",
        "configs.json",
        "download_manifest.json",
    }
)

SIDECAR_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "builder_name",
        "config_name",
        "dataset_size",
        "download_size",
        "citation",
        "homepage",
        "features",
        "splits",
        "dataset_info",
    }
)

TASK_RECORD_HINT_KEYS: frozenset[str] = frozenset(
    {
        "question",
        "Question",
        "prompt",
        "problem",
        "problem_statement",
        "question_content",
        "instruction",
        "train",
        "test",
        "instruction_id_list",
    }
)

DIRECT_PROMPT_KEYS: dict[str, tuple[str, ...]] = {
    "aime_2026": ("problem", "question", "prompt", "problem_statement", "Question"),
    "gpqa_diamond": ("Question", "question", "prompt", "problem", "Pre-Revision Question"),
    "hle": ("question", "prompt", "problem", "Question"),
    "ifbench": ("prompt", "instruction"),
    "livecodebench": ("question_content", "question", "prompt", "problem", "problem_statement"),
    "swebench_verified": ("problem_statement", "prompt"),
}

DEFAULT_DIRECT_PROMPT_KEYS: tuple[str, ...] = (
    "question",
    "question_content",
    "prompt",
    "problem",
    "problem_statement",
    "instruction",
    "Question",
)

PRIVATE_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "answer",
        "gold",
        "gold_answer",
        "correct_answer",
        "Correct Answer",
        "CorrectAnswer",
        "Pre-Revision Correct Answer",
        "pre_revision_correct_answer",
        "canonical_solution",
        "reference_solution",
        "solution",
        "expected_output",
        "expected_outputs",
        "expected_test_outputs",
        "test_outputs",
        "private_tests",
        "hidden_tests",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "label",
        "target",
        "tests",
        "test_cases",
        "input_output",
    }
)


@dataclass
class DatasetSource:
    dataset: str
    display_name: str
    task_type: str
    evaluator_type: str
    metric: str
    hf_repos: list[str] = field(default_factory=list)
    github_archives: list[str] = field(default_factory=list)
    preferred_patterns: list[str] = field(default_factory=list)
    split: str = "test"
    notes: str = ""


SOURCES: dict[str, DatasetSource] = {
    "hle": DatasetSource(
        dataset="hle",
        display_name="Humanity's Last Exam",
        task_type=TaskType.GENERAL_MAS,
        evaluator_type=EvaluatorType.EXACT_MATCH,
        metric="answer_accuracy",
        hf_repos=["cais/hle"],
        preferred_patterns=["*.parquet", "*.jsonl", "*.json", "*.csv"],
        notes="Humanity's Last Exam; media-bearing rows are excluded for the text-only endpoint.",
    ),
    "gpqa_diamond": DatasetSource(
        dataset="gpqa_diamond",
        display_name="GPQA Diamond",
        task_type=TaskType.RESEARCH,
        evaluator_type=EvaluatorType.EXACT_MATCH,
        metric="multiple_choice_accuracy",
        hf_repos=["Idavidrein/gpqa", "m-a-p/GPQA-Diamond"],
        preferred_patterns=["*diamond*.csv", "*diamond*.jsonl", "*.parquet", "*.csv"],
        notes="Graduate-level science QA; use the diamond subset when available.",
    ),
    "aime_2026": DatasetSource(
        dataset="aime_2026",
        display_name="AIME 2026",
        task_type=TaskType.GENERAL_MAS,
        evaluator_type=EvaluatorType.EXACT_MATCH,
        metric="exact_match",
        hf_repos=[
            "math-ai/aime26",
            "AI-MO/AIME_2026",
            "HuggingFaceH4/aime_2026",
            "Maxwell-Jia/AIME_2026",
        ],
        preferred_patterns=["*.jsonl", "*.parquet", "*.json", "*.csv"],
        notes="Source availability changes quickly; verify license/provenance before paper use.",
    ),
    "livecodebench": DatasetSource(
        dataset="livecodebench",
        display_name="LiveCodeBench",
        task_type=TaskType.CODE_GENERATION,
        evaluator_type=EvaluatorType.UNIT_TEST,
        metric="pass_at_1",
        hf_repos=["livecodebench/code_generation_lite", "livecodebench/code_generation"],
        preferred_patterns=["*.parquet", "*.jsonl", "*.json"],
        notes="Live programming benchmark; lite subset is preferred for smoke tests.",
    ),
    "swebench_verified": DatasetSource(
        dataset="swebench_verified",
        display_name="SWE-bench Verified",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        evaluator_type=EvaluatorType.OFFICIAL_GRADER,
        metric="resolved",
        hf_repos=["princeton-nlp/SWE-bench_Verified"],
        preferred_patterns=["*.parquet", "*.jsonl", "*.json"],
        notes="Human-validated SWE-bench subset; official harness required for final scoring.",
    ),
    "ifbench": DatasetSource(
        dataset="ifbench",
        display_name="IFBench",
        task_type=TaskType.GENERAL_MAS,
        evaluator_type=EvaluatorType.CUSTOM,
        metric="instruction_following_score",
        hf_repos=["allenai/IFBench_test"],
        preferred_patterns=["*.jsonl", "*.parquet", "*.json", "*.csv"],
        notes="AllenAI IFBench official test set (allenai/IFBench_test).",
    ),
    "tau2_bench": DatasetSource(
        dataset="tau2_bench",
        display_name="tau2-bench",
        task_type=TaskType.GENERAL_MAS,
        evaluator_type=EvaluatorType.ENVIRONMENT_REWARD,
        metric="task_success",
        hf_repos=[
            "Sierra/tau2-bench",
            "Salesforce/tau2-bench",
            "tau2-bench/tau2-bench",
        ],
        github_archives=[
            "https://github.com/sierra-research/tau2-bench/archive/refs/heads/main.zip",
        ],
        preferred_patterns=["*.jsonl", "*.parquet", "*.json", "*.yaml", "*.yml"],
        notes="Agentic tool-use/environment benchmark; exact official evaluator should be added later.",
    ),
    "arc_agi_2": DatasetSource(
        dataset="arc_agi_2",
        display_name="ARC-AGI-2",
        task_type=TaskType.PLANNING,
        evaluator_type=EvaluatorType.EXACT_MATCH,
        metric="grid_exact_match",
        hf_repos=[],
        github_archives=[
            "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip",
        ],
        preferred_patterns=["*.json", "*.jsonl"],
        notes="Official ARC Prize repository; normalization excludes training tasks.",
    ),
}


def stable_id(*parts: str) -> str:
    text = "::".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def first_value(record: dict[str, Any], keys: list[str] | tuple[str, ...]) -> Any:
    for key in keys:
        if key not in record:
            continue
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def sanitize_metadata_value(value: Any) -> Any:
    """Make raw metadata safe for Pydantic JSON serialization."""
    if isinstance(value, bytes):
        return {"type": "bytes", "num_bytes": len(value)}
    if isinstance(value, bytearray):
        return {"type": "bytearray", "num_bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): sanitize_metadata_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_metadata_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_metadata_value(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            return sanitize_metadata_value(value.item())
        except Exception:
            return str(value)
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return sanitize_metadata_value(to_list())
        except Exception:
            return str(value)
    return value


def coerce_jsonable(value: Any) -> Any:
    value = sanitize_metadata_value(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def safe_metadata(record: dict[str, Any]) -> dict[str, Any]:
    excluded = {"prompt", "question", "problem", "answer", "solution", "canonical_solution"}
    return {str(k): sanitize_metadata_value(v) for k, v in record.items() if k not in excluded}


def _is_sidecar_path(path: Path) -> bool:
    name = path.name.lower()
    if name in SIDECAR_FILENAMES:
        return True
    if "download_manifest" in name:
        return True
    return name.startswith("dataset_info") and path.suffix.lower() == ".json"


def _is_config_sidecar_record(record: dict[str, Any]) -> bool:
    keys = set(record)
    if not (keys & SIDECAR_RECORD_KEYS):
        return False
    return not bool(keys & TASK_RECORD_HINT_KEYS)


def _ifbench_instruction_ids(record: dict[str, Any]) -> list[str]:
    raw_ids = first_value(record, ["instruction_id_list", "instruction_ids"])
    if raw_ids is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            raw_ids = first_value(metadata, ["instruction_id_list", "instruction_ids"])
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        return [raw_ids] if raw_ids.strip() else []
    if isinstance(raw_ids, (list, tuple)):
        return [str(item) for item in raw_ids if item not in (None, "")]
    return []


def _has_source_appropriate_prompt(record: dict[str, Any], source: DatasetSource, prompt: str) -> bool:
    if not str(prompt or "").strip():
        return False
    if source.dataset == "arc_agi_2" or _is_arc_payload(record):
        return _is_arc_payload(record)
    keys = DIRECT_PROMPT_KEYS.get(source.dataset, DEFAULT_DIRECT_PROMPT_KEYS)
    return first_value(record, keys) is not None


def _has_nonempty_media(record: dict[str, Any]) -> bool:
    for key in ("image", "images", "image_url", "image_path"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _source_path_is_eligible(source: DatasetSource, raw_path: Path) -> bool:
    path_text = str(raw_path).replace("\\", "/").lower()
    if source.dataset == "arc_agi_2" and "/data/training/" in path_text:
        return False
    return True


def iter_raw_records(raw_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or _is_sidecar_path(path):
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict):
                            yield path, payload
        elif suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield path, item
            elif isinstance(data, dict):
                yielded = False
                if _is_arc_payload(data):
                    yield path, data
                    continue
                for key, value in data.items():
                    if isinstance(value, dict):
                        record = dict(value)
                        record.setdefault("raw_key", key)
                        yield path, record
                        yielded = True
                if not yielded:
                    yield path, data
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    yield path, dict(row)
        elif suffix == ".parquet":
            if pd is None:
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            for record in frame.to_dict("records"):
                if isinstance(record, dict):
                    yield path, record


def _is_arc_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    train = value.get("train")
    test = value.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        return False
    cells = [item for item in [*train, *test] if isinstance(item, dict)]
    return bool(cells) and all("input" in item for item in cells)


def _looks_like_grid(value: Any) -> bool:
    return isinstance(value, list)


def public_arc_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Train examples plus test inputs only. Test outputs are omitted."""
    train_raw = coerce_jsonable(record.get("train"))
    test_raw = coerce_jsonable(record.get("test"))
    train: list[dict[str, Any]] = []
    if isinstance(train_raw, list):
        for item in train_raw:
            if not isinstance(item, dict) or "input" not in item:
                continue
            example = {"input": item.get("input")}
            if "output" in item:
                example["output"] = item.get("output")
            train.append(example)
    test: list[dict[str, Any]] = []
    if isinstance(test_raw, list):
        for item in test_raw:
            if isinstance(item, dict) and "input" in item:
                test.append({"input": item.get("input")})
            elif _looks_like_grid(item):
                test.append({"input": item})
    return {"train": train, "test": test}


def extract_arc_expected_outputs(record: dict[str, Any]) -> list[Any]:
    raw_metadata = record.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    for key in ARC_EXPECTED_METADATA_KEYS:
        value = coerce_jsonable(metadata.get(key))
        if isinstance(value, list) and value:
            return value
    extra = {}
    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        extra = evaluation.get("extra") or {}
    elif evaluation is not None:
        extra = getattr(evaluation, "extra", None) or {}
    if isinstance(extra, dict):
        for key in ARC_EXPECTED_METADATA_KEYS:
            value = coerce_jsonable(extra.get(key))
            if isinstance(value, list) and value:
                return value
    reference = coerce_jsonable(record.get("reference_solution"))
    if isinstance(reference, list) and reference:
        return reference
    test_raw = coerce_jsonable(record.get("test"))
    expected: list[Any] = []
    if isinstance(test_raw, list):
        for item in test_raw:
            if isinstance(item, dict) and "output" in item:
                expected.append(item["output"])
    return expected


def resolve_gpqa_gold(record: dict[str, Any]) -> str | None:
    direct = first_value(record, GPQA_GOLD_KEYS)
    if direct is not None:
        return as_text(direct)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        nested = first_value(metadata, GPQA_GOLD_KEYS)
        if nested is not None:
            return as_text(nested)
        raw = metadata.get("raw")
        if isinstance(raw, dict):
            nested = first_value(raw, GPQA_GOLD_KEYS)
            if nested is not None:
                return as_text(nested)
    return None


def _parse_json_prefix(text: str) -> tuple[Any | None, str]:
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "{[":
        return None, stripped
    decoder = json.JSONDecoder()
    try:
        payload, index = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None, stripped
    return payload, stripped[index:]


def sanitize_public_prompt(prompt: str | None, *, dataset: str = "") -> str:
    """Drop ARC test outputs and JSON gold/private fields from a prompt string."""
    if not prompt:
        return ""
    payload, rest = _parse_json_prefix(prompt)
    dataset_name = str(dataset or "").lower()
    if isinstance(payload, dict) and (dataset_name == "arc_agi_2" or _is_arc_payload(payload)):
        public = public_arc_payload(payload)
        suffix = rest.rstrip()
        body = json.dumps(public, ensure_ascii=False)
        return f"{body}\n{suffix}".strip() if suffix else body
    if isinstance(payload, dict):
        cleaned = {key: value for key, value in payload.items() if key not in PRIVATE_RECORD_KEYS}
        if dataset_name == "arc_agi_2" or _is_arc_payload(cleaned):
            cleaned = public_arc_payload(cleaned)
        suffix = rest.rstrip()
        body = json.dumps(cleaned, ensure_ascii=False)
        return f"{body}\n{suffix}".strip() if suffix else body
    return str(prompt)


def public_task_view(task: Any) -> dict[str, Any]:
    """Return the task fields that may be shown to a model.

    Tests, gold/reference answers, ARC test outputs, and raw metadata stay off
    the public view even when a legacy JSONL still stores them on the record.
    """
    if hasattr(task, "model_dump"):
        record = task.model_dump(mode="json")
    else:
        record = dict(task or {})
    dataset = str(record.get("dataset") or "").lower()
    view: dict[str, Any] = {
        "task_id": record.get("task_id"),
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "task_type": record.get("task_type"),
        "prompt": sanitize_public_prompt(record.get("prompt"), dataset=dataset),
        "entry_point": record.get("entry_point"),
        "input_format": record.get("input_format"),
        "output_format": record.get("output_format"),
    }
    context = record.get("context")
    if context:
        view["context"] = sanitize_public_prompt(str(context), dataset=dataset)
    return view


def _gpqa_is_diamond(record: dict[str, Any], raw_path: Path) -> bool:
    path_markers = [raw_path.name.lower()]
    path_markers.extend(part.lower() for part in raw_path.parts if "__" in part)
    if any("diamond" in marker for marker in path_markers):
        return True
    subset = first_value(record, ["Set", "set", "subset", "subset_name", "split"])
    if subset is None:
        return False
    return "diamond" in str(subset).lower()


def _gpqa_prompt(record: dict[str, Any]) -> str:
    question = first_value(
        record,
        ["Question", "question", "prompt", "problem", "Pre-Revision Question"],
    )
    if question is None:
        return ""
    prompt = as_text(question).strip()
    choices = first_value(record, ["choices", "options", "multiple_choice_targets"])
    if choices is None:
        collected = []
        gold = resolve_gpqa_gold(record)
        for key in (
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ):
            value = record.get(key)
            if value is not None and str(value).strip():
                collected.append(as_text(value))
        if gold:
            collected.append(as_text(gold))
        if collected:
            choices = sorted(collected, key=lambda item: item.lower())
    if choices:
        prompt += "\n\nOptions:\n" + as_text(choices)
    return prompt


def infer_prompt(record: dict[str, Any], source: DatasetSource) -> str:
    if source.dataset == "gpqa_diamond":
        prompt = _gpqa_prompt(record)
        return prompt
    if source.dataset == "arc_agi_2" or _is_arc_payload(record):
        public = public_arc_payload(record)
        return (
            json.dumps(public, ensure_ascii=False)
            + "\n\nPredict the output grid(s) for the test input(s) as JSON."
        )
    keys = DIRECT_PROMPT_KEYS.get(source.dataset, DEFAULT_DIRECT_PROMPT_KEYS)
    question = first_value(record, keys)
    prompt = as_text(question).strip() if question is not None else ""
    choices = first_value(record, ["choices", "options", "multiple_choice_targets"])
    if choices:
        prompt += "\n\nOptions:\n" + as_text(choices)
    if source.dataset == "livecodebench":
        starter = first_value(record, ["starter_code", "code", "function_signature"])
        if starter:
            prompt += "\n\nStarter code:\n" + as_text(starter)
        prompt += "\n\nReturn only executable Python code for the solution."
    return prompt


def infer_answer(record: dict[str, Any], source: DatasetSource | None = None) -> str | None:
    dataset = source.dataset if source is not None else str(record.get("dataset") or "")
    if dataset == "gpqa_diamond":
        return resolve_gpqa_gold(record)
    if dataset == "arc_agi_2" or _is_arc_payload(record):
        expected = extract_arc_expected_outputs(record)
        return json.dumps(expected, ensure_ascii=False) if expected else None
    answer = first_value(
        record,
        [
            "answer",
            "gold",
            "gold_answer",
            "correct_answer",
            "target",
            "label",
            "solution",
            "canonical_solution",
            "expected_output",
        ],
    )
    return as_text(answer) if answer is not None else None


def infer_tests(record: dict[str, Any], source: DatasetSource | None = None) -> str | None:
    dataset = source.dataset if source is not None else str(record.get("dataset") or "")
    if dataset == "arc_agi_2" or _is_arc_payload(record):
        return None
    if dataset == "livecodebench":
        tests = first_value(record, ["private_tests", "hidden_tests", "tests", "test_cases", "input_output", "public_tests"])
        return as_text(tests) if tests is not None else None
    if dataset == "swebench_verified":
        tests = first_value(record, ["test_patch", "FAIL_TO_PASS", "tests", "private_tests"])
        return as_text(tests) if tests is not None else None
    tests = first_value(record, ["tests", "test_cases", "private_tests", "public_tests", "input_output"])
    return as_text(tests) if tests is not None else None


def infer_entry_point(record: dict[str, Any]) -> str | None:
    entry = first_value(record, ["entry_point", "function_name", "fn_name", "name"])
    if entry:
        return str(entry)
    code = as_text(first_value(record, ["starter_code", "prompt", "code"]))
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return match.group(1) if match else None


def _task_output_format(source: DatasetSource) -> str:
    if source.task_type in {TaskType.CODE_GENERATION, str(TaskType.CODE_GENERATION)}:
        return "python_code"
    if source.dataset == "arc_agi_2":
        return "json_grid"
    if source.dataset == "swebench_verified":
        return "patch"
    return "free_form_or_multiple_choice"


def normalize_record(
    record: dict[str, Any],
    source: DatasetSource,
    *,
    raw_path: Path,
    raw_dir: Path,
    count: int,
    seen: set[str],
) -> TaskRecord | None:
    if _is_sidecar_path(raw_path) or _is_config_sidecar_record(record):
        return None
    if not _source_path_is_eligible(source, raw_path):
        return None
    if source.dataset == "gpqa_diamond" and not _gpqa_is_diamond(record, raw_path):
        return None
    if source.dataset == "hle" and _has_nonempty_media(record):
        return None
    raw_id = as_text(first_value(record, ["task_id", "id", "qid", "question_id", "instance_id", "raw_key"]))
    if not raw_id:
        raw_id = stable_id(str(raw_path), json.dumps(record, sort_keys=True, ensure_ascii=False, default=str))
    task_id = f"{source.dataset}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', raw_id).strip('_')}"
    if task_id in seen:
        task_id = f"{task_id}_{count}"

    prompt = infer_prompt(record, source)
    if not _has_source_appropriate_prompt(record, source, prompt):
        return None
    reference = infer_answer(record, source)
    metadata = safe_metadata(record)
    if source.dataset == "arc_agi_2" or _is_arc_payload(record):
        expected = extract_arc_expected_outputs(record)
        if expected:
            metadata["expected_test_outputs"] = expected
            if reference is None:
                reference = json.dumps(expected, ensure_ascii=False)
    if source.dataset == "gpqa_diamond" and reference:
        metadata.setdefault("Correct Answer", reference)
    if source.dataset in LOCAL_ANSWER_DATASETS and (reference is None or not str(reference).strip()):
        return None
    if source.dataset == "ifbench" and not _ifbench_instruction_ids(record):
        return None

    try:
        relative_path = str(raw_path.relative_to(raw_dir))
    except ValueError:
        relative_path = str(raw_path)

    return TaskRecord(
        task_id=task_id,
        dataset=source.dataset,
        split=str(first_value(record, ["split", "subset"]) or source.split),
        task_type=source.task_type,
        prompt=prompt,
        output_format=_task_output_format(source),
        entry_point=infer_entry_point(record),
        reference_solution=reference,
        tests=infer_tests(record, source),
        evaluation=EvaluationConfig(
            evaluator_type=(
                source.evaluator_type
                if isinstance(source.evaluator_type, EvaluatorType)
                else EvaluatorType(source.evaluator_type)
            ),
            metric=source.metric,
            timeout_seconds=30,
            sandbox="docker" if str(source.evaluator_type) == EvaluatorType.UNIT_TEST.value else None,
            extra={
                "official_evaluator_required": source.evaluator_type
                in {EvaluatorType.OFFICIAL_GRADER, EvaluatorType.ENVIRONMENT_REWARD},
            },
        ),
        mas_metadata=MASMetadata(
            requires_planning=True,
            requires_coding=source.task_type in {TaskType.CODE_GENERATION, TaskType.SOFTWARE_ENGINEERING},
            requires_verification=True,
            requires_research=source.task_type in {TaskType.RESEARCH, TaskType.GENERAL_MAS},
            requires_tool_use=source.task_type
            in {TaskType.CODE_GENERATION, TaskType.SOFTWARE_ENGINEERING, TaskType.GENERAL_MAS, TaskType.PLANNING},
            estimated_agents=["planner", "researcher", "coder", "verifier", "finalizer"],
            extra={"benchmark_display_name": source.display_name},
        ),
        difficulty=TaskDifficulty(
            source="external_benchmark_default",
            level="unknown",
            input_length=len(prompt),
            extra={"benchmark": source.display_name},
        ),
        contribution_metadata=ContributionMetadata(
            eligible_roles=["planner", "researcher", "coder", "verifier", "critic", "finalizer"],
            default_architectures=["A2_chain", "A3_dag", "A5_star", "A7_graph"],
            permission_requirements=["read_task", "send_message", "receive_message"],
            intervention_tags=["generalization", "task_dependency", "communication"],
        ),
        source=SourceInfo(
            raw_dataset=source.display_name,
            raw_task_id=raw_id,
            raw_file_path=relative_path,
            conversion_version=CONVERSION_VERSION,
            extra={"dataset_key": source.dataset, "source_notes": source.notes},
        ),
        metadata=metadata,
    )


def normalize_dataset(
    raw_dir: Path,
    out_file: Path,
    source: DatasetSource,
    max_records: int | None = None,
) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with out_file.open("w", encoding="utf-8") as out:
        for raw_path, record in iter_raw_records(raw_dir):
            if _is_sidecar_path(raw_path):
                continue
            row = normalize_record(
                record,
                source,
                raw_path=raw_path,
                raw_dir=raw_dir,
                count=count,
                seen=seen,
            )
            if row is None:
                continue
            seen.add(row.task_id)
            out.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False, default=str) + "\n")
            count += 1
            if max_records and count >= max_records:
                break
    return count
