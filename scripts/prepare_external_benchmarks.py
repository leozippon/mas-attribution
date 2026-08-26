"""Download and normalize external benchmarks for MASContributionBench.

This script complements the first-stage HumanEval/MBPP data with broader
generalization benchmarks. It is intentionally conservative: every dataset is
stored under data/raw/<dataset>/ with a manifest, and processed rows are written
to data/processed/tasks/<dataset>_tasks.jsonl only when raw files are available.

Network access to Hugging Face/GitHub can be unreliable on shared servers, so
the script records per-dataset download status and can be re-run safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.data.schemas import (  # noqa: E402
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
        notes="Broad multi-domain expert QA; often used for frontier-model evaluation.",
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
        hf_repos=[
            "walledai/IFBench",
            "Salesforce/IFBench",
            "allenai/IFBench",
            "google/IFEval",
        ],
        preferred_patterns=["*.jsonl", "*.parquet", "*.json", "*.csv"],
        notes="Instruction-following benchmark; repo naming is not fully standardized.",
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
        hf_repos=["arcprize/ARC-AGI-2"],
        github_archives=[
            "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip",
        ],
        preferred_patterns=["*.json", "*.jsonl"],
        notes="Grid-based abstract reasoning benchmark.",
    ),
}


def stable_id(*parts: str) -> str:
    text = "::".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "MASContributionBench/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_json(url: str, timeout: int, retries: int) -> Any:
    last_error: Exception | None = None
    headers = hf_headers()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on network
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to fetch JSON {url}: {last_error}")


def http_download(url: str, path: Path, timeout: int, retries: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    headers = hf_headers()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, path.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as exc:  # pragma: no cover - depends on network
            last_error = exc
            if path.exists():
                path.unlink()
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def pattern_score(path: str, patterns: list[str]) -> int:
    lowered = path.lower()
    score = 0
    for idx, pattern in enumerate(patterns):
        token = pattern.replace("*", "").lower()
        if token and token in lowered:
            score += 100 - idx
    if any(lowered.endswith(ext) for ext in (".parquet", ".jsonl", ".json", ".csv", ".yaml", ".yml")):
        score += 10
    if any(skip in lowered for skip in (".gitattributes", "readme", "license")):
        score -= 100
    return score


def hf_tree(repo: str, endpoint: str, timeout: int, retries: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(repo, safe="/")
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/api/datasets/{encoded}/tree/main?recursive=1"
    data = http_json(url, timeout=timeout, retries=retries)
    return [item for item in data if item.get("type") == "file"]


def download_hf_dataset(
    source: DatasetSource,
    raw_dir: Path,
    endpoint: str,
    timeout: int,
    retries: int,
    max_files: int,
) -> dict[str, Any]:
    errors: list[str] = []
    endpoint = endpoint.rstrip("/")
    for repo in source.hf_repos:
        try:
            files = hf_tree(repo, endpoint=endpoint, timeout=timeout, retries=retries)
        except Exception as exc:
            errors.append(f"{repo}: {exc}")
            continue

        candidates = []
        for item in files:
            path = item.get("path", "")
            if not any(path.lower().endswith(ext) for ext in (".parquet", ".jsonl", ".json", ".csv", ".yaml", ".yml")):
                continue
            score = pattern_score(path, source.preferred_patterns)
            if score > 0:
                candidates.append((score, path, item))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if not candidates:
            errors.append(f"{repo}: no usable data files found")
            continue

        downloaded = []
        for _, path, item in candidates[:max_files]:
            url = f"{endpoint}/datasets/{repo}/resolve/main/{urllib.parse.quote(path, safe='/')}"
            local = raw_dir / "hf" / repo.replace("/", "__") / path
            try:
                http_download(url, local, timeout=timeout, retries=retries)
            except Exception as exc:
                errors.append(f"{repo}/{path}: {exc}")
                continue
            downloaded.append(
                {
                    "repo": repo,
                    "path": path,
                    "local_path": str(local.relative_to(raw_dir)),
                    "size": local.stat().st_size,
                }
            )
        if downloaded:
            return {"status": "ok", "source_type": "huggingface", "repo": repo, "files": downloaded, "errors": errors}

    return {"status": "failed", "source_type": "huggingface", "errors": errors}


def download_github_archives(source: DatasetSource, raw_dir: Path, timeout: int, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for url in source.github_archives:
        archive_name = url.rstrip("/").split("/")[-1] or "archive.zip"
        archive_path = raw_dir / "github" / archive_name
        try:
            http_download(url, archive_path, timeout=timeout, retries=retries)
            extract_dir = raw_dir / "github" / archive_name.replace(".zip", "")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
            return {
                "status": "ok",
                "source_type": "github_archive",
                "url": url,
                "archive": str(archive_path.relative_to(raw_dir)),
                "extract_dir": str(extract_dir.relative_to(raw_dir)),
            }
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return {"status": "failed", "source_type": "github_archive", "errors": errors}


def iter_raw_records(raw_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield path, json.loads(line)
                        except json.JSONDecodeError:
                            continue
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
                # ARC-style dicts often map task_id -> task payload.
                yielded = False
                for key, value in data.items():
                    if isinstance(value, dict):
                        record = dict(value)
                        record.setdefault("raw_key", key)
                        yield path, record
                        yielded = True
                if not yielded:
                    yield path, data
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.DictReader(f):
                    yield path, dict(row)
        elif suffix == ".parquet":
            if pd is None:
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            for record in df.to_dict("records"):
                yield path, record


def first_value(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def sanitize_metadata_value(value: Any) -> Any:
    """Make raw metadata safe for Pydantic JSON serialization.

    Some benchmarks, especially HLE, include binary image bytes in parquet
    columns. Those bytes are useful provenance but cannot be serialized as
    UTF-8 JSON, so we keep only a compact placeholder.
    """
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
    if hasattr(value, "item"):
        try:
            return sanitize_metadata_value(value.item())
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return sanitize_metadata_value(value.tolist())
        except Exception:
            return str(value)
    return value


def safe_metadata(record: dict[str, Any]) -> dict[str, Any]:
    excluded = {"prompt", "question", "problem", "answer", "solution", "canonical_solution"}
    return {str(k): sanitize_metadata_value(v) for k, v in record.items() if k not in excluded}


def infer_prompt(record: dict[str, Any], source: DatasetSource) -> str:
    question = first_value(
        record,
        [
            "question",
            "prompt",
            "problem",
            "problem_statement",
            "instruction",
            "query",
            "task",
            "description",
            "instance",
        ],
    )
    if question is None and {"train", "test"} & set(record):
        question = {
            "train": record.get("train"),
            "test": record.get("test"),
        }
    prompt = as_text(question).strip()
    choices = first_value(record, ["choices", "options", "multiple_choice_targets"])
    if choices:
        prompt += "\n\nOptions:\n" + as_text(choices)
    if source.dataset == "livecodebench":
        starter = first_value(record, ["starter_code", "code", "function_signature"])
        if starter:
            prompt += "\n\nStarter code:\n" + as_text(starter)
        prompt += "\n\nReturn only executable Python code for the solution."
    return prompt or json.dumps(record, ensure_ascii=False, default=str)


def infer_answer(record: dict[str, Any]) -> str | None:
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


def infer_tests(record: dict[str, Any]) -> str | None:
    tests = first_value(record, ["tests", "test", "test_cases", "private_tests", "public_tests", "input_output"])
    return as_text(tests) if tests is not None else None


def infer_entry_point(record: dict[str, Any]) -> str | None:
    entry = first_value(record, ["entry_point", "function_name", "fn_name", "name"])
    if entry:
        return str(entry)
    code = as_text(first_value(record, ["starter_code", "prompt", "code"]))
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return match.group(1) if match else None


def normalize_dataset(raw_dir: Path, out_file: Path, source: DatasetSource, max_records: int | None = None) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with out_file.open("w", encoding="utf-8") as out:
        for raw_path, record in iter_raw_records(raw_dir):
            if "download_manifest" in raw_path.name:
                continue
            raw_id = as_text(first_value(record, ["task_id", "id", "qid", "question_id", "instance_id", "raw_key"]))
            if not raw_id:
                raw_id = stable_id(str(raw_path), json.dumps(record, sort_keys=True, ensure_ascii=False, default=str))
            task_id = f"{source.dataset}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', raw_id).strip('_')}"
            if task_id in seen:
                task_id = f"{task_id}_{count}"
            seen.add(task_id)

            prompt = infer_prompt(record, source)
            row = TaskRecord(
                task_id=task_id,
                dataset=source.dataset,
                split=str(first_value(record, ["split", "subset"]) or source.split),
                task_type=source.task_type,
                prompt=prompt,
                output_format=(
                    "python_code"
                    if source.task_type in {TaskType.CODE_GENERATION, str(TaskType.CODE_GENERATION)}
                    else "free_form_or_multiple_choice"
                ),
                entry_point=infer_entry_point(record),
                reference_solution=infer_answer(record),
                tests=infer_tests(record),
                evaluation=EvaluationConfig(
                    evaluator_type=source.evaluator_type,
                    metric=source.metric,
                    timeout_seconds=30,
                    sandbox="docker" if source.evaluator_type == EvaluatorType.UNIT_TEST else None,
                    extra={"official_evaluator_required": source.evaluator_type in {EvaluatorType.OFFICIAL_GRADER, EvaluatorType.ENVIRONMENT_REWARD}},
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
                    raw_file_path=str(raw_path.relative_to(raw_dir)),
                    conversion_version=CONVERSION_VERSION,
                    extra={"dataset_key": source.dataset, "source_notes": source.notes},
                ),
                metadata=safe_metadata(record),
            )
            out.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False, default=str) + "\n")
            count += 1
            if max_records and count >= max_records:
                break
    return count


def write_manifest(raw_dir: Path, source: DatasetSource, result: dict[str, Any], processed_file: Path | None, records: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": source.dataset,
        "display_name": source.display_name,
        "status": result.get("status"),
        "download_result": result,
        "processed_file": str(processed_file) if processed_file else None,
        "processed_records": records,
        "task_type": source.task_type,
        "metric": source.metric,
        "notes": source.notes,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (raw_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "README.md").write_text(
        f"# {source.display_name}\n\n"
        f"- dataset key: `{source.dataset}`\n"
        f"- status: `{result.get('status')}`\n"
        f"- processed records: `{records}`\n"
        f"- notes: {source.notes}\n\n"
        "This directory is managed by `scripts/prepare_external_benchmarks.py`.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(SOURCES), help="Dataset key to prepare. Repeatable.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke-test conversion.")
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint. Use https://hf-mirror.com on restricted networks.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Only convert files already present in raw dirs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.dataset or list(SOURCES)
    summary: dict[str, Any] = {}
    tasks_dir = args.processed_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for key in selected:
        source = SOURCES[key]
        raw_dir = args.raw_root / source.dataset
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_file = tasks_dir / f"{source.dataset}_tasks.jsonl"

        if args.skip_download:
            result = {"status": "skipped_download", "reason": "--skip-download"}
        else:
            result = download_hf_dataset(
                source,
                raw_dir,
                endpoint=args.hf_endpoint,
                timeout=args.timeout,
                retries=args.retries,
                max_files=args.max_files,
            )
            if result.get("status") != "ok" and source.github_archives:
                github_result = download_github_archives(source, raw_dir, timeout=args.timeout, retries=args.retries)
                result = {"huggingface": result, "github": github_result, "status": github_result.get("status")}

        try:
            records = normalize_dataset(raw_dir, out_file, source, max_records=args.max_records)
        except Exception as exc:
            records = 0
            result = {**result, "conversion_error": f"{type(exc).__name__}: {exc}"}

        if records == 0 and out_file.exists():
            out_file.unlink()
        write_manifest(raw_dir, source, result, out_file if records else None, records)
        summary[key] = {
            "status": result.get("status"),
            "records": records,
            "raw_dir": str(raw_dir),
            "processed_file": str(out_file) if records else None,
            "result": result,
        }
        print(json.dumps({key: summary[key]}, ensure_ascii=False, indent=2))

    summary_path = args.processed_root / "external_benchmarks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    failed = [key for key, item in summary.items() if item["records"] == 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
