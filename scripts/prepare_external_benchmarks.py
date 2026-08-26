"""Download and normalize external benchmarks for MASContributionBench.

Reusable source descriptors and record normalization live in
``mas_contribution_bench.data.external``. This script is the download/argparse
CLI: HTTP, retries, resume, and fail-fast stay here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.data.external import (  # noqa: E402
    DEFAULT_EXTERNAL_DATASETS,
    SOURCES,
    DatasetSource,
    normalize_dataset,
)


MANIFEST_NAMES = frozenset({"download_manifest.json", "README.md"})


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


def has_nonempty_download(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.name not in MANIFEST_NAMES and path.stat().st_size > 0:
            return True
    return False


def has_nonempty_processed(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
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
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(SOURCES),
        help="Dataset key to prepare. Repeatable. Defaults to the seven phase-1 datasets.",
    )
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
    parser.add_argument("--force", action="store_true", help="Re-download and re-convert even if nonempty assets exist.")
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help="Continue after a dataset failure instead of stopping immediately.",
    )
    parser.set_defaults(fail_fast=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.dataset or list(DEFAULT_EXTERNAL_DATASETS)
    summary: dict[str, Any] = {}
    tasks_dir = args.processed_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0

    for key in selected:
        source = SOURCES[key]
        raw_dir = args.raw_root / source.dataset
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_file = tasks_dir / f"{source.dataset}_tasks.jsonl"
        raw_exists = has_nonempty_download(raw_dir)
        processed_exists = has_nonempty_processed(out_file)

        if args.skip_download:
            result: dict[str, Any] = {"status": "skipped_download", "reason": "--skip-download"}
        elif not args.force and raw_exists:
            result = {"status": "resumed_raw", "reason": "nonempty downloaded assets exist"}
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

        conversion_error = None
        if not args.force and processed_exists:
            records = count_jsonl(out_file)
            result = {**result, "conversion": "resumed_processed"}
        else:
            try:
                records = normalize_dataset(raw_dir, out_file, source, max_records=args.max_records)
            except Exception as exc:
                records = 0
                conversion_error = f"{type(exc).__name__}: {exc}"
                result = {**result, "conversion_error": conversion_error}

        if records == 0 and out_file.exists() and (args.force or not processed_exists):
            out_file.unlink()
        write_manifest(raw_dir, source, result, out_file if records else None, records)
        failed = records == 0 or result.get("status") == "failed" or conversion_error is not None
        if failed:
            exit_code = 1
        summary[key] = {
            "status": result.get("status"),
            "records": records,
            "raw_dir": str(raw_dir),
            "processed_file": str(out_file) if records else None,
            "result": result,
            "failed": failed,
        }
        print(json.dumps({key: summary[key]}, ensure_ascii=False, indent=2))
        if failed and args.fail_fast:
            break

    summary_path = args.processed_root / "external_benchmarks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
