"""Regression tests for exp08 input preflight."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mas_contribution_bench.runners.run_generalization import (
    _preflight_required_inputs,
    run_generalization,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _touch_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_exp08_config(
    path: Path,
    *,
    metadata: Path,
    score_files: list[Path],
    attribution_files: list[Path],
    coalition_files: list[Path],
    trace_files: list[Path],
    out_dir: Path,
) -> Path:
    def yaml_list(values: list[Path]) -> str:
        return "\n".join(f"    - {value}" for value in values)

    path.write_text(
        "\n".join(
            [
                "id: exp08_generalization",
                "datasets:",
                "  - name: humaneval",
                "    split: test",
                "inputs:",
                f"  task_metadata_file: {metadata}",
                "  score_files:",
                yaml_list(score_files),
                "  attribution_files:",
                yaml_list(attribution_files),
                "  coalition_files:",
                yaml_list(coalition_files),
                "  trace_files:",
                yaml_list(trace_files),
                "analysis:",
                f"  task_metadata_file: {metadata}",
                "outputs:",
                f"  summary_file: {out_dir / 'summary.jsonl'}",
                f"  task_slice_file: {out_dir / 'task_slices.jsonl'}",
                f"  communication_slice_file: {out_dir / 'communication_slices.jsonl'}",
                f"  ranking_file: {out_dir / 'rankings.jsonl'}",
                f"  correlation_file: {out_dir / 'correlations.jsonl'}",
                f"  feature_file: {out_dir / 'features.jsonl'}",
                f"  statistics_file: {out_dir / 'statistics.jsonl'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


class GeneralizationPreflightTests(unittest.TestCase):
    def test_preflight_accepts_nonempty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.jsonl"
            scores = root / "scores.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(metadata, [{"task_id": "humaneval/0", "dataset": "humaneval"}])
            _write_jsonl(scores, [{"task_id": "humaneval/0", "score": 1.0}])
            _write_jsonl(attribution, [{"task_id": "humaneval/0", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "humaneval/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "humaneval/0", "run_id": "r1"}])
            raw = {
                "analysis": {"task_metadata_file": str(metadata)},
                "inputs": {
                    "score_files": [str(scores)],
                    "attribution_files": [str(attribution)],
                    "coalition_files": [str(coalition)],
                    "trace_files": [str(traces)],
                },
            }
            _preflight_required_inputs(root, raw)

    def test_preflight_reports_all_missing_and_empty_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.jsonl"
            scores = root / "scores.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _touch_empty(metadata)
            _write_jsonl(attribution, [{"task_id": "humaneval/0"}])
            _touch_empty(coalition)
            raw = {
                "analysis": {"task_metadata_file": str(metadata)},
                "inputs": {
                    "score_files": [str(scores)],
                    "attribution_files": [str(attribution)],
                    "coalition_files": [str(coalition)],
                    "trace_files": [str(traces)],
                },
            }
            with self.assertRaises(FileNotFoundError) as ctx:
                _preflight_required_inputs(root, raw)
            message = str(ctx.exception)
            self.assertIn(str(metadata), message)
            self.assertIn(str(scores), message)
            self.assertIn(str(coalition), message)
            self.assertIn(str(traces), message)
            self.assertNotIn(str(attribution), message)
            self.assertIn("empty", message)
            self.assertIn("missing", message)

    def test_run_generalization_failure_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            config = _write_exp08_config(
                root / "exp08.yaml",
                metadata=root / "missing_metadata.jsonl",
                score_files=[root / "missing_scores.jsonl"],
                attribution_files=[root / "missing_attribution.jsonl"],
                coalition_files=[root / "missing_coalition.jsonl"],
                trace_files=[root / "missing_traces.jsonl"],
                out_dir=out_dir,
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                run_generalization(config)
            message = str(ctx.exception)
            self.assertIn("missing_metadata.jsonl", message)
            self.assertIn("missing_scores.jsonl", message)
            self.assertIn("missing_attribution.jsonl", message)
            self.assertIn("missing_coalition.jsonl", message)
            self.assertIn("missing_traces.jsonl", message)
            self.assertFalse(out_dir.exists())
            self.assertEqual(list(root.rglob("*.jsonl")), [])

    def test_run_generalization_accepts_nonempty_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            metadata = root / "metadata.jsonl"
            scores = root / "scores.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(
                metadata,
                [{"task_id": "humaneval/0", "dataset": "humaneval", "split": "test", "prompt_length": 80}],
            )
            _write_jsonl(
                scores,
                [
                    {
                        "task_id": "humaneval/0",
                        "run_id": "r1",
                        "dataset": "humaneval",
                        "architecture_id": "A1_pev",
                        "score": 1.0,
                        "passed": True,
                    }
                ],
            )
            _write_jsonl(
                attribution,
                [
                    {
                        "task_id": "humaneval/0",
                        "dataset": "humaneval",
                        "method": "loo",
                        "role": "coder",
                        "score": 0.25,
                    }
                ],
            )
            _write_jsonl(coalition, [{"task_id": "humaneval/0", "coalition_id": "c1", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "humaneval/0", "run_id": "r1", "input_tokens": 4, "output_tokens": 4}])
            config = _write_exp08_config(
                root / "exp08.yaml",
                metadata=metadata,
                score_files=[scores],
                attribution_files=[attribution],
                coalition_files=[coalition],
                trace_files=[traces],
                out_dir=out_dir,
            )
            summary = run_generalization(config)
            self.assertEqual(summary["selected_tasks"], 1)
            self.assertEqual(summary["attribution_rows"], 1)
            self.assertTrue((out_dir / "summary.jsonl").is_file())
            self.assertGreater((out_dir / "summary.jsonl").stat().st_size, 0)

    def test_zero_attribution_rows_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            metadata = root / "metadata.jsonl"
            scores = root / "scores.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(
                metadata,
                [{"task_id": "humaneval/0", "dataset": "humaneval", "split": "test"}],
            )
            _write_jsonl(scores, [{"task_id": "humaneval/0", "score": 1.0}])
            _write_jsonl(attribution, [{"task_id": "humaneval/other", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "humaneval/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "humaneval/0", "run_id": "r1"}])
            config = _write_exp08_config(
                root / "exp08.yaml",
                metadata=metadata,
                score_files=[scores],
                attribution_files=[attribution],
                coalition_files=[coalition],
                trace_files=[traces],
                out_dir=out_dir,
            )
            with self.assertRaises(ValueError) as ctx:
                run_generalization(config)
            self.assertIn("0 attribution rows", str(ctx.exception))
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
