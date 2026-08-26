"""Regression tests for exp08 input preflight and task-file metadata."""

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


def _task_record(task_id: str = "hle/0", *, prompt: str = "question about hle") -> dict:
    return {
        "task_id": task_id,
        "dataset": "hle",
        "split": "test",
        "task_type": "general_mas",
        "prompt": prompt,
        "evaluation": {"evaluator_type": "exact_match", "metric": "answer_accuracy"},
        "source": {"raw_dataset": "hle", "raw_task_id": task_id},
        "difficulty": {"level": "unknown", "source": "test"},
    }


def _write_exp08_config(
    path: Path,
    *,
    task_file: Path,
    attribution_files: list[Path],
    coalition_files: list[Path],
    trace_files: list[Path],
    out_dir: Path,
    extra_analysis: str = "",
) -> Path:
    def yaml_list(values: list[Path]) -> str:
        return "\n".join(f"    - {value}" for value in values)

    lines = [
        "id: exp08_generalization",
        "datasets:",
        "  - name: hle",
        f"    task_file: {task_file}",
        "    split: test",
        "    max_tasks: 30",
        "    task_selection: seeded_random",
        "sampling:",
        "  selection_seed: 0",
        "  task_selection: seeded_random",
        "inputs:",
        "  attribution_files:",
        yaml_list(attribution_files),
        "  coalition_files:",
        yaml_list(coalition_files),
        "  trace_files:",
        yaml_list(trace_files),
        "analysis:",
        "  analysis_only: true",
        "  descriptive_only: true",
        extra_analysis,
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
    path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    return path


class GeneralizationPreflightTests(unittest.TestCase):
    def test_preflight_accepts_nonempty_task_and_upstream_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(tasks, [_task_record()])
            _write_jsonl(attribution, [{"task_id": "hle/0", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "hle/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "hle/0", "run_id": "r1"}])
            raw = {
                "datasets": [{"name": "hle", "task_file": str(tasks)}],
                "inputs": {
                    "attribution_files": [str(attribution)],
                    "coalition_files": [str(coalition)],
                    "trace_files": [str(traces)],
                },
            }
            _preflight_required_inputs(root, raw)

    def test_preflight_does_not_require_score_or_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(tasks, [_task_record()])
            _write_jsonl(attribution, [{"task_id": "hle/0", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "hle/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "hle/0", "run_id": "r1"}])
            raw = {
                "datasets": [{"name": "hle", "task_file": str(tasks)}],
                "inputs": {
                    "score_files": [str(root / "missing_scores.jsonl")],
                    "task_metadata_file": str(root / "missing_metadata.jsonl"),
                    "attribution_files": [str(attribution)],
                    "coalition_files": [str(coalition)],
                    "trace_files": [str(traces)],
                },
            }
            _preflight_required_inputs(root, raw)

    def test_preflight_reports_all_missing_and_empty_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _touch_empty(tasks)
            _write_jsonl(attribution, [{"task_id": "hle/0"}])
            _touch_empty(coalition)
            raw = {
                "datasets": [{"name": "hle", "task_file": str(tasks)}],
                "inputs": {
                    "attribution_files": [str(attribution)],
                    "coalition_files": [str(coalition)],
                    "trace_files": [str(traces)],
                },
            }
            with self.assertRaises(FileNotFoundError) as ctx:
                _preflight_required_inputs(root, raw)
            message = str(ctx.exception)
            self.assertIn(str(tasks), message)
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
                task_file=root / "missing_tasks.jsonl",
                attribution_files=[root / "missing_attribution.jsonl"],
                coalition_files=[root / "missing_coalition.jsonl"],
                trace_files=[root / "missing_traces.jsonl"],
                out_dir=out_dir,
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                run_generalization(config)
            message = str(ctx.exception)
            self.assertIn("missing_tasks.jsonl", message)
            self.assertIn("missing_attribution.jsonl", message)
            self.assertIn("missing_coalition.jsonl", message)
            self.assertIn("missing_traces.jsonl", message)
            self.assertFalse(out_dir.exists())
            self.assertEqual(list(root.rglob("*.jsonl")), [])

    def test_run_generalization_uses_task_file_metadata_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            prompt = "short prompt text"
            _write_jsonl(tasks, [_task_record(prompt=prompt)])
            _write_jsonl(
                attribution,
                [
                    {
                        "task_id": "hle/0",
                        "dataset": "hle",
                        "method": "loo",
                        "role": "coder",
                        "score": 0.25,
                    }
                ],
            )
            _write_jsonl(coalition, [{"task_id": "hle/0", "coalition_id": "c1", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "hle/0", "run_id": "r1", "input_tokens": 4, "output_tokens": 4}])
            config = _write_exp08_config(
                root / "exp08.yaml",
                task_file=tasks,
                attribution_files=[attribution],
                coalition_files=[coalition],
                trace_files=[traces],
                out_dir=out_dir,
            )
            summary = run_generalization(config)
            self.assertEqual(summary["selected_tasks"], 1)
            self.assertEqual(summary["attribution_rows"], 1)
            self.assertEqual(summary["score_runs"], 0)
            self.assertTrue((out_dir / "summary.jsonl").is_file())
            self.assertGreater((out_dir / "summary.jsonl").stat().st_size, 0)
            features = [
                json.loads(line)
                for line in (out_dir / "features.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0]["prompt_length_bin"], "short_<=500")
            self.assertNotIn("single_agent_success_bin", features[0])
            self.assertNotIn("full_mas_success_bin", features[0])
            self.assertNotIn("dominant_failure_type", features[0])
            correlations = [
                json.loads(line)
                for line in (out_dir / "correlations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if correlations:
                self.assertEqual(correlations[0]["analysis_kind"], "descriptive_non_causal")
                self.assertFalse(correlations[0]["causal_claim"])

    def test_zero_attribution_rows_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(tasks, [_task_record()])
            _write_jsonl(attribution, [{"task_id": "hle/other", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "hle/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "hle/0", "run_id": "r1"}])
            config = _write_exp08_config(
                root / "exp08.yaml",
                task_file=tasks,
                attribution_files=[attribution],
                coalition_files=[coalition],
                trace_files=[traces],
                out_dir=out_dir,
            )
            with self.assertRaises(ValueError) as ctx:
                run_generalization(config)
            self.assertIn("0 attribution rows", str(ctx.exception))
            self.assertFalse(out_dir.exists())

    def test_banned_post_treatment_axes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "outputs"
            tasks = root / "hle_tasks.jsonl"
            attribution = root / "attribution.jsonl"
            coalition = root / "coalition.jsonl"
            traces = root / "traces.jsonl"
            _write_jsonl(tasks, [_task_record()])
            _write_jsonl(attribution, [{"task_id": "hle/0", "dataset": "hle", "method": "loo", "role": "coder", "score": 0.1}])
            _write_jsonl(coalition, [{"task_id": "hle/0", "run_id": "r1"}])
            _write_jsonl(traces, [{"task_id": "hle/0", "run_id": "r1"}])
            config = _write_exp08_config(
                root / "exp08.yaml",
                task_file=tasks,
                attribution_files=[attribution],
                coalition_files=[coalition],
                trace_files=[traces],
                out_dir=out_dir,
                extra_analysis="\n".join(
                    [
                        "  task_condition_axes:",
                        "    - dataset",
                        "    - single_agent_success_bin",
                    ]
                ),
            )
            with self.assertRaises(ValueError) as ctx:
                run_generalization(config)
            self.assertIn("single_agent_success_bin", str(ctx.exception))
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
