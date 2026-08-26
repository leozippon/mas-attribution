"""Offline tests for external benchmark normalization and public views."""

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

from mas_contribution_bench.data.external import (
    DEFAULT_EXTERNAL_DATASETS,
    SOURCES,
    infer_answer,
    infer_prompt,
    infer_tests,
    normalize_dataset,
    public_task_view,
    resolve_gpqa_gold,
    sanitize_public_prompt,
)


class ExternalDescriptorTests(unittest.TestCase):
    def test_default_datasets_are_the_seven_phase1_names(self) -> None:
        self.assertEqual(
            list(DEFAULT_EXTERNAL_DATASETS),
            [
                "aime_2026",
                "gpqa_diamond",
                "hle",
                "arc_agi_2",
                "ifbench",
                "livecodebench",
                "swebench_verified",
            ],
        )
        self.assertNotIn("tau2_bench", DEFAULT_EXTERNAL_DATASETS)
        self.assertIn("tau2_bench", SOURCES)
        for name in DEFAULT_EXTERNAL_DATASETS:
            self.assertIn(name, SOURCES)
        self.assertEqual(SOURCES["ifbench"].hf_repos, ["allenai/IFBench_test"])
        self.assertIn("AllenAI IFBench", SOURCES["ifbench"].notes)
        self.assertNotIn("google/IFEval", SOURCES["ifbench"].hf_repos)
        self.assertNotIn("walledai/IFBench", SOURCES["ifbench"].hf_repos)


class ExternalNormalizationTests(unittest.TestCase):
    def test_arc_prompt_keeps_train_and_test_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            payload = {
                "train": [{"input": [[1, 1]], "output": [[7, 7]]}],
                "test": [{"input": [[3, 3]], "output": [[99, 99]]}],
            }
            (raw_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")
            out_file = Path(tmp) / "arc.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["arc_agi_2"])
            self.assertEqual(count, 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("[[1, 1]]", row["prompt"])
            self.assertIn("[[7, 7]]", row["prompt"])
            self.assertIn("[[3, 3]]", row["prompt"])
            self.assertNotIn("99", row["prompt"])
            self.assertIsNone(row["tests"])
            self.assertEqual(row["metadata"]["expected_test_outputs"], [[[99, 99]]])
            self.assertEqual(json.loads(row["reference_solution"]), [[[99, 99]]])

    def test_gpqa_gold_from_correct_answer_field(self) -> None:
        record = {
            "Question": "Which element?",
            "Correct Answer": "Helium",
            "Incorrect Answer 1": "Neon",
            "Incorrect Answer 2": "Argon",
            "Incorrect Answer 3": "Xenon",
            "solution": "long explanation that is not the gold label",
        }
        self.assertEqual(resolve_gpqa_gold(record), "Helium")
        self.assertEqual(infer_answer(record, SOURCES["gpqa_diamond"]), "Helium")
        prompt = infer_prompt(record, SOURCES["gpqa_diamond"])
        self.assertIn("Which element?", prompt)
        self.assertIn("Helium", prompt)
        self.assertIn("Neon", prompt)

    def test_livecodebench_private_tests_stay_on_record(self) -> None:
        record = {
            "question_content": "Write foo",
            "starter_code": "def foo():\n    pass\n",
            "private_tests": "assert foo() == 1",
        }
        prompt = infer_prompt(record, SOURCES["livecodebench"])
        self.assertIn("Write foo", prompt)
        self.assertNotIn("assert foo", prompt)
        self.assertEqual(infer_tests(record, SOURCES["livecodebench"]), "assert foo() == 1")

    def test_unmarked_gpqa_row_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw" / "hf" / "Idavidrein__gpqa"
            raw_dir.mkdir(parents=True)
            record = {
                "Question": "Which element?",
                "Correct Answer": "Helium",
                "Incorrect Answer 1": "Neon",
                "Incorrect Answer 2": "Argon",
                "Incorrect Answer 3": "Xenon",
            }
            (raw_dir / "gpqa_main.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            out_file = Path(tmp) / "gpqa.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["gpqa_diamond"])
            self.assertEqual(count, 0)
            self.assertEqual(out_file.read_text(encoding="utf-8").strip(), "")

    def test_canonical_raw_directory_does_not_mark_main_gpqa_as_diamond(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "data" / "raw" / "gpqa_diamond"
            dest = raw_dir / "hf" / "Idavidrein__gpqa"
            dest.mkdir(parents=True)
            record = {
                "Question": "Which element?",
                "Correct Answer": "Helium",
                "Incorrect Answer 1": "Neon",
                "Incorrect Answer 2": "Argon",
                "Incorrect Answer 3": "Xenon",
            }
            (dest / "gpqa_main.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            out_file = Path(tmp) / "gpqa.jsonl"
            self.assertEqual(normalize_dataset(raw_dir, out_file, SOURCES["gpqa_diamond"]), 0)

    def test_diamond_repo_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            dest = raw_dir / "hf" / "m-a-p__GPQA-Diamond"
            dest.mkdir(parents=True)
            record = {
                "Question": "Which element?",
                "Correct Answer": "Helium",
                "Incorrect Answer 1": "Neon",
                "Incorrect Answer 2": "Argon",
                "Incorrect Answer 3": "Xenon",
            }
            (dest / "data.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            out_file = Path(tmp) / "gpqa.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["gpqa_diamond"])
            self.assertEqual(count, 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["reference_solution"], "Helium")

    def test_dataset_info_and_config_sidecars_yield_zero(self) -> None:
        sidecar = {
            "description": "GPQA is a graduate-level QA dataset.",
            "citation": "Rein et al.",
            "homepage": "https://example.com",
            "features": {"Question": {"dtype": "string"}},
            "splits": {"test": {"num_examples": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "dataset_info.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (raw_dir / "config.json").write_text(json.dumps(sidecar), encoding="utf-8")
            out_file = Path(tmp) / "gpqa.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["gpqa_diamond"])
            self.assertEqual(count, 0)
            self.assertEqual(out_file.read_text(encoding="utf-8").strip(), "")

    def test_description_only_metadata_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "meta.jsonl").write_text(
                json.dumps({"description": "Humanity's Last Exam", "instance": "row-1"}) + "\n",
                encoding="utf-8",
            )
            out_file = Path(tmp) / "hle.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["hle"])
            self.assertEqual(count, 0)
            self.assertEqual(out_file.read_text(encoding="utf-8").strip(), "")

    def test_aime_without_answer_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "aime.jsonl").write_text(
                json.dumps({"problem": "Compute 1+1"}) + "\n",
                encoding="utf-8",
            )
            out_file = Path(tmp) / "aime.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["aime_2026"])
            self.assertEqual(count, 0)

    def test_hle_media_rows_are_dropped_for_text_only_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            records = [
                {"question": "Use the image.", "answer": "x", "image": {"path": "x.png"}},
                {"question": "Text-only question.", "answer": "y", "image": None},
            ]
            (raw_dir / "hle.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            out_file = Path(tmp) / "hle.jsonl"
            self.assertEqual(normalize_dataset(raw_dir, out_file, SOURCES["hle"]), 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("Text-only", row["prompt"])

    def test_arc_official_training_paths_are_excluded(self) -> None:
        payload = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            training = raw_dir / "github" / "ARC-AGI-2-main" / "data" / "training"
            evaluation = raw_dir / "github" / "ARC-AGI-2-main" / "data" / "evaluation"
            training.mkdir(parents=True)
            evaluation.mkdir(parents=True)
            (training / "train.json").write_text(json.dumps(payload), encoding="utf-8")
            (evaluation / "eval.json").write_text(json.dumps(payload), encoding="utf-8")
            out_file = Path(tmp) / "arc.jsonl"
            self.assertEqual(normalize_dataset(raw_dir, out_file, SOURCES["arc_agi_2"]), 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("evaluation", row["source"]["raw_file_path"])

    def test_livecodebench_and_swe_unscored_records_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "lcb.jsonl").write_text(
                json.dumps({"question_content": "Write foo", "starter_code": "def foo():\n    pass\n"}) + "\n",
                encoding="utf-8",
            )
            out_file = Path(tmp) / "lcb.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["livecodebench"])
            self.assertEqual(count, 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["reference_solution"])

            swe_dir = Path(tmp) / "swe"
            swe_dir.mkdir()
            (swe_dir / "swe.jsonl").write_text(
                json.dumps({"problem_statement": "Fix the bug", "instance_id": "repo__1"}) + "\n",
                encoding="utf-8",
            )
            swe_out = Path(tmp) / "swe.jsonl"
            count = normalize_dataset(swe_dir, swe_out, SOURCES["swebench_verified"])
            self.assertEqual(count, 1)
            swe_row = json.loads(swe_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("Fix the bug", swe_row["prompt"])

    def test_ifbench_prompt_and_constraints_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            (raw_dir / "ifbench.jsonl").write_text(
                json.dumps(
                    {
                        "prompt": "Write a reply with two numbers.",
                        "instruction_id_list": ["count:numbers"],
                        "kwargs": [{"num_numbers": 2}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_file = Path(tmp) / "ifbench.jsonl"
            count = normalize_dataset(raw_dir, out_file, SOURCES["ifbench"])
            self.assertEqual(count, 1)
            row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("two numbers", row["prompt"])
            self.assertEqual(row["metadata"]["instruction_id_list"], ["count:numbers"])


class PublicTaskViewTests(unittest.TestCase):
    def test_legacy_json_prompt_drops_gold_and_arc_test_outputs(self) -> None:
        leaked = json.dumps(
            {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[88]]}],
                "Correct Answer": "SECRET_GOLD",
            }
        )
        cleaned = sanitize_public_prompt(leaked, dataset="arc_agi_2")
        self.assertIn("[[0]]", cleaned)
        self.assertIn("[[1]]", cleaned)
        self.assertIn("[[2]]", cleaned)
        self.assertNotIn("88", cleaned)
        self.assertNotIn("SECRET_GOLD", cleaned)

        view = public_task_view(
            {
                "task_id": "arc_agi_2/x",
                "dataset": "arc_agi_2",
                "prompt": leaked,
                "tests": "SECRET_TEST",
                "reference_solution": "SECRET_REF",
                "metadata": {"Correct Answer": "SECRET_META", "expected_test_outputs": [[[88]]]},
            }
        )
        self.assertNotIn("tests", view)
        self.assertNotIn("reference_solution", view)
        self.assertNotIn("metadata", view)
        self.assertNotIn("SECRET_TEST", json.dumps(view))
        self.assertNotIn("SECRET_REF", json.dumps(view))
        self.assertNotIn("SECRET_META", json.dumps(view))
        self.assertNotIn("88", view["prompt"])


if __name__ == "__main__":
    unittest.main()
