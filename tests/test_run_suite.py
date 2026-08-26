"""Offline tests for the thin parallel suite CLI."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_run_suite():
    path = ROOT / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_suite"] = module
    spec.loader.exec_module(module)
    return module


class FakeProc:
    def __init__(self, poll_values=None, wait_code=0):
        self._poll_values = list(poll_values or [])
        self.wait_code = wait_code
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        if self.terminated and self.returncode is None:
            self.returncode = -15
            return self.returncode
        if self._poll_values:
            value = self._poll_values.pop(0)
            self.returncode = value
            return value
        return self.returncode

    def wait(self, timeout=None):
        if timeout is not None:
            code = self.poll()
            if code is None:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            return code
        if self.returncode is None:
            self.returncode = self.wait_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.returncode is None:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def _write_config(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class RunSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_run_suite()

    def _argv(self, *args: str) -> list[str]:
        return ["run_suite.py", *args]

    def test_print_plan_resolves_aliases_and_temp_configs(self) -> None:
        stdout = io.StringIO()
        argv = self._argv("exp01_full_system", "--print-plan")
        with patch.dict(os.environ, {"MAS_FRESH_RUN": "0"}, clear=False):
            with patch.object(sys, "argv", argv), patch.object(sys, "stdout", stdout):
                code = self.mod.main()
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["jobs"], 1)
        self.assertEqual(len(payload["experiments"]), 1)
        self.assertEqual(payload["experiments"][0]["experiment_id"], "exp01_full_system")
        self.assertIn("run_experiment.py", " ".join(payload["experiments"][0]["cmd"]))
        self.assertTrue(payload["experiments"][0]["cache_dir"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_config(
                root,
                "a.yaml",
                "id: exp_a\noutputs:\n  run_dir: data/runs/a\n  score_file: data/results/scores/a.jsonl\n",
            )
            second = _write_config(
                root,
                "b.yaml",
                "id: exp_b\noutputs:\n  run_dir: data/runs/b\n  score_file: data/results/scores/b.jsonl\n",
            )
            stdout = io.StringIO()
            argv = self._argv(str(first), str(second), "--jobs", "2", "--max-tasks", "3", "--print-plan")
            with patch.dict(os.environ, {"MAS_FRESH_RUN": "0"}, clear=False):
                with patch.object(sys, "argv", argv), patch.object(sys, "stdout", stdout):
                    code = self.mod.main()
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["max_tasks"], 3)
            self.assertEqual([row["experiment_id"] for row in payload["experiments"]], ["exp_a", "exp_b"])
            self.assertTrue(all("--max-tasks" in row["cmd"] for row in payload["experiments"]))

    def test_rejects_duplicates_exp09_generalization_pair_fresh_run_jobs_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = _write_config(
                root,
                "shared.yaml",
                "id: exp_shared\noutputs:\n  run_dir: data/runs/shared\n",
            )
            other = _write_config(
                root,
                "other.yaml",
                "id: exp_other\noutputs:\n  run_dir: data/runs/shared/child\n",
            )
            exp09 = _write_config(root, "exp09.yaml", "id: exp09_contribution_predictor\noutputs: {}\n")
            exp08 = _write_config(root, "exp08.yaml", "id: exp08_generalization\noutputs:\n  summary_file: data/results/statistics/g.jsonl\n")
            ok = _write_config(root, "ok.yaml", "id: exp_ok\noutputs:\n  run_dir: data/runs/ok\n")

            with patch.dict(os.environ, {"MAS_FRESH_RUN": "0"}, clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(shared), str(shared)], jobs=2, max_tasks=None)
                self.assertIn("Duplicate config", str(ctx.exception))

                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(exp09)], jobs=1, max_tasks=None)
                self.assertIn("Exp09", str(ctx.exception))

                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(ok), str(exp08)], jobs=2, max_tasks=None)
                self.assertIn("Exp08", str(ctx.exception))

                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(ok), str(other), str(shared)], jobs=2, max_tasks=None)
                self.assertIn("at most 2", str(ctx.exception))

                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(ok)], jobs=3, max_tasks=None)
                self.assertIn("--jobs", str(ctx.exception))

                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(shared), str(other)], jobs=2, max_tasks=None)
                self.assertIn("Overlapping", str(ctx.exception))

            with patch.dict(os.environ, {"MAS_FRESH_RUN": "1"}, clear=False):
                with self.assertRaises(SystemExit) as ctx:
                    self.mod.build_plan([str(ok)], jobs=1, max_tasks=None)
                self.assertIn("MAS_FRESH_RUN", str(ctx.exception))

            with patch.dict(os.environ, {"MAS_FRESH_RUN": "0"}, clear=False):
                plan = self.mod.build_plan([str(exp08)], jobs=1, max_tasks=None)
            self.assertEqual(plan["experiments"][0]["experiment_id"], "exp08_generalization")

    def test_mocked_subprocess_uses_distinct_caches_and_stops_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_config(root, "a.yaml", "id: exp_a\noutputs:\n  run_dir: data/runs/a\n")
            second = _write_config(root, "b.yaml", "id: exp_b\noutputs:\n  run_dir: data/runs/b\n")
            with patch.dict(os.environ, {"MAS_FRESH_RUN": "0"}, clear=False):
                plan = self.mod.build_plan([str(first), str(second)], jobs=2, max_tasks=2)

            calls: list[tuple[list[str], dict[str, str]]] = []
            procs = [FakeProc(poll_values=[None, 7]), FakeProc(poll_values=[None, None])]

            def fake_popen(cmd, env):
                calls.append((list(cmd), dict(env)))
                return procs[len(calls) - 1]

            with patch.object(self.mod, "_wait_parallel", wraps=self.mod._wait_parallel):
                code = self.mod.execute_plan(plan, popen=fake_popen)
            self.assertEqual(code, 7)
            self.assertEqual(len(calls), 2)
            cache_a = calls[0][1]["MAS_LLM_CACHE_DIR"]
            cache_b = calls[1][1]["MAS_LLM_CACHE_DIR"]
            self.assertNotEqual(cache_a, cache_b)
            self.assertNotIn("MAS_FRESH_RUN", calls[0][1])
            self.assertTrue(all("run_experiment.py" in " ".join(cmd) for cmd, _env in calls))
            self.assertTrue(procs[1].terminated)

            sequential_plan = dict(plan)
            sequential_plan["jobs"] = 1
            seq_calls: list[tuple[list[str], dict[str, str]]] = []
            seq_proc = FakeProc(wait_code=4)

            def seq_popen(cmd, env):
                seq_calls.append((list(cmd), dict(env)))
                return seq_proc

            seq_code = self.mod.execute_plan(sequential_plan, popen=seq_popen)
            self.assertEqual(seq_code, 4)
            self.assertEqual(len(seq_calls), 1)


if __name__ == "__main__":
    unittest.main()
