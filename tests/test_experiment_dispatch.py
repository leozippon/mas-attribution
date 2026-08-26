"""Regression tests for experiment CLI dispatch."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_run_experiment():
    path = ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = module
    spec.loader.exec_module(module)
    return module


class ExperimentDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_run_experiment()

    def test_exp01_and_exp08_resolve_to_expected_runners(self) -> None:
        self.assertIs(self.mod.infer_runner("exp01_full_system", "auto"), self.mod.run_full_system)
        self.assertIs(self.mod.infer_runner("exp08_generalization", "auto"), self.mod.run_generalization)

    def test_exp09_refuses_auto_and_forced_modes(self) -> None:
        for mode in ("auto", "full", "single", "attribution", "intervention"):
            with self.subTest(mode=mode):
                with self.assertRaises(SystemExit) as ctx:
                    self.mod.infer_runner("exp09_contribution_predictor", mode)
                message = str(ctx.exception)
                self.assertIn("no runner", message)
                self.assertIn("exp09_contribution_predictor", message)

    def test_print_config_returns_before_exp09_dispatch(self) -> None:
        argv = [
            "run_experiment.py",
            "--experiment",
            "exp09_contribution_predictor",
            "--print-config",
        ]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(sys, "stdout", stdout):
            code = self.mod.main()
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["experiment_id"], "exp09_contribution_predictor")


if __name__ == "__main__":
    unittest.main()
