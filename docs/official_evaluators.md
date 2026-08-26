# Official Evaluator Integration Notes

This project separates two stages:

1. MAS generation produces `EvaluationRecord` JSONL files containing `final_answer`.
2. Official benchmark harnesses consume exported prediction files and produce official scores.

The separation is necessary because official harnesses such as SWE-bench are batch evaluators that build Docker images, restore repositories, apply patches, and run test suites. They should not be invoked once per MAS run inside `evaluate_task_output`.

## Currently wired

- HumanEval: local Docker Python unit tests.
- MBPP: local Docker Python unit tests.
- AIME/HLE/GPQA: normalized exact-match fallback.
- IFBench: partial rule-based fallback.
- SWE-bench Lite / SWE-bench Verified: official prediction export plus `swebench.harness.run_evaluation` launcher.

## Export predictions

```bash
python scripts/export_official_predictions.py \
  --score-file data/results/scores/qwen_all_tasksets_full_system_scores.jsonl \
  --output-dir data/results/official_predictions/qwen_all_exp01 \
  --model-name qwen-local
```

## Run SWE-bench official harness

```bash
python scripts/run_official_evaluators.py \
  --prediction-dir data/results/official_predictions/qwen_all_exp01 \
  --dataset swebench_verified \
  --run-id qwen_all_exp01_swe_verified \
  --max-workers 1 \
  --cache-level env
```

Use `--dry-run` first to print the exact command without launching Docker builds.

## Not yet official-scored

LiveCodeBench, TeamBench, ARC-AGI-2, tau2-bench, MARBLE, and MultiAgentBench require their benchmark-specific runners/environments. The export script writes prediction scaffolds, but formal scores require integrating those official packages or restoring missing labels/workspaces.
