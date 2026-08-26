# Official Evaluator Integration Notes

This project separates two stages:

1. MAS generation produces `EvaluationRecord` JSONL files containing
   `final_answer`.
2. Official benchmark harnesses consume exported prediction files and
   produce official scores.

Official harnesses such as SWE-bench are batch evaluators. They should
not be invoked once per MAS run inside `evaluate_task_output`.

## Local in-loop scoring

These five canonical datasets are scored inside the generation loop:

- AIME 2026: integer exact match
- GPQA Diamond: normalized exact match, including gold stored under
  common metadata keys
- HLE: normalized exact match on text-only rows; preparation excludes
  media-bearing tasks because the configured Qwen endpoint is text-only
- ARC-AGI-2: exact grid match on public evaluation tasks against expected
  test outputs stored off the public prompt; training paths are excluded
- IFBench: official AllenAI IFBench strict verifier (`ifbench==0.2.0`)
  over `allenai/IFBench_test`. Prompt-level score is 1 iff every
  instruction ID passes `instructions_registry.INSTRUCTION_DICT`.
  Missing package, missing instruction IDs, or unknown IDs fail closed.
  Empty model output is an empty-output failure without importing the
  verifier.

## Prediction/export-only in-loop

LiveCodeBench and SWE-bench Verified stay:

```text
score = None
passed = None
```

in `evaluate_task_output`. They are not attribution-eligible until
official results are joined. Do not treat a later official score as if
it had been available during LOO, Shapley, or intervention runs.

HumanEval and MBPP unit-test helpers still exist in the codebase but
are not part of the canonical suite.

## Export predictions

```bash
python scripts/export_official_predictions.py \
  --score-file data/results/exp01/scores.jsonl \
  --output-dir data/results/exp01/official_predictions \
  --model-name qwen-local
```

The generation endpoint is an externally managed local Qwen server.
This repository does not launch vLLM.

## Run SWE-bench official harness

```bash
python scripts/run_official_evaluators.py \
  --prediction-dir data/results/exp01/official_predictions \
  --dataset swebench_verified \
  --run-id exp01_swe_verified \
  --max-workers 1 \
  --cache-level env
```

Use `--dry-run` first to print the command without launching Docker
builds.

## Not yet official-scored in-loop

LiveCodeBench still needs its official package/environment. The export
script can write prediction scaffolds. Formal scores require that
external harness, then a join step before any attribution use.
