# MASContributionBench

MASContributionBench studies agent-level contribution attribution in
LLM-based multi-agent systems. The canonical suite uses seven datasets,
a fixed seeded task subset, an externally managed local Qwen endpoint,
and nine experiment IDs (`exp01`–`exp09`).

## Canonical datasets

Ordered seven:

1. `aime_2026`
2. `gpqa_diamond`
3. `hle`
4. `arc_agi_2`
5. `ifbench`
6. `livecodebench`
7. `swebench_verified`

Local in-loop scoring is implemented for the first five. HLE preparation
keeps text-only rows because the configured endpoint is text-only; ARC-AGI-2
preparation uses public evaluation tasks and excludes training tasks.
LiveCodeBench and SWE-bench Verified are prediction/export-only in the
generation loop: `score=None`, `passed=None`, and they are not eligible for
attribution until official results are joined.

HumanEval and MBPP are not part of the canonical suite.

Exp01 and Exp02 include all seven datasets. Exp03–Exp07 and Exp08 use
only the five locally scored datasets. Every canonical dataset entry
selects `max_tasks: 30` with `task_selection: seeded_random` and
`sampling.selection_seed: 0`.

## Prompt leakage and identity

Agent prompts receive a public task view only. Tests, gold answers, ARC
test outputs, and other private fields stay off the model-facing
messages even if a processed record still stores them.

Task subsets are seed-aware and independent of run seeds. LLM cache keys
and run/attribution IDs include the request seed and the executable
treatment, so a seed or runtime-treatment change cannot reuse another
condition's cache or checkpoint.

## Local Qwen endpoint

Canonical model-generating configs (Exp01–Exp07) target an externally
managed OpenAI-compatible server:

- backend: `vllm`
- `base_url`: `http://127.0.0.1:8000/v1`
- `api_key`: `dummy`
- served name: `qwen-local`
- weights: `/Data/public/Qwen3.8-27B-FP8`
- context length: `262144`
- tensor parallel: `2`
- `max-num-seqs`: `2`
- managed by `zhangqy`

This repository does not launch or manage vLLM. Request `max_tokens` is
an honest completion cap (2048), not a claim that the server context is
that small. Model request failures abort by default. Convert a failed
request into a scored failure record only with explicit
`MAS_LLM_ERROR_FALLBACK=1`.

## Experiments

| ID | Role | Scope |
| --- | --- | --- |
| Exp01 | Full MAS | All seven datasets, architectures A1–A7 |
| Exp02 | Baselines | All seven datasets, single-agent / solo-role / random-team |
| Exp03 | LOO | Five local datasets, null-agent replacement only |
| Exp04 | Sampled Shapley / Banzhaf | Five local datasets, modest sample counts |
| Exp05 | Topology | Fixed roles and permissions; single-pass chain/star/DAG; LOO only |
| Exp06 | Role | Planner↔Coder functional/prompt swap; topology and position permissions stay fixed; A2/A3/A5; LOO only |
| Exp07 | Permission | A2/A3; strict `write_solution` and `final_answer` only; LOO only |
| Exp08 | Analysis | Descriptive summaries only; no LLM; no causal or inferential claim |
| Exp09 | Design only | Unimplemented; CLI refused |

The graph builder executes each role once. A5/A6/A7 keep their IDs, but
their YAML no longer claims multi-round debate, star, or graph replay.

## Prepare data

Canonical preparation is `scripts/prepare_external_benchmarks.py`.

```bash
python scripts/prepare_external_benchmarks.py
python scripts/prepare_external_benchmarks.py --dataset aime_2026 --max-records 30
python scripts/prepare_external_benchmarks.py --force --max-files 8
python scripts/prepare_external_benchmarks.py --no-fail-fast
```

Resume is the default: nonempty downloaded raw assets and nonempty
processed task files are reused unless `--force` is set. Fail-fast is
the default; `--no-fail-fast` continues after a dataset failure.
`--max-files` caps Hugging Face files per dataset. `--max-records` caps
converted rows for smoke conversion.

Processed tasks, metadata, summaries, runs, traces, results, caches,
logs, and other local state stay untracked.

## Run experiments

```bash
python scripts/run_experiment.py --experiment exp01_full_system --print-config
python scripts/run_experiment.py --experiment exp03_loo_attribution --max-tasks 1
python scripts/run_suite.py exp01_full_system exp02_single_agent_baseline --jobs 2 --print-plan
```

`scripts/run_suite.py` accepts at most two configs and at most two
concurrent children. It rejects overlapping declared output paths,
inherited `MAS_FRESH_RUN`, Exp09, and any pair that includes Exp08.
Run Exp08 from `run_experiment.py` after upstream artifacts exist. Exp09
has no runner.

Exp01–Exp08 write to experiment-owned directories such as
`data/runs/exp01`, `data/traces/exp01`, and `data/results/exp01/`.

## Directory contract

- `data/raw/`: downloaded sources
- `data/processed/tasks/`: normalized JSONL task files
- `data/runs/`, `data/traces/`, `data/results/`: experiment-owned outputs
- `configs/`: architecture, agent, permission, prompt, and experiment YAML
- `src/`: reusable implementation
- `scripts/`: CLI entry points

See `docs/benchmark_protocol.md`, `docs/data_schema.md`,
`docs/experiment_plan.md`, and `docs/official_evaluators.md`.
