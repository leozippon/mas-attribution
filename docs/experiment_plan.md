# Experiment Plan

## Stage 1: Data construction

Normalize the ordered seven-dataset suite:

`aime_2026`, `gpqa_diamond`, `hle`, `arc_agi_2`, `ifbench`,
`livecodebench`, `swebench_verified`.

Use `scripts/prepare_external_benchmarks.py`. Nonempty raw and
processed assets resume unless `--force` is set. Fail-fast is default;
`--no-fail-fast` continues. `--max-files` and `--max-records` are
smoke-test bounds.

Every canonical experiment entry uses `max_tasks: 30`,
`task_selection: seeded_random`, and `sampling.selection_seed: 0`.

## Stage 2: Full-system and baseline performance

Exp01 runs architectures A1–A7 on all seven datasets. Exp02 runs
single-agent, solo-role, and random-team baselines on the same subset.

Local in-loop scores exist for the first five datasets. LiveCodeBench
and SWE-bench Verified are prediction/export-only (`score=None`,
`passed=None`) and are not attribution-eligible until official results
are joined.

The model endpoint is an externally managed local Qwen server
(`qwen-local` at `http://127.0.0.1:8000/v1`). This repository does not
launch it.

## Stage 3: Contribution attribution

Exp03 computes LOO with null-agent replacement only on the five
locally scored datasets. Unwired sensitivity protocols are not
advertised.

Exp04 compares modest sampled Shapley and sampled Banzhaf on the same
five datasets and 30-task subsets.

External full-system donor reuse is disabled. Checkpoint IDs are
executable- and treatment-aware.

## Stage 4: Controlled interventions

Exp05 keeps roles and position permissions fixed and changes only
single-pass controlled chain, star, and DAG topology. There is no
debate variant and no fallback policy. Attribution is LOO only.

Exp06 swaps only Planner and Coder functional/prompt assignments on
A2_chain, A3_dag, and A5_star. Topology and graph-position permissions
stay fixed. Attribution is LOO only.

Exp07 uses A2_chain and A3_dag with strict enforcement and only
`write_solution` and `final_answer` conditions. Attribution is LOO
only.

The graph builder executes each role once. A5/A6/A7 keep their IDs for
compatibility but do not claim multi-round debate, star, or graph
execution.

## Stage 5: Analysis

Exp08 is analysis-only and descriptive. It derives selected-task
metadata from configured normalized task files. Score files are not
covariate inputs. Communication summaries and correlations, when
emitted, are labeled descriptive and non-causal. Exp08 does not support
causal or inferential claims.

Exp09 is a planned contribution-predictor design. It has no runner and
is refused by the CLI.

`scripts/run_suite.py` can pair any two non-Exp08 generating configs
because Exp01–Exp08 own disjoint run/trace/result directories. Suite
concurrency is at most 2. Exp08 must run alone after upstream
artifacts exist. Exp09 cannot be dispatched.
