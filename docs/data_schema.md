# MASContributionBench Data Schema

This document defines the unified data schema used by
MASContributionBench. The goal is to normalize task inputs and make
multi-agent contribution attribution reproducible.

The benchmark stores six record types:

```text
TaskRecord          unified task input
ConfigRecord        architecture, agent, role, model, protocol, and permission setup
RunRecord           one execution of a team/coalition on one task
TraceRecord         message-level and tool-level interaction logs
EvaluationRecord    final answer, evaluator settings, score, and failure type
AttributionRecord   LOO / sampled Shapley / sampled Banzhaf scores
```

All long-running experiment outputs are JSONL. Raw datasets remain in
`data/raw/`. Converted task files are written to
`data/processed/tasks/`. Run and trace logs are written under
experiment-owned directories such as `data/runs/exp01` and
`data/traces/exp01`. Paper-facing aggregates may be written to
`outputs/`. Generated processed metadata, summaries, runs, traces,
results, caches, and logs stay untracked.

## Canonical datasets

The ordered seven-dataset suite is:

```text
aime_2026
gpqa_diamond
hle
arc_agi_2
ifbench
livecodebench
swebench_verified
```

IFBench uses the official AllenAI test set `allenai/IFBench_test` and
the pinned `ifbench==0.2.0` verifier. Local in-loop scoring covers the
first five. LiveCodeBench and
SWE-bench Verified are prediction/export-only in-loop:

```text
score = None
passed = None
attribution_eligible = false
```

until official results are joined. HumanEval and MBPP are not
canonical.

Every canonical experiment entry uses `max_tasks: 30`,
`task_selection: seeded_random`, and experiment
`sampling.selection_seed: 0`.

## 1. TaskRecord

`TaskRecord` is the normalized input format. Agent prompts receive a
public view of the task: `prompt`, `context`, `output_format`, and
`entry_point` after gold/private fields are stripped. Tests, reference
solutions, ARC test outputs, and raw metadata stay off the public view.

Required fields:

```text
task_id
dataset
split
task_type
prompt
evaluation
mas_metadata
contribution_metadata
source
```

Recommended JSONL example:

```json
{
  "task_id": "hle/000001",
  "dataset": "hle",
  "split": "test",
  "task_type": "general_mas",
  "prompt": "Answer the question ...",
  "context": null,
  "input_format": null,
  "output_format": "free_form_or_multiple_choice",
  "entry_point": null,
  "reference_solution": "stored privately, not shown to the model",
  "tests": null,
  "evaluation": {
    "evaluator_type": "exact_match",
    "metric": "answer_accuracy",
    "timeout_seconds": 30,
    "sandbox": null,
    "official_evaluator_version": null,
    "extra": {}
  },
  "mas_metadata": {
    "requires_planning": true,
    "requires_coding": false,
    "requires_verification": true,
    "requires_research": true,
    "requires_tool_use": false,
    "estimated_agents": ["planner", "researcher", "finalizer"]
  },
  "difficulty": {
    "source": "external_benchmark_default",
    "level": "unknown",
    "single_agent_score": null,
    "input_length": 120,
    "constraint_count": null,
    "requires_cross_file_edit": false
  },
  "contribution_metadata": {
    "eligible_roles": ["planner", "coder", "verifier", "critic"],
    "default_architectures": ["A1_pev", "A2_chain", "A3_dag", "A5_star"],
    "permission_requirements": ["read_task", "write_solution"],
    "intervention_tags": ["topology", "role", "permission"]
  },
  "source": {
    "raw_dataset": "Humanity's Last Exam",
    "raw_task_id": "example",
    "raw_file_path": "data/raw/hle/hf/cais__hle/data.parquet",
    "license": null,
    "conversion_version": "external_v1"
  },
  "metadata": {}
}
```

## 2. ConfigRecord

`ConfigRecord` captures the complete experimental condition:

```text
A = (G, R, P, M, Pi, T)
```

It should save architecture, design graph, agents, roles, models,
permissions, protocol, prompt set, and task filter.

```text
design_graph      graph specified by the architecture template
trace_graph       graph reconstructed from runtime messages
dependency_graph  graph inferred from downstream information use
```

Only `design_graph` belongs in `ConfigRecord`. The current graph
builder executes each role once. A5/A6/A7 IDs remain for compatibility,
but they are single-pass design graphs, not multi-round debate, star,
or graph engines.

## 3. RunRecord

Required fields:

```text
run_id
experiment_id
task_id
dataset
architecture_id
seed
coalition
removal
config_hash
prompt_hash
started_at
status
```

`config_hash` is a deterministic hash of the executable bundle: the
experiment YAML plus loaded permission sets, agent specs including
prompt text, and architecture specs. Runtime treatment (resolved model
backend/name, code execution, sandbox backend, and removal protocol) is
stored as an execution fingerprint and is part of run, coalition-cache,
and attribution IDs. Request seed is part of the LLM cache key.

The wired removal protocol for canonical attribution is
`null_agent_replacement`. Other protocol names exist in the schema but
are not advertised as wired sensitivity methods.

## 4. TraceRecord

Trace records should include run_id, task_id, event_id, event_type,
sender, receiver, role, timestamp, token counts, content, and optional
tool or intermediate-output fields.

## 5. EvaluationRecord

Required fields:

```text
run_id
task_id
final_answer
score
metric
passed
failure_type
evaluator
cost
```

For LiveCodeBench and SWE-bench Verified, `score` and `passed` remain
`None` in-loop. Those records are export/prediction artifacts, not
attribution inputs.

## 6. AttributionRecord

Required fields:

```text
attribution_id
experiment_id
task_id
architecture_id
agent_id
role
method
utility_type
score
baseline_score
coalition
removal_protocol
```

Canonical wired methods are `loo`, `shapley_sampled`, and
`banzhaf_sampled`. Exact, Myerson, and Owen estimators remain unwired
and are rejected by experiment runners.

## Output layout

```text
data/processed/tasks/*_tasks.jsonl
data/runs/exp01/runs.jsonl
data/traces/exp01/exp01_full_system_traces.jsonl
data/results/exp01/scores.jsonl
data/results/exp03/loo_attribution.jsonl
data/results/exp08/summary.jsonl
```

Exp01–Exp08 own disjoint run/trace/result directories so
`scripts/run_suite.py` can pair any two non-Exp08 configs.
