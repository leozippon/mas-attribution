# MASContributionBench Data Schema

This document defines the unified data schema used by MASContributionBench. The goal is not only to normalize task inputs from different datasets, but also to make multi-agent contribution attribution reproducible and explainable.

The benchmark stores six record types:

```text
TaskRecord          unified task input
ConfigRecord        architecture, agent, role, model, protocol, and permission setup
RunRecord           one execution of a team/coalition on one task
TraceRecord         message-level and tool-level interaction logs
EvaluationRecord    final answer, evaluator settings, score, and failure type
AttributionRecord   LOO/Shapley/Banzhaf/Myerson/Owen contribution scores
```

All long-running experiment outputs should be written as JSONL. Each line is one record. Raw datasets remain in `data/raw/`; converted task files are written to `data/processed/tasks/`; run and trace logs are written to `data/runs/` and `data/traces/`; paper-facing aggregates are written to `outputs/`.

## 1. TaskRecord

`TaskRecord` is the normalized input format for HumanEval, MBPP, TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite. It should contain enough information for any architecture to run the task and for any evaluator to score the final answer.

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
  "task_id": "humaneval/000001",
  "dataset": "humaneval",
  "split": "test",
  "task_type": "code_generation",
  "prompt": "Write a function ...",
  "context": null,
  "input_format": null,
  "output_format": "python_function",
  "entry_point": "has_close_elements",
  "reference_solution": "def has_close_elements(...): ...",
  "tests": "assert has_close_elements(...)",
  "evaluation": {
    "evaluator_type": "unit_test",
    "metric": "pass_at_1",
    "timeout_seconds": 10,
    "sandbox": "python",
    "official_evaluator_version": null,
    "extra": {}
  },
  "mas_metadata": {
    "requires_planning": true,
    "requires_coding": true,
    "requires_verification": true,
    "requires_research": false,
    "requires_tool_use": true,
    "estimated_agents": ["planner", "coder", "verifier"]
  },
  "difficulty": {
    "source": "single_agent_baseline",
    "level": "unknown",
    "single_agent_score": null,
    "input_length": 120,
    "constraint_count": null,
    "requires_cross_file_edit": false
  },
  "contribution_metadata": {
    "eligible_roles": ["planner", "coder", "verifier", "critic"],
    "default_architectures": ["A1_pev", "A2_chain", "A3_dag", "A5_star"],
    "permission_requirements": ["read_task", "write_solution", "run_tests"],
    "intervention_tags": ["topology", "role", "permission"]
  },
  "source": {
    "raw_dataset": "HumanEval",
    "raw_task_id": "HumanEval/0",
    "raw_file_path": "data/raw/humaneval/openai_humaneval/test-00000-of-00001.parquet",
    "license": null,
    "conversion_version": "v1"
  },
  "metadata": {}
}
```

## 2. ConfigRecord

`ConfigRecord` captures the complete experimental condition. It corresponds to the paper's structured MAS definition:

```text
A = (G, R, P, M, Pi, T)
```

It should save:

```text
architecture_id
design_graph
agents
roles
models
permissions
protocol
prompt_set
task_filter
```

Important distinction:

```text
design_graph      graph specified by the architecture template
trace_graph       graph reconstructed from runtime messages
dependency_graph  graph inferred from downstream information use
```

Only `design_graph` belongs in `ConfigRecord`. `trace_graph` and `dependency_graph` are produced after execution.

## 3. RunRecord

`RunRecord` represents one execution of one architecture or coalition on one task. It must be enough to reproduce the run and link it to trace and evaluation outputs.

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

`config_hash` is a deterministic hash of the executable bundle: the experiment YAML plus loaded permission sets, agent specs including prompt text, and architecture specs. Runtime treatment (resolved model backend/name, code execution, sandbox backend, and removal protocol) is stored as an execution fingerprint and is part of run, coalition-cache, and attribution IDs. Existing checkpoints from YAML-only hashes are therefore invalid.

The `coalition` and `removal` fields are essential for LOO, Shapley, Banzhaf, Myerson, and Owen attribution.

Removal protocols:

```text
hard_removal
bypass_removal
null_agent_replacement
weak_model_replacement
none
```

## 4. TraceRecord

`TraceRecord` stores agent-level process data. It may be one JSONL row per message/tool event, or one row per run containing nested events. For large experiments, one row per event is easier to stream and analyze.

Trace records should include:

```text
run_id
task_id
event_id
event_type
sender
receiver
role
timestamp
input_tokens
output_tokens
content
tool_call
intermediate_output
```

Trace-derived features should be stored separately in `outputs/results/features/` after aggregation:

```text
messages_sent
messages_received
tokens_sent
tokens_received
avg_message_length
tool_call_count
tool_success_rate
revision_count
quoted_by_downstream
role_violation_count
latency_seconds
```

## 5. EvaluationRecord

`EvaluationRecord` stores final outputs and scoring details.

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

Common failure types:

```text
none
wrong_answer
test_failure
syntax_error
runtime_error
timeout
tool_error
protocol_error
role_violation
empty_output
invalid_format
evaluator_error
unknown
```

## 6. AttributionRecord

`AttributionRecord` stores contribution scores for one agent under one attribution method.

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

For sampled Shapley or Banzhaf, also save:

```text
sample_id
num_samples
permutation_order
sampling_seed
standard_error
confidence_interval
```

Supported methods:

```text
loo
shapley_exact
shapley_sampled
banzhaf_exact
banzhaf_sampled
myerson
owen
```

Schema method names include exact/Myerson/Owen estimators, but experiment runners reject those methods because they are not wired.

Supported utility types:

```text
task
net
process
```

## Minimal First-Stage Requirement

For the first working version, implement these files first:

```text
data/processed/tasks/humaneval_tasks.jsonl
data/processed/tasks/mbpp_tasks.jsonl
data/processed/metadata/task_metadata.jsonl
data/processed/metadata/task_difficulty.jsonl
data/runs/full_system/*.jsonl
data/traces/humaneval/*.jsonl
data/traces/mbpp/*.jsonl
outputs/results/scores/*.jsonl
outputs/results/attribution/*.jsonl
```

The first-stage experiments only need HumanEval and MBPP. TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite can be added after the pipeline is stable.

