# MASContributionBench

MASContributionBench is a benchmark framework for studying agent-level contribution attribution in LLM-based multi-agent systems. It evaluates which agents matter, why they matter, and how contribution changes under task, topology, role, permission, and communication interventions.

## Core Idea

1. **Tasks**: HumanEval, MBPP, TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite.
2. **Architectures**: PEV, Chain, DAG, MetaGPT-lite, Star, Debate, and Graph coordination.
3. **Agents**: planner, coder, verifier, critic, supervisor, finalizer, and researcher.
4. **Attribution**: Wired experiment runners estimate LOO, Shapley, and Banzhaf. Myerson, Owen, and exact methods remain unwired and are rejected. External full-system donor reuse is disabled; checkpoint IDs are executable- and treatment-aware, which invalidates older IDs.

## Directory Contract

- `data/raw/`: downloaded source datasets, kept as close to original as possible.
- `data/processed/`: unified JSONL task format and metadata.
- `data/runs/`: raw experiment records for each run.
- `data/traces/`: message-level and tool-level MAS traces.
- `configs/`: benchmark, agent, permission, prompt, and experiment definitions.
- `src/`: reusable benchmark implementation.
- `scripts/`: command-line entry points.
- `outputs/`: paper-facing results, tables, figures, and reports.

## Evaluation and experiment dispatch

- Real HumanEval/MBPP `pass@1` requires code execution (`--execute-code`). Formal runs should use Docker (`--sandbox-backend docker`). The evaluator will not emit a unit-test `pass@1` record from text equality when execution is off.
- `exp08_generalization` is analysis-only and requires completed, non-empty upstream metadata, score, attribution, coalition, and trace artifacts.
- `exp09_contribution_predictor` is a planned design and intentionally has no runner; the CLI refuses it instead of falling through to another experiment.
