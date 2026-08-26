# Experiment Plan

## Stage 1: Data Construction
Build unified task files for HumanEval and MBPP first, then add TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite as external validation.

## Stage 2: Full-System Performance
Run A1-A7 architectures and record task score, cost, latency, and trace-level features.

## Stage 3: Contribution Attribution
Compute LOO for all tasks and sampled Shapley/Banzhaf for representative subsets. External full-system donor reuse is disabled; each attribution experiment computes its own grand coalition. Checkpoint IDs are executable- and treatment-aware, which invalidates older IDs. Myerson, Owen, and exact methods remain unwired and are rejected.

## Stage 4: Controlled Interventions
Change topology, role placement, and permissions while holding task and model fixed.

## Stage 5: Analysis
Compare contribution rankings, architecture sensitivity, and task-difficulty interactions. Exp08 generalization is analysis-only and requires completed non-empty upstream artifacts. Exp09 contribution prediction is a planned design and has no runner.
