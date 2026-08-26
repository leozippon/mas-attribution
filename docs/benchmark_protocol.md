# Benchmark Protocol

1. Convert raw datasets into unified JSONL tasks.
2. Run single-agent baseline for task difficulty.
3. Run full MAS architectures for performance and cost. HumanEval/MBPP pass@1 requires `--execute-code`; formal runs should use Docker.
4. Estimate agent contribution with LOO, Shapley, and Banzhaf.
5. Run interventions over topology, role, permission, and information access.
6. Aggregate scores, trace features, and contribution rankings. Exp08 generalization requires completed non-empty upstream artifacts.
7. Export paper tables and figures into `outputs/`.

Exp09 contribution prediction is a planned design and intentionally has no runner.
