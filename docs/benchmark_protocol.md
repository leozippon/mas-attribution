# Benchmark Protocol

1. Download and normalize the seven canonical datasets with
   `scripts/prepare_external_benchmarks.py`. Resume is default;
   `--force` re-downloads and re-converts; fail-fast is default;
   `--max-files` and `--max-records` bound smoke conversion.
2. Select the fixed seeded subset: `max_tasks: 30`,
   `task_selection: seeded_random`, `sampling.selection_seed: 0`.
   Run seeds do not change task IDs.
3. Send only the public task view to models. Tests, gold answers, ARC
   test outputs, and other private fields stay out of prompts.
4. Call the externally managed local Qwen endpoint at
   `http://127.0.0.1:8000/v1` (`qwen-local`). Do not launch vLLM from
   this repository. Cache keys and run IDs are seed- and
   treatment-aware. Model request failures abort by default. Convert a
   failed request into a scored failure record only with explicit
   `MAS_LLM_ERROR_FALLBACK=1`.
5. Score AIME 2026, GPQA Diamond, HLE, ARC-AGI-2, and IFBench in-loop.
   LiveCodeBench and SWE-bench Verified stay `score=None` / `passed=None`
   until official results are joined. They are not attribution-eligible
   in-loop.
6. Run Exp01 (full MAS) and Exp02 (baselines) on all seven datasets.
   Run Exp03–Exp07 only on the five locally scored datasets.
7. Exp03 is LOO with null-agent replacement only. Exp04 is modest
   sampled Shapley and sampled Banzhaf. Exp05 is single-pass
   chain/star/DAG with fixed roles and permissions. Exp06 is a
   Planner↔Coder functional/prompt swap with fixed topology and
   position permissions. Exp07 toggles only `write_solution` and
   `final_answer` under strict enforcement.
8. Exp08 is descriptive analysis only. It derives selected-task
   metadata from configured task files, does not use score files as
   covariates, and makes no causal or inferential claim. Exp09 is an
   unimplemented design and is CLI-refused.
9. `scripts/run_suite.py` runs at most two non-Exp08 configs with at
   most two subprocesses and rejects overlapping outputs.

HumanEval and MBPP are not canonical. DeepSeek is not a canonical
model path. A5/A6/A7 are single-pass design graphs, not multi-round
debate/star/graph engines.
