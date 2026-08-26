"""Unified schemas for MASContributionBench.

The schema is intentionally broader than a normal code-generation benchmark
because the benchmark studies agent-level contribution attribution. The core
objects therefore cover tasks, configurations, runs, traces, evaluations, and
attribution scores.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetName(str, Enum):
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    TEAMBENCH = "teambench"
    MULTIAGENTBENCH = "multiagentbench"
    MARBLE = "marble"
    SWEBENCH_LITE = "swebench_lite"
    AIME_2026 = "aime_2026"
    GPQA_DIAMOND = "gpqa_diamond"
    HLE = "hle"
    ARC_AGI_2 = "arc_agi_2"
    IFBENCH = "ifbench"
    LIVECODEBENCH = "livecodebench"
    SWEBENCH_VERIFIED = "swebench_verified"


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    DATABASE = "database"
    BARGAINING = "bargaining"
    PLANNING = "planning"
    GENERAL_MAS = "general_mas"


class EvaluatorType(str, Enum):
    UNIT_TEST = "unit_test"
    OFFICIAL_GRADER = "official_grader"
    ENVIRONMENT_REWARD = "environment_reward"
    EXACT_MATCH = "exact_match"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"
    CUSTOM = "custom"


class FailureType(str, Enum):
    NONE = "none"
    WRONG_ANSWER = "wrong_answer"
    TEST_FAILURE = "test_failure"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    PROTOCOL_ERROR = "protocol_error"
    ROLE_VIOLATION = "role_violation"
    EMPTY_OUTPUT = "empty_output"
    INVALID_FORMAT = "invalid_format"
    EVALUATOR_ERROR = "evaluator_error"
    UNKNOWN = "unknown"


class RemovalProtocol(str, Enum):
    NONE = "none"
    HARD_REMOVAL = "hard_removal"
    BYPASS_REMOVAL = "bypass_removal"
    NULL_AGENT_REPLACEMENT = "null_agent_replacement"
    WEAK_MODEL_REPLACEMENT = "weak_model_replacement"


class AttributionMethod(str, Enum):
    LOO = "loo"
    SHAPLEY_EXACT = "shapley_exact"
    SHAPLEY_SAMPLED = "shapley_sampled"
    BANZHAF_EXACT = "banzhaf_exact"
    BANZHAF_SAMPLED = "banzhaf_sampled"
    MYERSON = "myerson"
    OWEN = "owen"


class UtilityType(str, Enum):
    TASK = "task"
    NET = "net"
    PROCESS = "process"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TraceEventType(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ROUTING = "routing"
    INTERMEDIATE_OUTPUT = "intermediate_output"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class SourceInfo(BaseModel):
    raw_dataset: str
    raw_task_id: str | None = None
    raw_file_path: str | None = None
    license: str | None = None
    conversion_version: str = "v1"
    download_manifest_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    evaluator_type: EvaluatorType
    metric: str
    timeout_seconds: float | None = None
    sandbox: str | None = None
    official_evaluator_version: str | None = None
    test_cases: list[dict[str, Any]] | None = None
    script_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MASMetadata(BaseModel):
    requires_planning: bool = False
    requires_coding: bool = False
    requires_verification: bool = False
    requires_research: bool = False
    requires_tool_use: bool = False
    estimated_agents: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskDifficulty(BaseModel):
    source: str | None = None
    level: Literal["unknown", "easy", "medium", "hard", "expert"] = "unknown"
    single_agent_score: float | None = None
    input_length: int | None = None
    constraint_count: int | None = None
    requires_cross_file_edit: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ContributionMetadata(BaseModel):
    eligible_roles: list[str] = Field(default_factory=list)
    default_architectures: list[str] = Field(default_factory=list)
    permission_requirements: list[str] = Field(default_factory=list)
    intervention_tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    task_id: str
    dataset: DatasetName | str
    split: str = "test"
    task_type: TaskType | str
    prompt: str
    context: str | None = None
    input_format: str | None = None
    output_format: str | None = None
    entry_point: str | None = None
    reference_solution: str | None = None
    tests: str | None = None
    evaluation: EvaluationConfig
    mas_metadata: MASMetadata = Field(default_factory=MASMetadata)
    difficulty: TaskDifficulty = Field(default_factory=TaskDifficulty)
    contribution_metadata: ContributionMetadata = Field(default_factory=ContributionMetadata)
    source: SourceInfo
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSpec(BaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    edge_type: str = "communication"
    weighted_edges: list[dict[str, Any]] = Field(default_factory=list)
    graph_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    model_id: str
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    agent_id: str
    role: str
    model: ModelConfig
    prompt_file: str | None = None
    prompt_hash: str | None = None
    permissions: dict[str, bool] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class ProtocolConfig(BaseModel):
    execution_order: str | list[str] | None = None
    routing_policy: str | None = None
    stopping_condition: str | None = None
    aggregation_policy: str | None = None
    retry_policy: str | None = None
    message_window_policy: str | None = None
    removal_protocol: RemovalProtocol = RemovalProtocol.NONE
    extra: dict[str, Any] = Field(default_factory=dict)


class ConfigRecord(BaseModel):
    config_id: str
    architecture_id: str
    design_graph: GraphSpec
    agents: list[AgentConfig]
    roles: dict[str, str] = Field(default_factory=dict)
    permissions: dict[str, dict[str, bool]] = Field(default_factory=dict)
    protocol: ProtocolConfig
    prompt_set: dict[str, str] = Field(default_factory=dict)
    task_filter: dict[str, Any] = Field(default_factory=dict)
    config_hash: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoalitionInfo(BaseModel):
    active_agents: list[str]
    removed_agents: list[str] = Field(default_factory=list)
    replacement_agents: dict[str, str] = Field(default_factory=dict)
    coalition_id: str | None = None


class RemovalInfo(BaseModel):
    protocol: RemovalProtocol = RemovalProtocol.NONE
    removed_agents: list[str] = Field(default_factory=list)
    bypass_edges: list[tuple[str, str]] = Field(default_factory=list)
    replacement_policy: str | None = None
    weak_model_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CostInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    latency_seconds: float | None = None
    estimated_cost_usd: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    experiment_id: str
    task_id: str
    dataset: DatasetName | str
    architecture_id: str
    seed: int
    coalition: CoalitionInfo
    removal: RemovalInfo = Field(default_factory=RemovalInfo)
    config_id: str | None = None
    config_hash: str | None = None
    prompt_hash: str | None = None
    git_commit: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    trace_path: str | None = None
    evaluation_path: str | None = None
    cost: CostInfo = Field(default_factory=CostInfo)
    failure_type: FailureType = FailureType.NONE
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    success: bool | None = None
    error: str | None = None
    latency_seconds: float | None = None


class IntermediateOutput(BaseModel):
    name: str
    content: Any
    artifact_type: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    run_id: str
    task_id: str
    event_id: str
    event_index: int
    event_type: TraceEventType
    sender: str | None = None
    receiver: str | None = None
    role: str | None = None
    timestamp: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    content: str | None = None
    tool_call: ToolCallRecord | None = None
    intermediate_output: IntermediateOutput | None = None
    design_graph_position: dict[str, Any] = Field(default_factory=dict)
    dependency_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceGraphRecord(BaseModel):
    run_id: str
    task_id: str
    graph_type: Literal["trace_graph", "dependency_graph"]
    graph: GraphSpec
    construction_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    run_id: str
    task_id: str
    dataset: DatasetName | str
    architecture_id: str
    final_answer: str | None = None
    score: float | None = None
    metric: str
    passed: bool | None = None
    failure_type: FailureType = FailureType.NONE
    evaluator: EvaluationConfig
    cost: CostInfo = Field(default_factory=CostInfo)
    raw_evaluator_output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConfidenceInterval(BaseModel):
    low: float
    high: float
    level: float = 0.95


class AttributionRecord(BaseModel):
    attribution_id: str
    experiment_id: str
    task_id: str
    dataset: DatasetName | str
    architecture_id: str
    agent_id: str
    role: str
    method: AttributionMethod
    utility_type: UtilityType = UtilityType.TASK
    score: float
    baseline_score: float | None = None
    coalition: CoalitionInfo
    removal_protocol: RemovalProtocol
    full_team_score: float | None = None
    ablated_score: float | None = None
    sample_id: str | None = None
    num_samples: int | None = None
    permutation_order: list[str] | None = None
    sampling_seed: int | None = None
    standard_error: float | None = None
    confidence_interval: ConfidenceInterval | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceFeatureRecord(BaseModel):
    run_id: str
    task_id: str
    architecture_id: str
    agent_id: str
    role: str
    messages_sent: int = 0
    messages_received: int = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    avg_message_length: float | None = None
    tool_call_count: int = 0
    tool_success_rate: float | None = None
    revision_count: int = 0
    quoted_by_downstream: int = 0
    role_violation_count: int = 0
    latency_seconds: float | None = None
    topology_features: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
