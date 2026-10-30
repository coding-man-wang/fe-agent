from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.codebase.impact import (
    ChangeImpactReport,
)
from agent_core.benchmark.impact_verification import (
    ImpactVerificationPlan,
)
from agent_core.benchmark.execution_plan import (
    ExecutionPlanCompressionSummary,
)

TaskExecutionStatus = Literal[
    "skipped_no_change",
    "skipped_no_production_change",
    "skipped_runtime_noop",
    "skipped_non_code_task",
    "skipped_redundant_verification",
    "completed",
    "failed",
]


class VerificationExecutionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)

    kind: Literal[
        "initial",
        "repair",
    ]

    proposal_id: str

    workspace_id: str

    status: Literal[
        "succeeded",
        "failed",
    ]

    command_ids: list[str] = Field(default_factory=list)

    checks: list[VerificationCheck] = Field(default_factory=list)

    error: str | None = None


class FullExecutionTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    task_type: str

    coalesced_task_ids: list[str] = Field(
        default_factory=list,
        description=("当前 Runtime Execution Task " "同时覆盖的其他 Logical Task IDs"),
    )

    status: TaskExecutionStatus

    execution_disposition: str | None = None

    base_commit_sha: str | None = None

    context_revision: str | None = None

    context_retrieval_mode: (
        Literal[
            "hybrid",
            "lexical_only",
            "lexical_fallback",
        ]
        | None
    ) = None

    semantic_fallback_reason: str | None = None

    retrieval_files: list[str] = Field(default_factory=list)

    selected_context_files: list[str] = Field(default_factory=list)

    graph_added_files: list[str] = Field(default_factory=list)

    applied_policy_rule_ids: list[str] = Field(default_factory=list)

    workspace_id: str | None = None

    proposal_id: str | None = None

    proposed_files: list[str] = Field(default_factory=list)

    planned_impact: ChangeImpactReport | None = None

    planned_verification: ImpactVerificationPlan | None = None

    scheduled_verification_command_ids: list[str] = Field(default_factory=list)

    executed_verification_command_ids: list[str] = Field(default_factory=list)

    verification_status: str | None = None

    verification_checks: list[VerificationCheck] = Field(default_factory=list)

    verification_repair_count: int = Field(
        default=0,
        ge=0,
    )

    verification_attempts: list[VerificationExecutionAttempt] = Field(
        default_factory=list
    )

    verification_repair_correction_count: int = Field(
        default=0,
        ge=0,
    )

    completion_status: str | None = None

    commit_sha: str | None = None

    error: str | None = None

    expected_execution_obligation_ids: list[str] = Field(default_factory=list)

    satisfied_execution_obligation_ids: list[str] = Field(default_factory=list)

    unresolved_execution_obligation_ids: list[str] = Field(default_factory=list)


class FullExecutionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed"]
    expected_change_files: list[str] = Field(default_factory=list)
    allowed_change_files: list[str] = Field(default_factory=list)
    actual_change_files: list[str] = Field(default_factory=list)
    matched_change_files: list[str] = Field(default_factory=list)
    missed_change_files: list[str] = Field(default_factory=list)
    unexpected_change_files: list[str] = Field(default_factory=list)
    forbidden_change_hits: list[str] = Field(default_factory=list)
    completed_task_count: int = 0
    failed_task_ids: list[str] = Field(default_factory=list)
    verification_passed: bool = False
    hidden_acceptance_passed: bool = False
    failure_reasons: list[str] = Field(default_factory=list)
    # =====================================
    # v1.2.4 Impact Analysis Evaluation
    # =====================================

    expected_impact_files: list[str] = Field(default_factory=list)

    observed_impact_files: list[str] = Field(default_factory=list)

    missed_impact_files: list[str] = Field(default_factory=list)

    required_impact_verification_command_ids: list[str] = Field(default_factory=list)

    observed_verification_command_ids: list[str] = Field(default_factory=list)

    missing_impact_verification_command_ids: list[str] = Field(default_factory=list)

    impact_coverage_passed: bool = True

    impact_verification_passed: bool = True


class FullExecutionRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    baseline_run_id: str
    case_id: str
    base_commit_sha: str
    execution_plan_compression: ExecutionPlanCompressionSummary | None = None
    final_commit_sha: str | None = None
    final_workspace_root: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    tasks: list[FullExecutionTaskRecord] = Field(default_factory=list)
    hidden_acceptance: PatchVerificationResult | None = None
    evaluation: FullExecutionEvaluation | None = None
    error: str | None = None
    # ==================================
    # v1.3D Verification Strategy
    # ==================================

    verification_strategy: dict[str, Any] | None = None

    test_strategy_proposal_correction_count: int = Field(
        default=0,
        ge=0,
    )
