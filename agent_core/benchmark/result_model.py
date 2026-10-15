from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class BenchmarkPlanningSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    task_count: int = Field(
        ge=0,
    )

    investigation_tasks: int = Field(
        ge=0,
    )

    code_change_tasks: int = Field(
        ge=0,
    )

    test_tasks: int = Field(
        ge=0,
    )

    verification_tasks: int = Field(
        ge=0,
    )


class BenchmarkRetrievalSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "not_evaluated",
        "passed",
        "partial",
        "failed",
    ]

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    expected_files: list[str] = Field(
        default_factory=list,
    )

    retrieved_files: list[str] = Field(
        default_factory=list,
    )

    matched_files: list[str] = Field(
        default_factory=list,
    )

    missed_files: list[str] = Field(
        default_factory=list,
    )

    unexpected_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_hits: list[str] = Field(
        default_factory=list,
    )

    failure_reasons: list[str] = Field(
        default_factory=list,
    )


class BenchmarkCaseResult(BaseModel):
    """
    一次 Case Run 的压缩版成绩单。

    RunRecord 是完整运行证据。

    CaseResult 是面向 Benchmark 分析的摘要。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    run_id: str

    case_id: str

    base_commit_sha: str

    run_status: Literal[
        "running",
        "completed",
        "failed",
    ]

    final_stage: str

    planning: BenchmarkPlanningSummary | None = None

    retrieval: BenchmarkRetrievalSummary

    execution_error: str | None = None
