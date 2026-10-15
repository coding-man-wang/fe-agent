from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.benchmark.result_model import (
    BenchmarkCaseResult,
)


class BenchmarkSuiteCaseExecution(BaseModel):
    """
    Suite 中某一个 Case 的执行结果。

    注意：

    CaseResult 存在：
        表示 BenchmarkRunner 至少生成了
        一个 RunRecord。

    CaseResult 不存在：
        通常表示 Case 加载、Baseline Guard
        等 Harness 前置阶段直接抛异常。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str

    status: Literal[
        "completed",
        "error",
    ]

    result: BenchmarkCaseResult | None = None

    error: str | None = None


class BenchmarkSuiteSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    total_cases: int = Field(
        ge=0,
    )

    completed_cases: int = Field(
        ge=0,
    )

    harness_error_cases: int = Field(
        ge=0,
    )

    retrieval_passed: int = Field(
        ge=0,
    )

    retrieval_partial: int = Field(
        ge=0,
    )

    retrieval_failed: int = Field(
        ge=0,
    )

    retrieval_not_evaluated: int = Field(
        ge=0,
    )

    average_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    average_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class BenchmarkSuiteResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    suite_id: str

    started_at: datetime

    finished_at: datetime

    cases: list[BenchmarkSuiteCaseExecution] = Field(
        default_factory=list,
    )

    summary: BenchmarkSuiteSummary
