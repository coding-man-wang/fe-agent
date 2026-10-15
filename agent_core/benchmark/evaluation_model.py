from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

RetrievalEvaluationStatus = Literal[
    "passed",
    "partial",
    "failed",
]


class BenchmarkRetrievalEvaluation(BaseModel):
    """
    Benchmark 对 Retrieval Capture
    自动评分后的结果。

    注意：

    Capture 是“Agent 实际找到了什么”。

    Evaluation 是：
    “这些结果和标准答案相比怎么样”。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: RetrievalEvaluationStatus

    scored_task_types: list[str] = Field(
        default_factory=list,
    )

    expected_files: list[str] = Field(
        default_factory=list,
    )

    retrieved_files: list[str] = Field(
        default_factory=list,
    )

    matched_expected_files: list[str] = Field(
        default_factory=list,
    )

    missed_expected_files: list[str] = Field(
        default_factory=list,
    )

    unexpected_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_hits: list[str] = Field(
        default_factory=list,
    )

    recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    failure_reasons: list[str] = Field(
        default_factory=list,
    )
