from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CodeFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str

    evidence: str = Field(
        min_length=1,
        description="从提供的代码上下文中引用的原文片段",
    )

    explanation: str = Field(
        min_length=1,
        description="基于证据的分析结论",
    )


class ProposedCodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str

    objective: str = Field(min_length=1)

    proposed_approach: str = Field(min_length=1)


EvidenceGapSeverity = Literal[
    "blocking",
    "advisory",
]


class EvidenceGap(BaseModel):
    """
    Coding Analysis 阶段发现的证据缺口。

    blocking:
        当前缺失信息会阻止 Agent 判断
        是否需要修改 / 如何安全修改。

    advisory:
        信息有帮助，但不影响当前修改决策。
        例如：
        - 测试尚未执行
        - 参考组件未展开
        - 测试覆盖情况未知
        - 缺少运行时日志
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)

    severity: EvidenceGapSeverity

    related_paths: list[str] = Field(
        default_factory=list,
    )

    reason: str = Field(min_length=1)


class ReadOnlyCodingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    summary: str = Field(min_length=1)

    findings: list[CodeFinding] = Field(
        default_factory=list,
    )

    proposed_changes: list[ProposedCodeChange] = Field(
        default_factory=list,
    )

    verification_plan: list[str] = Field(
        default_factory=list,
    )

    # Legacy compatibility.
    # 新的 Coding Agent 应优先使用 evidence_gaps。
    missing_information: list[str] = Field(
        default_factory=list,
    )

    evidence_gaps: list[EvidenceGap] = Field(
        default_factory=list,
    )

    execution_status: Literal["not_executed"] = "not_executed"
