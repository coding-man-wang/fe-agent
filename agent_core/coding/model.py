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


CodeChangeAction = Literal[
    "modify",
    "create",
]


class ProposedCodeChange(BaseModel):

    model_config = ConfigDict(extra="forbid")

    relative_path: str

    action: CodeChangeAction = "modify"

    objective: str = Field(min_length=1)

    proposed_approach: str = Field(min_length=1)


EvidenceGapSeverity = Literal[
    "blocking",
    "advisory",
]

EvidenceGapResolutionStage = Literal[
    "analysis",
    "verification",
]


class EvidenceGap(BaseModel):
    """
    Coding Analysis 阶段发现的证据缺口。

    severity:
        blocking:
            缺失信息阻止当前阶段做出可靠判断。

        advisory:
            信息有帮助，但不阻止当前阶段继续。

    resolution_stage:
        analysis:
            必须通过更多代码、接口、业务规则等
            分析证据解决。

        verification:
            需要通过测试、运行时日志、请求记录、
            build 等后续执行结果解决。

    示例：

    analysis:
        - implementation target 内容未提供；
        - API contract 不清楚；
        - 业务约束缺失；
        - 无法判断应该如何安全修改。

    verification:
        - 尚未运行 unit test；
        - 尚未 build；
        - 缺少 runtime log；
        - 缺少网络请求记录；
        - 尚未证明不会重复请求。
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)

    severity: EvidenceGapSeverity

    resolution_stage: EvidenceGapResolutionStage = "analysis"

    related_paths: list[str] = Field(
        default_factory=list,
    )

    reason: str = Field(min_length=1)


ExecutionObligationCoverageStatus = Literal[
    "satisfied_by_current_code",
    "addressed_by_proposed_change",
    "unresolved",
]


class ExecutionObligationCoverage(BaseModel):
    """
    Coding Analysis 对一项
    Execution Obligation 的明确结论。

    satisfied_by_current_code:
        最新代码已经满足，
        不需要修改。

    addressed_by_proposed_change:
        当前 proposed_changes
        会负责完成。

    unresolved:
        当前证据不足，
        Harness 不允许进入 Patch。
    """

    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1)

    status: ExecutionObligationCoverageStatus

    related_paths: list[str] = Field(default_factory=list)

    reason: str = Field(min_length=1)


class ExecutionObligationReviewResult(BaseModel):
    """
    Bounded Execution Obligation Review
    的唯一允许输出。

    Review 不允许重新生成：

    - findings
    - proposed_changes
    - evidence_gaps
    - verification_plan

    只重新判断：
        obligation_coverage
    """

    model_config = ConfigDict(extra="forbid")

    obligation_coverage: list[ExecutionObligationCoverage] = Field(default_factory=list)


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

    obligation_coverage: list[ExecutionObligationCoverage] = Field(
        default_factory=list,
    )


CodeChangeAction = Literal[
    "modify",
    "create",
]


class ProposedCodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str

    action: CodeChangeAction = "modify"

    objective: str = Field(min_length=1)

    proposed_approach: str = Field(min_length=1)
