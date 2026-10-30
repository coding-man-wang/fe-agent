from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class ExecutionObligationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool

    expected_obligation_ids: list[str] = Field(default_factory=list)

    satisfied_obligation_ids: list[str] = Field(default_factory=list)

    unresolved_obligation_ids: list[str] = Field(default_factory=list)

    reasons: list[str] = Field(default_factory=list)


class ExecutionObligationGate:
    """
    Compression 后的 Execution Unit
    在真正进入 Patch 之前必须通过。

    Gate 不负责判断代码语义。

    Agent 已经完成分析；

    Gate 只检查：

        每个由 Compression 保留下来的
        Logical Obligation

    是否都被模型明确分类为：

        1. 当前代码已经满足

        或

        2. 当前 Proposal 会负责完成

    unresolved 永远不能进入 Patch。
    """

    def evaluate(
        self,
        *,
        task: TaskDefinition,
        analysis: ReadOnlyCodingResult,
    ) -> ExecutionObligationDecision:

        obligations = list(
            getattr(
                task,
                "execution_obligations",
                [],
            )
        )

        # 非 Compression Task
        # 不改变原有行为。
        if not obligations:
            return ExecutionObligationDecision(
                allowed=True,
            )

        expected_ids = [item.obligation_id for item in obligations]

        coverage_by_id = {
            item.obligation_id: item for item in analysis.obligation_coverage
        }

        satisfied: list[str] = []

        unresolved: list[str] = []

        reasons: list[str] = []

        for obligation_id in expected_ids:

            coverage = coverage_by_id.get(obligation_id)

            if coverage is None:

                unresolved.append(obligation_id)

                reasons.append(f"{obligation_id}: " "missing obligation coverage")

                continue

            if coverage.status == "unresolved":

                unresolved.append(obligation_id)

                reasons.append(f"{obligation_id}: " + coverage.reason)

                continue

            satisfied.append(obligation_id)

        return ExecutionObligationDecision(
            allowed=(not unresolved and len(satisfied) == len(expected_ids)),
            expected_obligation_ids=(expected_ids),
            satisfied_obligation_ids=(satisfied),
            unresolved_obligation_ids=(unresolved),
            reasons=reasons,
        )
