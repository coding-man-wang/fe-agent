from __future__ import annotations

from agent_core.benchmark.execution_model import (
    FullExecutionTaskRecord,
)
from agent_core.benchmark.impact_verification import (
    ImpactVerificationPlan,
)


class FinalVerificationPlanner:
    """
    构建 Run-level Final Verification Plan。

    Final Verification 包含：

        Case 基础 Verification
        +
        前面各 Task 的 Impact-driven additions

    例如：

        CASE-006 base:
            lint
            typecheck
            unit
            build

        TASK-003 Impact:
            + e2e

        Final:
            lint
            typecheck
            unit
            build
            e2e

    Final Verification 永远只增加，
    不根据 Impact 删除基础检查。
    """

    def build(
        self,
        *,
        base_command_ids: list[str],
        task_records: list[FullExecutionTaskRecord],
    ) -> ImpactVerificationPlan:

        base = self._dedupe(base_command_ids)

        base_set = set(base)

        added: list[str] = []

        added_set: set[str] = set()

        for record in task_records:

            planned = record.planned_verification

            if planned is None:
                continue

            for command_id in planned.added_command_ids:

                if command_id in base_set:
                    continue

                if command_id in added_set:
                    continue

                added_set.add(command_id)

                added.append(command_id)

        final = [
            *base,
            *added,
        ]

        reasons: list[str] = []

        if added:
            reasons.append(
                "Final Verification "
                "合并前序 Task 的 "
                "Impact-driven verification: " + ", ".join(added)
            )

        return ImpactVerificationPlan(
            base_command_ids=base,
            added_command_ids=added,
            final_command_ids=final,
            reasons=reasons,
        )

    @staticmethod
    def _dedupe(
        values: list[str],
    ) -> list[str]:

        seen: set[str] = set()

        result: list[str] = []

        for value in values:

            normalized = value.strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(normalized)

        return result
