from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.impact import (
    ChangeImpactReport,
)


class ImpactVerificationPlan(BaseModel):
    """
    根据影响分析得到的验证建议。

    注意：

    当前阶段只能“增加验证”，
    不能删除原本要求的验证。
    """

    model_config = ConfigDict(extra="forbid")

    base_command_ids: list[str] = Field(default_factory=list)

    added_command_ids: list[str] = Field(default_factory=list)

    final_command_ids: list[str] = Field(default_factory=list)

    reasons: list[str] = Field(default_factory=list)


class ImpactAwareVerificationPlanner:
    """
    根据 Impact Analysis，
    对现有 Verification Plan 做增强。

    当前策略：

    1. 原有命令永远保留；
    2. 如果存在 unit test：
       确保 frontend-unit-test 存在；
    3. 如果存在 e2e test：
       增加 frontend-e2e；
    4. unknown impact：
       保守增加 frontend-e2e。

    当前绝不因为影响范围小而删除验证。
    """

    UNIT_COMMAND = "frontend-unit-test"

    E2E_COMMAND = "frontend-e2e"

    def build(
        self,
        *,
        base_command_ids: list[str],
        impact: ChangeImpactReport,
    ) -> ImpactVerificationPlan:

        final = self._dedupe(base_command_ids)

        added: list[str] = []

        reasons: list[str] = []

        suggested_tests = [path.lower() for path in impact.suggested_test_files]

        has_unit = any("/unit/" in path or ".test." in path for path in suggested_tests)

        has_e2e = any("/e2e/" in path for path in suggested_tests)

        if has_unit and self.UNIT_COMMAND not in final:
            final.append(self.UNIT_COMMAND)

            added.append(self.UNIT_COMMAND)

            reasons.append("影响分析发现相关 unit test，" "增加 frontend-unit-test")

        if has_e2e and self.E2E_COMMAND not in final:
            final.append(self.E2E_COMMAND)

            added.append(self.E2E_COMMAND)

            reasons.append("影响分析发现相关 E2E 测试，" "增加 frontend-e2e")

        if impact.impact_level == "unknown" and self.E2E_COMMAND not in final:
            final.append(self.E2E_COMMAND)

            added.append(self.E2E_COMMAND)

            reasons.append("影响范围无法完整确认，" "保守增加 frontend-e2e")

        return ImpactVerificationPlan(
            base_command_ids=(self._dedupe(base_command_ids)),
            added_command_ids=added,
            final_command_ids=final,
            reasons=reasons,
        )

    @staticmethod
    def _dedupe(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:
            normalized = value.strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            result.append(normalized)

        return result

    def build_for_test_files(
        self,
        *,
        base_command_ids: list[str],
        test_files: list[str],
    ) -> ImpactVerificationPlan:
        """
        Patch 最终 no-op 时，
        根据原计划涉及的测试文件，
        保留对应 Verification Intent。

        例如：

            tests/e2e/users.spec.ts

        即使最终没有产生 Patch，
        Final Verification 仍然应该跑 E2E。
        """

        final = self._dedupe(base_command_ids)

        added: list[str] = []
        reasons: list[str] = []

        normalized_files = [path.replace("\\", "/").lower() for path in test_files]

        has_unit = any(
            ("/unit/" in path or ".test." in path) and "/e2e/" not in path
            for path in normalized_files
        )

        has_e2e = any("/e2e/" in path for path in normalized_files)

        if has_unit and self.UNIT_COMMAND not in final:
            final.append(self.UNIT_COMMAND)

            added.append(self.UNIT_COMMAND)

            reasons.append("Late no-op 涉及 unit test，" "保留 frontend-unit-test")

        if has_e2e and self.E2E_COMMAND not in final:
            final.append(self.E2E_COMMAND)

            added.append(self.E2E_COMMAND)

            reasons.append("Late no-op 涉及 E2E test，" "保留 frontend-e2e")

        return ImpactVerificationPlan(
            base_command_ids=(self._dedupe(base_command_ids)),
            added_command_ids=(added),
            final_command_ids=(final),
            reasons=reasons,
        )
