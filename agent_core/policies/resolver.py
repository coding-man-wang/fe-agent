from __future__ import annotations

import hashlib

from agent_core.policies.model import (
    EffectivePolicy,
    EngineeringRule,
)
from agent_core.policies.test_authoring import (
    DEFAULT_TEST_AUTHORING_RULES,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class EngineeringPolicyResolver:
    """
    Policy Resolution。

    中文：
        工程规则解析 / 合并。

    作用：
        根据当前 Task，
        把不同来源的工程规范
        合并成 Agent 真正消费的
        EffectivePolicy。

    当前 v1：

        Core Default
        +
        Project engineering_rules

    后续可以增加：

        Organization Policy
        Repository .fe-agent/policies
        Team Policy

    但 Agent 本身无需修改。
    """

    def resolve(
        self,
        *,
        task: TaskDefinition,
        project_rules: list[str],
    ) -> EffectivePolicy:

        rules: list[EngineeringRule] = []

        # ==================================
        # Capability-specific Policy
        # ==================================

        if task.task_type == "test":

            rules.extend(DEFAULT_TEST_AUTHORING_RULES)

        # ==================================
        # Project Engineering Rules
        #
        # 当前 ProjectContext
        # 已经使用 list[str]。
        #
        # 暂时在 Resolver Boundary
        # 转成结构化 Rule，
        # 不要求现在重构整个
        # ProjectContext Schema。
        # ==================================

        for raw_rule in project_rules:

            description = raw_rule.strip()

            if not description:
                continue

            rule_id = self._project_rule_id(description)

            rules.append(
                EngineeringRule(
                    rule_id=rule_id,
                    category=("project_engineering"),
                    severity="must",
                    source="project",
                    description=(description),
                )
            )

        return EffectivePolicy(rules=self._deduplicate(rules))

    @staticmethod
    def _project_rule_id(
        description: str,
    ) -> str:
        """
        不使用 list index，
        避免仅因为规则排序变化
        就改变 rule identity。
        """

        digest = hashlib.sha256(description.encode("utf-8")).hexdigest()[:12]

        return "PROJECT-" + digest.upper()

    @staticmethod
    def _deduplicate(
        rules: list[EngineeringRule],
    ) -> list[EngineeringRule]:

        result: list[EngineeringRule] = []

        seen: set[str] = set()

        for rule in rules:

            if rule.rule_id in seen:
                continue

            seen.add(rule.rule_id)

            result.append(rule)

        return result
