from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

PolicySeverity = Literal[
    "must",
    "should",
]

PolicySource = Literal[
    "core_default",
    "project",
    "organization",
]


class EngineeringRule(BaseModel):
    """
    一条可被 Agent 消费的工程规则。

    这里表达的是：

        “开发时应该遵守什么”

    而不是 Harness 的安全不变量。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    rule_id: str = Field(
        min_length=1,
    )

    category: str = Field(
        min_length=1,
    )

    severity: PolicySeverity

    source: PolicySource

    description: str = Field(
        min_length=1,
    )


class EffectivePolicy(BaseModel):
    """
    当前 Task 最终生效的 Policy。

    Policy Resolver 负责把：

        默认规则
        +
        项目规则
        +
        未来组织规则

    合并成这一份结构化结果。

    Agent / PatchGenerator
    只消费 EffectivePolicy，
    不负责决定规则来源。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    rules: list[EngineeringRule] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_unique_rule_ids(
        self,
    ) -> "EffectivePolicy":

        rule_ids = [rule.rule_id for rule in self.rules]

        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("EffectivePolicy " "存在重复 rule_id")

        return self

    def rule_ids(
        self,
    ) -> list[str]:

        return [rule.rule_id for rule in self.rules]
