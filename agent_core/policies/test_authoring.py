from agent_core.policies.model import (
    EngineeringRule,
)

DEFAULT_TEST_AUTHORING_RULES = [
    EngineeringRule(
        rule_id=("TEST-BEHAVIOR-001"),
        category="test_authoring",
        severity="must",
        source="core_default",
        description=(
            "测试应优先验证公开的 "
            "Behavioral Contract。"
            "除非 Requirement、公开 API Contract "
            "或项目规范明确要求具体 delegation behavior，"
            "否则不得把内部 helper 调用方式、"
            "私有函数参数或内部调用次数"
            "当成业务验收契约。"
        ),
    ),
    EngineeringRule(
        rule_id=("TEST-RUNTIME-PATH-001"),
        category="test_authoring",
        severity="must",
        source="core_default",
        description=(
            "当 Remote、Mock、Cache、Network "
            "等多个 Runtime Path "
            "需要保持相同业务语义时，"
            "应验证 Behavioral Equivalence，"
            "而不是强制这些路径使用"
            "完全相同的内部实现形式。"
        ),
    ),
    EngineeringRule(
        rule_id=("TEST-ENV-001"),
        category="test_authoring",
        severity="must",
        source="core_default",
        description=(
            "不得假设 module-level "
            "environment state "
            "可以在模块加载完成后动态切换。"
            "对 import.meta.env、process.env "
            "或模块初始化常量进行多模式测试时，"
            "必须使用当前测试框架支持的"
            "模块隔离、reset module、"
            "重新 import 或项目已有测试模式。"
        ),
    ),
    EngineeringRule(
        rule_id=("TEST-FILTER-STATE-001"),
        category="test_authoring",
        severity="must",
        source="core_default",
        description=(
            "必须区分 Filter State "
            "与 Domain State。"
            "例如 all 可能仅表示"
            "不应用过滤条件，"
            "并不一定是合法的 Domain Status。"
            "测试不得无依据强制内部层"
            "传递 all 字面量。"
        ),
    ),
    EngineeringRule(
        rule_id=("TEST-PATTERN-001"),
        category="test_authoring",
        severity="should",
        source="core_default",
        description=(
            "新增测试时应优先沿用"
            "当前 Repository 已有的测试框架、"
            "Mock 方式、fixture、locator "
            "和测试文件组织模式。"
        ),
    ),
]
