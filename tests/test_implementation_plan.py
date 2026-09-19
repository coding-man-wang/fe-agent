import pytest

from agent_core.llm.model import ChatResponse
from agent_core.sdd.model import (
    ImplementationPlan,
    ProjectContext,
    RequirementSpec,
    TechnicalSpec,
)
from agent_core.sdd.plan import (
    ImplementationPlanGenerator,
)


class FakePlanLLM:

    async def think(
            self,
            request,
    ):
        return ChatResponse(
            content="""
            {
                "summary":
                    "按照接口、页面、交互和测试的顺序实现用户列表功能",

                "preconditions": [
                    "确认用户分页 API 的请求参数和响应结构"
                ],

                "steps": [
                    {
                        "order": 1,
                        "title": "实现用户 API 封装",
                        "goal": "建立用户列表数据访问能力",

                        "actions": [
                            "增加用户分页查询 API 封装"
                        ],

                        "affected_modules": [
                            "src/api"
                        ],

                        "verification": [
                            "确认 API 请求参数符合接口契约"
                        ]
                    },

                    {
                        "order": 2,
                        "title": "实现用户列表页面",
                        "goal": "完成用户列表展示",

                        "actions": [
                            "新增用户列表页面",
                            "复用 Table 和 Pagination",
                            "接入用户分页 API"
                        ],

                        "affected_modules": [
                            "src/views",
                            "src/router"
                        ],

                        "verification": [
                            "页面能够正确展示用户列表"
                        ]
                    },

                    {
                        "order": 3,
                        "title": "实现查询交互",
                        "goal": "完成搜索、筛选和分页",

                        "actions": [
                            "实现用户名搜索",
                            "实现状态筛选",
                            "实现分页切换"
                        ],

                        "affected_modules": [
                            "src/views"
                        ],

                        "verification": [
                            "搜索结果正确",
                            "状态筛选结果正确",
                            "分页结果正确"
                        ]
                    },

                    {
                        "order": 4,
                        "title": "增加自动化测试",
                        "goal": "验证用户列表主要功能",

                        "actions": [
                            "增加 Vitest 测试",
                            "增加 Playwright E2E 测试"
                        ],

                        "affected_modules": [
                            "tests"
                        ],

                        "verification": [
                            "Unit Test 全部通过",
                            "E2E Test 全部通过"
                        ]
                    }
                ],

                "completion_criteria": [
                    "Requirement Acceptance Criteria 全部满足",
                    "lint、typecheck 和自动化测试全部通过"
                ],

                "risks": [
                    "用户 API 实际契约可能与当前假设不一致"
                ],

                "open_questions": []
            }
            """,
            tool_calls=[],
            model="fake-model",
        )


def create_requirement() -> RequirementSpec:
    return RequirementSpec(
        title="用户列表功能",

        summary=(
            "在 Vue3 管理后台增加用户列表页面"
        ),

        goals=[
            "展示用户列表",
            "支持用户名搜索",
            "支持状态筛选",
            "支持分页",
        ],

        non_goals=[
            "不实现用户新增",
            "不实现用户删除",
        ],

        acceptance_criteria=[
            "页面能够展示用户列表",
            "用户名搜索结果正确",
            "状态筛选结果正确",
            "分页切换结果正确",
        ],

        constraints=[
            "使用 Vue3",
            "必须复用内部 Table",
        ],

        assumptions=[
            "后端已经提供用户分页查询接口",
        ],

        open_questions=[],
    )


def create_project_context() -> ProjectContext:
    return ProjectContext(
        project_name="admin-web",
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
        state_management="Pinia",

        test_stack=[
            "Vitest",
            "Playwright",
        ],

        available_components=[
            "Table",
            "SearchForm",
            "Pagination",
        ],

        engineering_rules=[
            "使用 Composition API",
            "API 请求统一放在 src/api",
            "页面禁止直接调用 axios",
        ],

        directory_summary=[
            "src/views: 页面目录",
            "src/api: API 请求目录",
            "src/components: 公共组件",
        ],

        relevant_files=[
            "src/router/index.ts",
            "src/api/user.ts",
        ],
    )


def create_technical_spec() -> TechnicalSpec:
    return TechnicalSpec(
        summary=(
            "基于现有 Vue3 技术栈实现用户列表页面"
        ),

        solution_approach=[
            "新增用户列表页面",
            "增加用户分页 API 封装",
            "使用页面局部状态维护查询条件",
            "复用 Table、SearchForm 和 Pagination",
        ],

        affected_modules=[
            "用户管理页面",
            "API 层",
            "路由",
        ],

        reuse_plan=[
            "复用 Table",
            "复用 SearchForm",
            "复用 Pagination",
        ],

        data_flow=[
            "用户修改查询条件",
            "页面组合请求参数",
            "调用用户分页 API",
            "API 返回分页数据",
            "页面更新 Table",
        ],

        technical_decisions=[
            {
                "topic": "状态管理",
                "decision": "使用页面局部状态",
                "rationale": (
                    "查询条件不需要跨页面共享"
                ),
            }
        ],

        testing_strategy=[
            "使用 Vitest 测试查询逻辑",
            "使用 Playwright 验证搜索、筛选和分页",
        ],

        risks=[
            "用户 API 实际契约尚未完全确认"
        ],

        assumptions=[
            "后端存在分页查询 API"
        ],

        open_questions=[],
    )


def test_implementation_plan_model():
    plan = ImplementationPlan(
        summary="实现用户列表",

        preconditions=[
            "确认 API 契约"
        ],

        steps=[
            {
                "order": 1,
                "title": "实现 API",
                "goal": "提供用户数据访问能力",
                "actions": [
                    "增加用户查询接口"
                ],
                "affected_modules": [
                    "src/api"
                ],
                "verification": [
                    "确认 API 可正常调用"
                ],
            }
        ],

        completion_criteria=[
            "功能验证通过"
        ],

        risks=[],

        open_questions=[],
    )

    assert (
            plan.summary
            == "实现用户列表"
    )

    assert len(plan.steps) == 1

    assert plan.steps[0].order == 1


@pytest.mark.asyncio
async def test_generate_implementation_plan():
    requirement = create_requirement()

    project_context = (
        create_project_context()
    )

    technical_spec = (
        create_technical_spec()
    )

    generator = (
        ImplementationPlanGenerator(
            llm=FakePlanLLM()
        )
    )

    plan = await generator.generate(
        requirement=requirement,
        technical_spec=technical_spec,
        project_context=project_context,
    )

    assert isinstance(
        plan,
        ImplementationPlan,
    )

    assert len(plan.steps) == 4

    assert (
            plan.steps[0].title
            == "实现用户 API 封装"
    )

    assert (
            plan.steps[0].order
            == 1
    )

    assert (
            "确认用户分页 API 的请求参数和响应结构"
            in plan.preconditions
    )

    assert (
            "Requirement Acceptance Criteria 全部满足"
            in plan.completion_criteria
    )
