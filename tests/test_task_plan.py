import pytest

from agent_core.llm.model import (
    ChatResponse,
)
from agent_core.sdd.model import RequirementSpec, ProjectContext, TechnicalSpec, ImplementationPlan, TaskPlan
from agent_core.sdd.task import TaskPlanner
from agent_core.sdd.task_validation import validate_task_plan, has_dependency_cycle


class FakeTaskLLM:

    async def chat(
            self,
            request,
    ):
        return ChatResponse(
            content="""
            {
              "summary":
                "将用户列表实施计划拆分为可执行任务",

              "tasks": [
                {
                  "task_id": "TASK-001",
                  "task_type": "investigation",
                  "title": "确认用户分页 API 契约",
                  "objective":
                    "确认用户列表 API 的请求参数和响应结构",

                  "source_step_order": 1,

                  "depends_on": [],

                  "inputs": [
                    "TechnicalSpec"
                  ],

                  "target_modules": [
                    "用户 API"
                  ],

                  "expected_outputs": [
                    "明确的用户分页 API 契约"
                  ],

                  "verification": [
                    "确认分页参数和响应字段定义"
                  ],

                  "acceptance_criteria_refs": []
                },

                {
                  "task_id": "TASK-002",
                  "task_type": "code_change",
                  "title": "实现用户 API 封装",
                  "objective":
                    "提供用户分页数据访问能力",

                  "source_step_order": 1,

                  "depends_on": [
                    "TASK-001"
                  ],

                  "inputs": [
                    "用户分页 API 契约"
                  ],

                  "target_modules": [
                    "src/api"
                  ],

                  "expected_outputs": [
                    "用户分页查询 API 封装"
                  ],

                  "verification": [
                    "分页查询请求参数符合 API 契约"
                  ],

                  "acceptance_criteria_refs": []
                },

                {
                  "task_id": "TASK-003",
                  "task_type": "code_change",
                  "title": "实现用户列表页面",
                  "objective":
                    "完成用户列表展示、搜索、筛选和分页",

                  "source_step_order": 2,

                  "depends_on": [
                    "TASK-002"
                  ],

                  "inputs": [
                    "用户分页 API 封装",
                    "TechnicalSpec"
                  ],

                  "target_modules": [
                    "src/views",
                    "src/router"
                  ],

                  "expected_outputs": [
                    "用户列表页面",
                    "搜索功能",
                    "状态筛选功能",
                    "分页功能"
                  ],

                  "verification": [
                    "页面能够展示用户",
                    "搜索结果正确",
                    "状态筛选结果正确",
                    "分页切换结果正确"
                  ],

                  "acceptance_criteria_refs": [
                    "页面能够展示用户列表",
                    "用户名搜索结果正确",
                    "状态筛选结果正确",
                    "分页切换结果正确"
                  ]
                },

                {
                  "task_id": "TASK-004",
                  "task_type": "test",
                  "title": "增加自动化测试",
                  "objective":
                    "覆盖用户列表主要行为",

                  "source_step_order": 4,

                  "depends_on": [
                    "TASK-003"
                  ],

                  "inputs": [
                    "用户列表实现",
                    "Testing Strategy"
                  ],

                  "target_modules": [
                    "tests"
                  ],

                  "expected_outputs": [
                    "Vitest 测试",
                    "Playwright E2E 测试"
                  ],

                  "verification": [
                    "Unit Test 全部通过",
                    "E2E Test 全部通过"
                  ],

                  "acceptance_criteria_refs": [
                    "用户名搜索结果正确",
                    "状态筛选结果正确",
                    "分页切换结果正确"
                  ]
                },

                {
                  "task_id": "TASK-005",
                  "task_type": "verification",
                  "title": "执行最终验证",
                  "objective":
                    "确认本次需求满足交付条件",

                  "source_step_order": 4,

                  "depends_on": [
                    "TASK-004"
                  ],

                  "inputs": [
                    "最终代码",
                    "自动化测试"
                  ],

                  "target_modules": [
                    "整个用户列表功能"
                  ],

                  "expected_outputs": [
                    "最终验证结果"
                  ],

                  "verification": [
                    "lint 通过",
                    "typecheck 通过",
                    "Unit Test 通过",
                    "E2E Test 通过"
                  ],

                  "acceptance_criteria_refs": [
                    "页面能够展示用户列表",
                    "用户名搜索结果正确",
                    "状态筛选结果正确",
                    "分页切换结果正确"
                  ]
                }
              ],

              "completion_criteria": [
                "所有 Task 均完成验证",
                "Requirement Acceptance Criteria 全部满足"
              ]
            }
            """,
            tool_calls=[],
            model="fake-model",
        )


def create_requirement() -> RequirementSpec:
    return RequirementSpec(
        title="用户列表功能",

        summary="实现用户列表页面",

        goals=[
            "展示用户列表",
            "支持搜索",
            "支持状态筛选",
            "支持分页",
        ],

        non_goals=[
            "不实现用户删除",
        ],

        acceptance_criteria=[
            "页面能够展示用户列表",
            "用户名搜索结果正确",
            "状态筛选结果正确",
            "分页切换结果正确",
        ],

        constraints=[
            "使用 Vue3"
        ],

        assumptions=[],

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
            "API 请求统一放在 src/api"
        ],

        directory_summary=[
            "src/api",
            "src/views",
            "src/router",
        ],

        relevant_files=[],
    )


def create_technical_spec() -> TechnicalSpec:
    return TechnicalSpec(
        summary="实现用户列表",

        solution_approach=[
            "增加 API 封装",
            "增加用户列表页面",
            "增加搜索、筛选和分页",
        ],

        affected_modules=[
            "API",
            "用户管理页面",
        ],

        reuse_plan=[
            "Table",
            "SearchForm",
            "Pagination",
        ],

        data_flow=[
            "查询条件 -> API -> Table"
        ],

        technical_decisions=[],

        testing_strategy=[
            "Vitest",
            "Playwright",
        ],

        risks=[],

        assumptions=[],

        open_questions=[],
    )


def create_implementation_plan() -> ImplementationPlan:
    return ImplementationPlan(
        summary="实现用户列表",

        preconditions=[
            "确认用户 API 契约"
        ],

        steps=[
            {
                "order": 1,
                "title": "实现 API 能力",
                "goal": "提供用户数据访问能力",
                "actions": [
                    "确认 API",
                    "增加 API 封装",
                ],
                "affected_modules": [
                    "src/api"
                ],
                "verification": [
                    "API 可正常使用"
                ],
            },
            {
                "order": 2,
                "title": "实现页面",
                "goal": "展示用户列表",
                "actions": [
                    "新增用户列表页面"
                ],
                "affected_modules": [
                    "src/views",
                    "src/router",
                ],
                "verification": [
                    "页面正常展示"
                ],
            },
            {
                "order": 3,
                "title": "实现查询交互",
                "goal": "实现搜索筛选分页",
                "actions": [
                    "搜索",
                    "状态筛选",
                    "分页",
                ],
                "affected_modules": [
                    "src/views"
                ],
                "verification": [
                    "查询功能正常"
                ],
            },
            {
                "order": 4,
                "title": "测试与验证",
                "goal": "验证功能正确",
                "actions": [
                    "Vitest",
                    "Playwright",
                    "lint",
                    "typecheck",
                ],
                "affected_modules": [
                    "tests"
                ],
                "verification": [
                    "全部测试通过"
                ],
            },
        ],

        completion_criteria=[
            "Acceptance Criteria 全部满足"
        ],

        risks=[],

        open_questions=[],
    )


@pytest.mark.asyncio
async def test_generate_task_plan():
    requirement = create_requirement()

    technical_spec = (
        create_technical_spec()
    )

    implementation_plan = (
        create_implementation_plan()
    )

    project_context = (
        create_project_context()
    )

    planner = TaskPlanner(
        llm=FakeTaskLLM()
    )

    task_plan = await planner.generate(
        requirement=requirement,
        technical_spec=technical_spec,
        implementation_plan=implementation_plan,
        project_context=project_context,
    )

    assert isinstance(
        task_plan,
        TaskPlan,
    )

    assert len(task_plan.tasks) == 5

    assert (
            task_plan.tasks[0].task_id
            == "TASK-001"
    )

    assert (
            task_plan.tasks[1].depends_on
            == ["TASK-001"]
    )


async def create_task_plan() -> TaskPlan:
    planner = TaskPlanner(
        llm=FakeTaskLLM()
    )

    return await planner.generate(
        requirement=create_requirement(),
        technical_spec=create_technical_spec(),
        implementation_plan=(
            create_implementation_plan()
        ),
        project_context=create_project_context(),
    )


@pytest.mark.asyncio
async def test_task_plan_validation_success():
    task_plan = await create_task_plan()

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(
            create_implementation_plan()
        ),
    )

    assert result.is_valid is True

    assert result.errors == []


@pytest.mark.asyncio
async def test_task_plan_missing_dependency():
    task_plan = await create_task_plan()

    task_plan.tasks[0].depends_on = [
        "TASK-999"
    ]

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(
            create_implementation_plan()
        ),
    )

    assert result.is_valid is False

    assert any(
        "TASK-999" in error
        for error in result.errors
    )


@pytest.mark.asyncio
async def test_task_plan_cycle():
    task_plan = await create_task_plan()

    task_plan.tasks[0].depends_on = [
        "TASK-002"
    ]

    task_plan.tasks[1].depends_on = [
        "TASK-001"
    ]

    assert (
            has_dependency_cycle(
                task_plan
            )
            is True
    )

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(
            create_implementation_plan()
        ),
    )

    assert result.is_valid is False

    assert (
            "TaskPlan 中存在循环依赖"
            in result.errors
    )
