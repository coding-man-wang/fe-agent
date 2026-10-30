import pytest

from agent_core.llm.model import (
    ChatResponse,
)
from agent_core.sdd.model import (
    RequirementSpec,
    ProjectContext,
    TechnicalSpec,
    ImplementationPlan,
    TaskPlan,
)
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
            "planning_stage": "logical",

            "summary":
              "将用户列表需求拆分为调查、实现、测试和验证责任",

            "tasks": [
              {
                "task_id": "TASK-001",
                "task_type": "investigation",
                "title": "确认用户列表现有实现责任",
                "objective":
                  "确认用户分页 API、页面交互和路由相关的现有实现事实",

                "source_step_order": 1,

                "depends_on": [],

                "inputs": [
                  "TechnicalSpec"
                ],

                "responsibilities": [
                  "确认用户分页数据访问责任",
                  "确认用户列表页面交互责任",
                  "确认页面与路由之间的现有关系"
                ],

                "investigation_questions": [
                  "用户分页请求参数和响应结构由哪一层负责？",
                  "用户列表页面当前由哪一层负责展示和交互？",
                  "用户列表页面与路由之间如何关联？"
                ],

                "repository_hints": [
                  "src/api",
                  "src/views",
                  "src/router"
                ],

                "target_modules": [],

                "expected_outputs": [
                  "用户列表相关 Repository Evidence"
                ],

                "verification": [
                  "调查结论必须基于真实仓库代码证据"
                ],

                "acceptance_criteria_refs": []
              },

              {
                "task_id": "TASK-002",
                "task_type": "code_change",
                "title": "实现用户数据访问能力",
                "objective":
                  "提供满足用户列表需求的数据访问能力",

                "source_step_order": 1,

                "depends_on": [
                  "TASK-001"
                ],

                "inputs": [
                  "用户列表 Repository Evidence"
                ],

                "responsibilities": [
                  "实现符合现有契约的用户分页数据访问能力"
                ],

                "investigation_questions": [],

                "repository_hints": [
                  "src/api"
                ],

                "target_modules": [],

                "expected_outputs": [
                  "满足分页需求的数据访问行为"
                ],

                "verification": [
                  "分页查询行为符合已确认的 API 契约"
                ],

                "acceptance_criteria_refs": []
              },

              {
                "task_id": "TASK-003",
                "task_type": "code_change",
                "title": "实现用户列表交互",
                "objective":
                  "完成用户列表展示、搜索、筛选和分页",

                "source_step_order": 2,

                "depends_on": [
                  "TASK-002"
                ],

                "inputs": [
                  "用户列表 Repository Evidence",
                  "用户数据访问能力"
                ],

                "responsibilities": [
                  "实现用户列表展示",
                  "实现用户名搜索行为",
                  "实现状态筛选行为",
                  "实现分页切换行为"
                ],

                "investigation_questions": [],

                "repository_hints": [
                  "src/views",
                  "src/router"
                ],

                "target_modules": [],

                "expected_outputs": [
                  "满足用户列表需求的页面行为"
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
                  "验证用户列表主要行为",

                "source_step_order": 4,

                "depends_on": [
                  "TASK-003"
                ],

                "inputs": [
                  "用户列表实现",
                  "Testing Strategy"
                ],

                "responsibilities": [
                  "验证用户列表核心行为",
                  "验证搜索、筛选和分页行为"
                ],

                "investigation_questions": [],

                "repository_hints": [],

                "target_modules": [],

                "expected_outputs": [
                  "Unit Test Verification Evidence",
                  "E2E Verification Evidence"
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

                "responsibilities": [],

                "investigation_questions": [],

                "repository_hints": [],

                "target_modules": [],

                "expected_outputs": [
                  "最终 Verification Evidence"
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
        constraints=["使用 Vue3"],
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
        engineering_rules=["API 请求统一放在 src/api"],
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
        data_flow=["查询条件 -> API -> Table"],
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
        preconditions=["确认用户 API 契约"],
        steps=[
            {
                "order": 1,
                "title": "实现 API 能力",
                "goal": "提供用户数据访问能力",
                "actions": [
                    "确认 API",
                    "增加 API 封装",
                ],
                "affected_modules": ["src/api"],
                "verification": ["API 可正常使用"],
            },
            {
                "order": 2,
                "title": "实现页面",
                "goal": "展示用户列表",
                "actions": ["新增用户列表页面"],
                "affected_modules": [
                    "src/views",
                    "src/router",
                ],
                "verification": ["页面正常展示"],
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
                "affected_modules": ["src/views"],
                "verification": ["查询功能正常"],
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
                "affected_modules": ["tests"],
                "verification": ["全部测试通过"],
            },
        ],
        completion_criteria=["Acceptance Criteria 全部满足"],
        risks=[],
        open_questions=[],
    )


@pytest.mark.asyncio
async def test_generate_task_plan():
    requirement = create_requirement()

    technical_spec = create_technical_spec()

    implementation_plan = create_implementation_plan()

    project_context = create_project_context()

    planner = TaskPlanner(llm=FakeTaskLLM())

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

    assert task_plan.tasks[0].task_id == "TASK-001"

    assert task_plan.tasks[1].depends_on == ["TASK-001"]

    assert task_plan.planning_stage == "logical"

    assert all(not task.target_modules for task in task_plan.tasks)

    assert task_plan.tasks[0].investigation_questions

    assert task_plan.tasks[1].responsibilities

    assert task_plan.tasks[1].repository_hints == ["src/api"]


async def create_task_plan() -> TaskPlan:
    planner = TaskPlanner(llm=FakeTaskLLM())

    return await planner.generate(
        requirement=create_requirement(),
        technical_spec=create_technical_spec(),
        implementation_plan=(create_implementation_plan()),
        project_context=create_project_context(),
    )


@pytest.mark.asyncio
async def test_task_plan_validation_success():
    task_plan = await create_task_plan()

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(create_implementation_plan()),
        project_context=(create_project_context()),
    )

    assert result.is_valid is True

    assert result.errors == []


@pytest.mark.asyncio
async def test_task_plan_missing_dependency():
    task_plan = await create_task_plan()

    task_plan.tasks[0].depends_on = ["TASK-999"]

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(create_implementation_plan()),
        project_context=(create_project_context()),
    )

    assert result.is_valid is False

    assert any("TASK-999" in error for error in result.errors)


@pytest.mark.asyncio
async def test_task_plan_cycle():
    task_plan = await create_task_plan()

    task_plan.tasks[0].depends_on = ["TASK-002"]

    task_plan.tasks[1].depends_on = ["TASK-001"]

    assert has_dependency_cycle(task_plan) is True

    result = validate_task_plan(
        plan=task_plan,
        requirement=create_requirement(),
        implementation_plan=(create_implementation_plan()),
    )

    assert result.is_valid is False

    assert "TaskPlan 中存在循环依赖" in result.errors
