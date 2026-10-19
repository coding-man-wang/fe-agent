from agent_core.change_surface.aggregate import ChangeSurfaceAggregator
from agent_core.change_surface.model import ResolvedChangeSurface


def test_aggregator_resolves_cross_task_role_conflict():
    code_change = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/views/UsersView.vue",
            "src/stores/users.ts",
        ],
        dependencies=[
            "src/api/users.ts",
        ],
    )

    test_task = ResolvedChangeSurface(
        task_id="TASK-003",
        dependencies=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        tests=[
            "tests/unit/usersStore.test.ts",
        ],
    )

    combined = ChangeSurfaceAggregator.combine(
        task_id="CASE-006",
        surfaces=[code_change, test_task],
    )

    assert combined.implementation_targets == [
        "src/stores/users.ts",
        "src/views/UsersView.vue",
    ]
    assert combined.dependencies == [
        "src/api/users.ts",
    ]
    assert combined.tests == [
        "tests/unit/usersStore.test.ts",
    ]
    assert any(
        "src/stores/users.ts" in item and "归一化" in item
        for item in combined.rationale
    )


def test_aggregator_prefers_test_over_dependency_when_not_change_target():
    first = ResolvedChangeSurface(
        task_id="TASK-002",
        dependencies=["tests/unit/usersStore.test.ts"],
    )
    second = ResolvedChangeSurface(
        task_id="TASK-003",
        tests=["tests/unit/usersStore.test.ts"],
    )

    combined = ChangeSurfaceAggregator.combine(
        task_id="CASE-006",
        surfaces=[first, second],
    )

    assert combined.tests == ["tests/unit/usersStore.test.ts"]
    assert combined.dependencies == []

from agent_core.sdd.model import TaskDefinition, TaskPlan


def make_task_plan_for_aggregation():
    return TaskPlan(
        summary="aggregation",
        tasks=[
            TaskDefinition(
                task_id="TASK-002",
                task_type="code_change",
                title="实现",
                objective="修改生产代码",
                source_step_order=1,
            ),
            TaskDefinition(
                task_id="TASK-005",
                task_type="test",
                title="测试",
                objective="补充测试",
                source_step_order=2,
                depends_on=["TASK-002"],
            ),
        ],
    )


def test_aggregator_is_task_type_aware_for_test_task():
    code_change = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/views/UsersView.vue"],
    )
    # 模拟模型在 test Task 中把测试文件/生产文件错误标成 implementation target。
    test_task = ResolvedChangeSurface(
        task_id="TASK-005",
        implementation_targets=[
            "tests/unit/usersStore.test.ts",
            "src/stores/users.ts",
        ],
    )

    combined = ChangeSurfaceAggregator.combine(
        task_id="CASE-006",
        surfaces=[code_change, test_task],
        task_plan=make_task_plan_for_aggregation(),
    )

    assert combined.implementation_targets == ["src/views/UsersView.vue"]
    assert combined.tests == ["tests/unit/usersStore.test.ts"]
    assert combined.dependencies == ["src/stores/users.ts"]


def test_aggregator_consolidates_duplicate_unresolved_questions():
    first = ResolvedChangeSurface(
        task_id="TASK-002",
        unresolved=[
            "PaginationBar.vue 的具体 props/事件尚未确认，需要确认是否支持 pageSize。",
            "现有请求防抖/去重机制未确认。",
        ],
    )
    second = ResolvedChangeSurface(
        task_id="TASK-005",
        unresolved=[
            "PaginationBar.vue 的 props 和事件未在证据中确认，需确认 pageSize 接口。",
            "现有请求防抖或去重机制未在证据中明确。",
        ],
    )

    combined = ChangeSurfaceAggregator.combine(
        task_id="CASE-006",
        surfaces=[first, second],
    )

    assert len(combined.unresolved) <= 2
