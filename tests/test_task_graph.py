from agent_core.sdd.model import (
    TaskPlan,
)

from agent_core.sdd.task_graph import (
    build_execution_batches,
    get_ready_tasks,
    topological_sort_tasks,
)


def create_task_plan() -> TaskPlan:
    return TaskPlan(
        summary="用户列表开发任务",
        tasks=[
            {
                "task_id": "TASK-001",
                "task_type": "investigation",
                "title": "确认 API",
                "objective": "确认用户 API 契约",
                "source_step_order": 1,
                "depends_on": [],
                "inputs": [],
                "target_modules": ["用户 API"],
                "expected_outputs": ["API 契约"],
                "verification": ["API 参数确认完成"],
                "acceptance_criteria_refs": [],
            },
            {
                "task_id": "TASK-002",
                "task_type": "code_change",
                "title": "实现 API",
                "objective": "实现用户 API 封装",
                "source_step_order": 1,
                "depends_on": ["TASK-001"],
                "inputs": ["API 契约"],
                "target_modules": ["src/api"],
                "expected_outputs": ["API 封装"],
                "verification": ["API 调用正常"],
                "acceptance_criteria_refs": [],
            },
            {
                "task_id": "TASK-003",
                "task_type": "code_change",
                "title": "实现页面",
                "objective": "实现用户列表页面",
                "source_step_order": 2,
                "depends_on": ["TASK-002"],
                "inputs": ["API 封装"],
                "target_modules": ["src/views"],
                "expected_outputs": ["用户列表页面"],
                "verification": ["页面展示正常"],
                "acceptance_criteria_refs": ["页面能够展示用户列表"],
            },
            {
                "task_id": "TASK-004",
                "task_type": "test",
                "title": "编写测试",
                "objective": "增加用户列表测试",
                "source_step_order": 4,
                "depends_on": ["TASK-002"],
                "inputs": ["API 封装"],
                "target_modules": ["tests"],
                "expected_outputs": ["自动化测试"],
                "verification": ["测试能够执行"],
                "acceptance_criteria_refs": [],
            },
            {
                "task_id": "TASK-005",
                "task_type": "verification",
                "title": "最终验证",
                "objective": "验证需求是否完成",
                "source_step_order": 4,
                "depends_on": [
                    "TASK-003",
                    "TASK-004",
                ],
                "inputs": ["页面", "测试"],
                "target_modules": ["用户列表功能"],
                "expected_outputs": ["最终验证结果"],
                "verification": ["所有测试通过"],
                "acceptance_criteria_refs": ["页面能够展示用户列表"],
            },
        ],
        completion_criteria=["所有任务完成"],
    )


def test_topological_sort():
    plan = create_task_plan()

    ordered = topological_sort_tasks(plan)

    ordered_ids = [task.task_id for task in ordered]

    assert ordered_ids.index("TASK-001") < ordered_ids.index("TASK-002")

    assert ordered_ids.index("TASK-002") < ordered_ids.index("TASK-003")

    assert ordered_ids.index("TASK-002") < ordered_ids.index("TASK-004")

    assert ordered_ids.index("TASK-003") < ordered_ids.index("TASK-005")

    assert ordered_ids.index("TASK-004") < ordered_ids.index("TASK-005")


def test_execution_batches():
    plan = create_task_plan()

    batches = build_execution_batches(plan)

    batch_ids = [[task.task_id for task in batch] for batch in batches]

    assert batch_ids == [
        ["TASK-001"],
        ["TASK-002"],
        [
            "TASK-003",
            "TASK-004",
        ],
        ["TASK-005"],
    ]


def test_execution_batches():
    plan = create_task_plan()

    batches = build_execution_batches(plan)

    batch_ids = [[task.task_id for task in batch] for batch in batches]

    assert batch_ids == [
        ["TASK-001"],
        ["TASK-002"],
        [
            "TASK-003",
            "TASK-004",
        ],
        ["TASK-005"],
    ]


def test_get_ready_tasks_initial():
    plan = create_task_plan()

    ready = get_ready_tasks(
        plan,
        completed_task_ids=set(),
    )

    ready_ids = [task.task_id for task in ready]

    assert ready_ids == ["TASK-001"]
