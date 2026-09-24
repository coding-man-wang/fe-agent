from openai import BaseModel
from pydantic import Field

from agent_core.sdd.model import TaskPlan, RequirementSpec, ImplementationPlan


class TaskPlanValidationResult(BaseModel):
    is_valid: bool

    errors: list[str] = Field(
        default_factory=list,
    )


def has_dependency_cycle(
    plan: TaskPlan,
) -> bool:
    graph = {task.task_id: task.depends_on for task in plan.tasks}

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        # 当前 DFS 路径再次遇到
        # 说明形成了环
        if task_id in visiting:
            return True

        # 以前已经完整检查过
        if task_id in visited:
            return False

        visiting.add(task_id)

        for dependency in graph.get(
            task_id,
            [],
        ):
            if dependency not in graph:
                continue

            if visit(dependency):
                return True

        visiting.remove(task_id)
        visited.add(task_id)

        return False

    for task_id in graph:
        if visit(task_id):
            return True

    return False


def validate_task_plan(
    plan: TaskPlan,
    requirement: RequirementSpec,
    implementation_plan: ImplementationPlan,
) -> TaskPlanValidationResult:
    errors = []

    task_ids = [task.task_id for task in plan.tasks]

    task_id_set = set(task_ids)

    # 1. task_id 不能重复
    if len(task_ids) != len(task_id_set):
        errors.append("TaskPlan 中存在重复的 task_id")

    valid_step_orders = {step.order for step in implementation_plan.steps}

    valid_acceptance_criteria = set(requirement.acceptance_criteria)

    for task in plan.tasks:

        # 2. source_step_order 必须存在
        if task.source_step_order not in valid_step_orders:
            errors.append(
                f"{task.task_id} 引用了不存在的 "
                f"ImplementationStep: "
                f"{task.source_step_order}"
            )

        # 3. dependency 必须存在
        for dependency in task.depends_on:

            if dependency not in task_id_set:
                errors.append(f"{task.task_id} 依赖不存在的任务 " f"{dependency}")

            # 4. 不能依赖自己
            if dependency == task.task_id:
                errors.append(f"{task.task_id} 不能依赖自己")

        # 5. Acceptance Criteria 引用必须真实存在
        for criterion in task.acceptance_criteria_refs:
            if criterion not in valid_acceptance_criteria:
                errors.append(
                    f"{task.task_id} 引用了不存在的 "
                    f"Acceptance Criteria: "
                    f"{criterion}"
                )

        # 6. 每个任务都应该能够验证
        if not task.verification:
            errors.append(f"{task.task_id} 缺少 verification")

        # 7. 检查循环依赖
    if has_dependency_cycle(plan):
        errors.append("TaskPlan 中存在循环依赖")

    return TaskPlanValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
    )
