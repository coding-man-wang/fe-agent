from pydantic import (
    BaseModel,
    Field,
)

from agent_core.sdd.model import (
    ImplementationPlan,
    ProjectContext,
    RequirementSpec,
    TaskDefinition,
    TaskPlan,
)


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

    def visit(
        task_id: str,
    ) -> bool:

        if task_id in visiting:
            return True

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


def _ancestor_task_ids(
    *,
    task: TaskDefinition,
    task_map: dict[
        str,
        TaskDefinition,
    ],
) -> set[str]:

    visited: set[str] = set()

    stack = list(task.depends_on)

    while stack:

        task_id = stack.pop()

        if task_id in visited:
            continue

        visited.add(task_id)

        upstream = task_map.get(task_id)

        if upstream is None:
            continue

        stack.extend(upstream.depends_on)

    return visited


def _has_investigation_ancestor(
    *,
    task: TaskDefinition,
    task_map: dict[
        str,
        TaskDefinition,
    ],
) -> bool:

    ancestor_ids = _ancestor_task_ids(
        task=task,
        task_map=task_map,
    )

    return any(
        task_map[task_id].task_type == "investigation"
        for task_id in ancestor_ids
        if task_id in task_map
    )


def validate_task_plan(
    plan: TaskPlan,
    requirement: RequirementSpec,
    implementation_plan: ImplementationPlan,
    project_context: ProjectContext | None = None,
) -> TaskPlanValidationResult:

    errors: list[str] = []

    task_ids = [task.task_id for task in plan.tasks]

    task_id_set = set(task_ids)

    task_map = {task.task_id: task for task in plan.tasks}

    # ==================================
    # Existing Contract
    # ==================================

    if len(task_ids) != len(task_id_set):
        errors.append("TaskPlan 中存在重复的 task_id")

    valid_step_orders = {step.order for step in implementation_plan.steps}

    valid_acceptance_criteria = set(requirement.acceptance_criteria)

    for task in plan.tasks:

        if task.source_step_order not in valid_step_orders:
            errors.append(
                f"{task.task_id} "
                "引用了不存在的 "
                "ImplementationStep: "
                f"{task.source_step_order}"
            )

        for dependency in task.depends_on:

            if dependency not in task_id_set:
                errors.append(f"{task.task_id} " "依赖不存在的任务 " f"{dependency}")

            if dependency == task.task_id:
                errors.append(f"{task.task_id} " "不能依赖自己")

        for criterion in task.acceptance_criteria_refs:

            if criterion not in valid_acceptance_criteria:
                errors.append(
                    f"{task.task_id} "
                    "引用了不存在的 "
                    "Acceptance Criteria: "
                    f"{criterion}"
                )

        if not task.verification:

            errors.append(f"{task.task_id} " "缺少 verification")

    if has_dependency_cycle(plan):
        errors.append("TaskPlan 中存在循环依赖")

    # ==================================
    # v1.3 Logical Planning Contract
    #
    # 只约束新 Plan。
    #
    # 历史 frozen plan 仍然可以读取。
    # ==================================

    if plan.planning_stage == "logical":

        allowed_repository_hints: set[str] = set()

        if project_context is not None:
            allowed_repository_hints = {
                value.replace("\\", "/").strip().strip("/").lower()
                for value in (project_context.repository_hint_candidates())
            }

        for task in plan.tasks:

            # --------------------------
            # Legacy 字段禁止继续生成
            # --------------------------

            if task.target_modules:

                errors.append(
                    f"{task.task_id} "
                    "logical plan "
                    "不得使用 target_modules；"
                    "请使用 responsibilities / "
                    "investigation_questions / "
                    "repository_hints"
                )

            # --------------------------
            # verification 是 Gate，
            # 不强制业务责任。
            # --------------------------

            if task.task_type != "verification" and not [
                value for value in task.responsibilities if value.strip()
            ]:
                errors.append(
                    f"{task.task_id} " "logical plan " "缺少 responsibilities"
                )

            # --------------------------
            # Investigation 必须明确：
            # 到仓库里要回答什么问题。
            # --------------------------

            if task.task_type == "investigation" and not [
                value for value in (task.investigation_questions) if value.strip()
            ]:
                errors.append(
                    f"{task.task_id} "
                    "investigation Task "
                    "缺少 "
                    "investigation_questions"
                )

            # --------------------------
            # 与当前 Harness Invariant
            # 对齐：
            #
            # code_change 不能在没有
            # Investigation Evidence
            # 的情况下直接进入写阶段。
            # --------------------------

            if task.task_type == "code_change" and not (
                _has_investigation_ancestor(
                    task=task,
                    task_map=task_map,
                )
            ):
                errors.append(
                    f"{task.task_id} "
                    "code_change Task "
                    "缺少上游 "
                    "investigation ancestor"
                )

            if task.execution_obligations:
                raise ValueError(
                    f"{task.task_id} logical planning "
                    "阶段不允许声明 execution_obligations；"
                    "该字段只能由 Runtime "
                    "ExecutionPlanCompressor 生成"
                )
            # --------------------------
            # Repository Hint
            # 必须来自 ProjectContext。
            # --------------------------

            if project_context is not None:

                for hint in task.repository_hints:

                    normalized = (
                        hint.replace(
                            "\\",
                            "/",
                        )
                        .strip()
                        .strip("/")
                        .lower()
                    )

                    if normalized not in allowed_repository_hints:
                        errors.append(
                            f"{task.task_id} "
                            "包含未经 "
                            "ProjectContext "
                            "证明的 "
                            "repository_hint: "
                            f"{hint}"
                        )

    return TaskPlanValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
    )
