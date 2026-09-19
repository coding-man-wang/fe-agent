from collections import defaultdict
from heapq import heapify, heappop, heappush

from agent_core.sdd.model import (
    TaskDefinition,
    TaskPlan,
)


def topological_sort_tasks(
        plan: TaskPlan,
) -> list[TaskDefinition]:
    """
    根据 Task 的 depends_on 关系，
    返回合法的任务执行顺序。
    """

    task_map = {
        task.task_id: task
        for task in plan.tasks
    }

    # 每个任务还有多少个未完成依赖
    indegree = {
        task.task_id: len(task.depends_on)
        for task in plan.tasks
    }

    # dependency -> 哪些任务依赖它
    dependents: dict[str, list[str]] = (
        defaultdict(list)
    )

    for task in plan.tasks:

        for dependency in task.depends_on:

            if dependency not in task_map:
                raise ValueError(
                    f"{task.task_id} 依赖不存在的任务 "
                    f"{dependency}"
                )

            dependents[dependency].append(
                task.task_id
            )

    # 所有没有前置依赖的任务
    ready = [
        task_id
        for task_id, degree
        in indegree.items()
        if degree == 0
    ]

    # 让测试和结果顺序稳定
    heapify(ready)

    ordered_task_ids: list[str] = []

    while ready:

        task_id = heappop(ready)

        ordered_task_ids.append(
            task_id
        )

        for dependent_id in dependents[
            task_id
        ]:

            indegree[dependent_id] -= 1

            if indegree[dependent_id] == 0:
                heappush(
                    ready,
                    dependent_id,
                )

    # 如果没有把所有任务排完，
    # 说明一定存在循环依赖
    if (
            len(ordered_task_ids)
            != len(plan.tasks)
    ):
        raise ValueError(
            "TaskPlan 中存在循环依赖，"
            "无法生成执行顺序"
        )

    return [
        task_map[task_id]
        for task_id in ordered_task_ids
    ]


def build_execution_batches(
        plan: TaskPlan,
) -> list[list[TaskDefinition]]:
    """
    根据 Task Dependency DAG，
    把任务拆分成可并行执行的批次。
    """

    task_map = {
        task.task_id: task
        for task in plan.tasks
    }

    indegree = {
        task.task_id: len(task.depends_on)
        for task in plan.tasks
    }

    dependents: dict[str, list[str]] = (
        defaultdict(list)
    )

    for task in plan.tasks:

        for dependency in task.depends_on:

            if dependency not in task_map:
                raise ValueError(
                    f"{task.task_id} 依赖不存在的任务 "
                    f"{dependency}"
                )

            dependents[dependency].append(
                task.task_id
            )

    ready = sorted([
        task_id
        for task_id, degree
        in indegree.items()
        if degree == 0
    ])

    batches: list[
        list[TaskDefinition]
    ] = []

    processed_count = 0

    while ready:

        current_batch_ids = ready

        batches.append([
            task_map[task_id]
            for task_id in current_batch_ids
        ])

        processed_count += len(
            current_batch_ids
        )

        next_ready: list[str] = []

        for task_id in current_batch_ids:

            for dependent_id in dependents[
                task_id
            ]:

                indegree[dependent_id] -= 1

                if (
                        indegree[dependent_id]
                        == 0
                ):
                    next_ready.append(
                        dependent_id
                    )

        ready = sorted(next_ready)

    if processed_count != len(plan.tasks):
        raise ValueError(
            "TaskPlan 中存在循环依赖，"
            "无法生成执行批次"
        )

    return batches


def get_ready_tasks(
        plan: TaskPlan,
        completed_task_ids: set[str],
) -> list[TaskDefinition]:
    """
    根据已经完成的任务，
    返回当前可以执行的 Task。
    """

    ready_tasks: list[
        TaskDefinition
    ] = []

    for task in plan.tasks:

        # 已经完成的不再返回
        if (
                task.task_id
                in completed_task_ids
        ):
            continue

        all_dependencies_completed = all(
            dependency
            in completed_task_ids
            for dependency
            in task.depends_on
        )

        if all_dependencies_completed:
            ready_tasks.append(task)

    return sorted(
        ready_tasks,
        key=lambda task: task.task_id,
    )
