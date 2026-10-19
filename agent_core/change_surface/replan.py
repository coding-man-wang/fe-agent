from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.artifacts.resolver import TaskArtifactResolver

from agent_core.change_surface.model import (
    RepositoryAwareTask,
    RepositoryAwareTaskPlan,
    ResolvedChangeSurface,
)
from agent_core.sdd.model import TaskPlan


class RepositoryAwareReplanner:
    """
    用 Repository Facts / Change Surface 精炼初始 Plan。

    Replan 不只是把 module-level target 变成 file-level target，
    还允许 Repository Evidence 把初始 code_change Task 判定为 no-op
    或 insufficient_evidence。
    """

    def __init__(self, *, artifact_resolver: TaskArtifactResolver) -> None:
        self.artifact_resolver = artifact_resolver

    def build(self, *, task_plan: TaskPlan) -> RepositoryAwareTaskPlan:
        resolved_tasks: list[RepositoryAwareTask] = []
        for task in task_plan.tasks:
            surface = self.artifact_resolver.latest_change_surface(task.task_id)
            if surface is None:
                resolved_tasks.append(
                    RepositoryAwareTask(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        title=task.title,
                        depends_on=list(task.depends_on),
                        original_target_modules=list(task.target_modules),
                        resolution_status=(
                            "resolved"
                            if task.task_type == "investigation"
                            else "unresolved"
                        ),
                        execution_disposition=None,
                        reasons=[
                            "investigation task 不要求 change surface"
                            if task.task_type == "investigation"
                            else "尚无 ResolvedChangeSurface"
                        ],
                    )
                )
                continue

            resolved_tasks.append(
                self._from_surface(task=task, surface=surface)
            )

        return RepositoryAwareTaskPlan(
            summary=(
                "Initial TaskPlan 已根据 Repository Evidence 精炼为文件级执行计划；"
                "Repository Evidence 可以把初始 change task 判定为 change/no-op/evidence-gap"
            ),
            tasks=resolved_tasks,
        )

    @staticmethod
    def _from_surface(
        *,
        task,
        surface: ResolvedChangeSurface,
    ) -> RepositoryAwareTask:
        disposition = surface.execution_disposition

        if disposition == "insufficient_evidence":
            status = "partial"
        elif disposition in {"change_required", "no_change_required"}:
            status = "resolved"
        else:
            status = (
                "resolved"
                if surface.implementation_targets or surface.tests
                else "partial"
            )
            if surface.unresolved:
                status = "partial"

        return RepositoryAwareTask(
            task_id=task.task_id,
            task_type=task.task_type,
            title=task.title,
            depends_on=list(task.depends_on),
            original_target_modules=list(task.target_modules),
            resolved_target_files=list(surface.implementation_targets),
            dependency_files=list(surface.dependencies),
            reference_files=list(surface.references),
            test_files=list(surface.tests),
            resolution_status=status,
            execution_disposition=disposition,
            reasons=list(surface.rationale) + list(surface.unresolved),
        )
