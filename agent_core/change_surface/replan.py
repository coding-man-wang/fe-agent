from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.artifacts.resolver import (
        TaskArtifactResolver,
    )

from agent_core.change_surface.model import (
    RepositoryAwareTask,
    RepositoryAwareTaskPlan,
    ResolvedChangeSurface,
)
from agent_core.sdd.model import (
    TaskDefinition,
    TaskPlan,
)


class RepositoryAwareReplanner:
    """
    使用已经存在的 Repository Evidence
    把低承诺 Logical Plan
    精炼成 Repository-aware Plan。

    IMPORTANT:

    这里不调用 LLM。

    Replan 只是把：

        Logical Responsibility
        +
        ResolvedChangeSurface

    合并成文件级执行视图。

    它不是最终 Write Permission。

    FullExecution 仍必须在最新
    Repository Revision 上重新解析
    Current Change Surface。
    """

    def __init__(
        self,
        *,
        artifact_resolver: TaskArtifactResolver,
    ) -> None:

        self.artifact_resolver = artifact_resolver

    def build(
        self,
        *,
        task_plan: TaskPlan,
        repository_revision: str | None = None,
    ) -> RepositoryAwareTaskPlan:

        resolved_tasks: list[RepositoryAwareTask] = []

        for task in task_plan.tasks:

            surface = self.artifact_resolver.latest_change_surface(task.task_id)

            if surface is None:

                resolved_tasks.append(
                    self._without_surface(
                        task=task,
                        planning_stage=(task_plan.planning_stage),
                        repository_revision=(repository_revision),
                    )
                )

                continue

            resolved_tasks.append(
                self._from_surface(
                    task=task,
                    surface=surface,
                    planning_stage=(task_plan.planning_stage),
                    repository_revision=(repository_revision),
                )
            )

        return RepositoryAwareTaskPlan(
            summary=(
                "Logical TaskPlan 已根据 "
                "Repository Evidence "
                "精炼为 Repository-aware "
                "文件级计划；"
                "该计划绑定到特定 "
                "Repository Revision，"
                "最终 Write Permission "
                "仍由执行时的 "
                "Current Change Surface "
                "决定。"
            ),
            source_planning_stage=(task_plan.planning_stage),
            repository_revision=(repository_revision),
            tasks=resolved_tasks,
        )

    @staticmethod
    def _base_fields(
        *,
        task: TaskDefinition,
        planning_stage: str,
        repository_revision: str | None,
    ) -> dict:

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "title": task.title,
            "objective": task.objective,
            "depends_on": list(task.depends_on),
            "source_planning_stage": (planning_stage),
            "responsibilities": list(task.responsibilities),
            "investigation_questions": (list(task.investigation_questions)),
            "acceptance_criteria_refs": (list(task.acceptance_criteria_refs)),
            "original_repository_hints": (task.repository_search_hints()),
            "original_target_modules": (list(task.target_modules)),
            "repository_revision": (repository_revision),
        }

    @classmethod
    def _without_surface(
        cls,
        *,
        task: TaskDefinition,
        planning_stage: str,
        repository_revision: str | None,
    ) -> RepositoryAwareTask:

        # Investigation 本身负责生成 Evidence，
        # verification 本身不产生代码修改。
        #
        # 这两类 Task 不需要
        # ResolvedChangeSurface
        # 才能认为 Planning Resolution 完成。

        if task.task_type in {
            "investigation",
            "verification",
        }:

            status = "resolved"

            if task.task_type == "investigation":
                reason = (
                    "investigation Task "
                    "负责建立 Repository "
                    "Evidence，本身不要求 "
                    "Change Surface"
                )
            else:
                reason = (
                    "verification Task "
                    "不产生代码 Side Effect，"
                    "不要求 Change Surface"
                )

        else:

            status = "unresolved"

            reason = "当前 Side Effect Task " "尚无 ResolvedChangeSurface"

        return RepositoryAwareTask(
            **cls._base_fields(
                task=task,
                planning_stage=(planning_stage),
                repository_revision=(repository_revision),
            ),
            resolution_status=status,
            execution_disposition=None,
            reasons=[reason],
        )

    @classmethod
    def _from_surface(
        cls,
        *,
        task: TaskDefinition,
        surface: ResolvedChangeSurface,
        planning_stage: str,
        repository_revision: str | None,
    ) -> RepositoryAwareTask:

        disposition = surface.execution_disposition

        if disposition == "insufficient_evidence":

            status = "partial"

        elif disposition in {
            "change_required",
            "no_change_required",
        }:

            status = "resolved"

        else:

            status = (
                "resolved"
                if (surface.implementation_targets or surface.tests)
                else "partial"
            )

            if surface.unresolved:
                status = "partial"

        return RepositoryAwareTask(
            **cls._base_fields(
                task=task,
                planning_stage=(planning_stage),
                repository_revision=(repository_revision),
            ),
            resolved_target_files=list(surface.implementation_targets),
            dependency_files=list(surface.dependencies),
            reference_files=list(surface.references),
            test_files=list(surface.tests),
            resolution_status=status,
            execution_disposition=(disposition),
            reasons=[
                *surface.rationale,
                *surface.unresolved,
            ],
        )
