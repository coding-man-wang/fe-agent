from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.change_surface.model import (
    InvestigationArtifact,
    ResolvedChangeSurface,
)
from agent_core.sdd.model import TaskDefinition, TaskPlan
from agent_core.artifacts.execution_model import (
    TaskExecutionArtifact,
)


class TaskArtifactResolver:
    """Resolve durable upstream artifacts for one Task.

    Without a TaskPlan we preserve the original direct-dependency behavior.
    When a TaskPlan is available we traverse the dependency DAG so downstream
    tasks can inherit InvestigationArtifact facts from transitive ancestors.
    """

    def __init__(self, *, store: JsonTaskArtifactStore) -> None:
        self.store = store

    @staticmethod
    def _ancestor_task_ids(
        *,
        task: TaskDefinition,
        task_plan: TaskPlan,
    ) -> list[str]:
        task_map = {item.task_id: item for item in task_plan.tasks}
        visited: set[str] = set()
        stack = list(task.depends_on)

        while stack:
            upstream_task_id = stack.pop()
            if upstream_task_id in visited:
                continue

            upstream_task = task_map.get(upstream_task_id)
            if upstream_task is None:
                # TaskPlan validation/scheduler will normally reject this.
                # Resolver stays defensive and simply ignores an unknown edge.
                continue

            visited.add(upstream_task_id)
            stack.extend(upstream_task.depends_on)

        return sorted(visited)

    def upstream_investigation_task_ids(
        self,
        *,
        task: TaskDefinition,
        task_plan: TaskPlan | None = None,
    ) -> list[str]:
        if task_plan is None:
            candidate_ids = list(task.depends_on)
            task_map: dict[str, TaskDefinition] | None = None
        else:
            candidate_ids = self._ancestor_task_ids(
                task=task,
                task_plan=task_plan,
            )
            task_map = {item.task_id: item for item in task_plan.tasks}

        result: list[str] = []
        for task_id in candidate_ids:
            if task_map is not None:
                upstream_task = task_map.get(task_id)
                if upstream_task is None or upstream_task.task_type != "investigation":
                    continue
            # With no TaskPlan we cannot know the upstream task type.  The
            # existence of an investigation artifact is therefore the source
            # of truth.
            envelope = self.store.latest(
                task_id=task_id,
                artifact_type="investigation",
            )
            if envelope is not None or task_map is not None:
                result.append(task_id)

        return result

    def upstream_investigations(
        self,
        *,
        task: TaskDefinition,
        task_plan: TaskPlan | None = None,
    ) -> list[InvestigationArtifact]:
        artifacts: list[InvestigationArtifact] = []
        for upstream_task_id in self.upstream_investigation_task_ids(
            task=task,
            task_plan=task_plan,
        ):
            envelope = self.store.latest(
                task_id=upstream_task_id,
                artifact_type="investigation",
            )
            if envelope is None:
                continue
            artifacts.append(InvestigationArtifact.model_validate(envelope.payload))
        return artifacts

    def upstream_execution_task_ids(
        self,
        *,
        task: TaskDefinition,
        task_plan: TaskPlan | None = None,
    ) -> list[str]:
        """
        找出当前 Task 的祖先 Task 中，
        已经真实产生 Execution Artifact 的 Task。

        有 TaskPlan 时：
            遍历完整 depends_on DAG。

        没有 TaskPlan 时：
            只检查直接 depends_on。
        """

        if task_plan is None:
            candidate_ids = list(task.depends_on)
        else:
            candidate_ids = self._ancestor_task_ids(
                task=task,
                task_plan=task_plan,
            )

        result: list[str] = []

        for task_id in candidate_ids:
            envelope = self.store.latest(
                task_id=task_id,
                artifact_type="execution",
            )

            if envelope is not None:
                result.append(task_id)

        return result

    def upstream_executions(
        self,
        *,
        task: TaskDefinition,
        task_plan: TaskPlan | None = None,
    ) -> list[TaskExecutionArtifact]:
        """
        读取当前 Task 所有已完成的
        upstream Execution Artifact。
        """

        artifacts: list[TaskExecutionArtifact] = []

        for upstream_task_id in self.upstream_execution_task_ids(
            task=task,
            task_plan=task_plan,
        ):
            envelope = self.store.latest(
                task_id=upstream_task_id,
                artifact_type="execution",
            )

            if envelope is None:
                continue

            artifacts.append(TaskExecutionArtifact.model_validate(envelope.payload))

        return artifacts

    def latest_change_surface(
        self,
        task_id: str,
    ) -> ResolvedChangeSurface | None:
        envelope = self.store.latest(
            task_id=task_id,
            artifact_type="change_surface",
        )
        if envelope is None:
            return None
        return ResolvedChangeSurface.model_validate(envelope.payload)
