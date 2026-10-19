from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.change_surface.model import InvestigationArtifact
from agent_core.sdd.model import TaskDefinition, TaskPlan


def _task(task_id: str, task_type: str, depends_on: list[str] | None = None):
    return TaskDefinition(
        task_id=task_id,
        task_type=task_type,
        title=task_id,
        objective=task_id,
        source_step_order=1,
        depends_on=list(depends_on or []),
    )


def test_resolver_finds_transitive_investigation(tmp_path):
    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")
    store.put_model(
        task_id="TASK-001",
        artifact_type="investigation",
        model=InvestigationArtifact(
            task_id="TASK-001",
            summary="facts",
        ),
    )

    plan = TaskPlan(
        summary="demo",
        tasks=[
            _task("TASK-003", "code_change", ["TASK-002"]),
            _task("TASK-001", "investigation"),
            _task("TASK-002", "code_change", ["TASK-001"]),
        ],
    )

    resolver = TaskArtifactResolver(store=store)
    artifacts = resolver.upstream_investigations(
        task=plan.tasks[0],
        task_plan=plan,
    )

    assert [item.task_id for item in artifacts] == ["TASK-001"]
