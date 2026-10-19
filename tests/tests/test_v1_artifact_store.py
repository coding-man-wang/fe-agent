from agent_core.artifacts import JsonTaskArtifactStore, TaskArtifactResolver
from agent_core.change_surface.model import InvestigationArtifact
from agent_core.sdd.model import TaskDefinition


def test_artifact_store_and_upstream_resolution(tmp_path):
    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")
    artifact = InvestigationArtifact(
        task_id="TASK-001",
        summary="found facts",
        findings=["pageSize exists"],
    )
    store.put_model(
        task_id="TASK-001",
        artifact_type="investigation",
        model=artifact,
    )

    task = TaskDefinition(
        task_id="TASK-002",
        task_type="code_change",
        title="change",
        objective="change",
        source_step_order=2,
        depends_on=["TASK-001"],
    )
    resolver = TaskArtifactResolver(store=store)
    resolved = resolver.upstream_investigations(task=task)
    assert len(resolved) == 1
    assert resolved[0].findings == ["pageSize exists"]
