from agent_core.artifacts import ArtifactDagBuilder, JsonTaskArtifactStore
from agent_core.change_surface.model import InvestigationArtifact
from agent_core.sdd.model import TaskDefinition, TaskPlan


def test_artifact_dag_connects_upstream_artifact_to_downstream_task(tmp_path):
    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")
    store.put_model(
        task_id="TASK-001",
        artifact_type="investigation",
        model=InvestigationArtifact(task_id="TASK-001", summary="facts"),
    )
    plan = TaskPlan(
        summary="demo",
        tasks=[
            TaskDefinition(
                task_id="TASK-001",
                task_type="investigation",
                title="investigate",
                objective="investigate",
                source_step_order=1,
            ),
            TaskDefinition(
                task_id="TASK-002",
                task_type="code_change",
                title="change",
                objective="change",
                source_step_order=2,
                depends_on=["TASK-001"],
            ),
        ],
    )
    graph = ArtifactDagBuilder().build(task_plan=plan, store=store)
    assert any(
        edge.relation == "consumes" and edge.target == "task:TASK-002"
        for edge in graph.edges
    )
