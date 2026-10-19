from agent_core.artifacts.query_builder import ArtifactAwareRetrievalQueryBuilder
from agent_core.sdd.model import TaskDefinition


def test_structural_task_inputs_are_not_repository_queries():
    task = TaskDefinition(
        task_id="TASK-002",
        task_type="code_change",
        title="接入分页控件",
        objective="复用 pageSize",
        source_step_order=2,
        inputs=[
            "ImplementationPlan step 2",
            "TASK-001 输出：pageSize 流转链路与改动边界",
        ],
    )
    queries = ArtifactAwareRetrievalQueryBuilder(max_queries=10).build(task=task)
    texts = {item.text for item in queries}
    assert "ImplementationPlan step 2" not in texts
    assert "TASK-001 输出：pageSize 流转链路与改动边界" not in texts
    assert "接入分页控件" in texts
