from agent_core.artifacts.model import (
    ArtifactGraph,
    ArtifactGraphEdge,
    ArtifactGraphNode,
)
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.sdd.model import TaskPlan


class ArtifactDagBuilder:
    def build(
        self,
        *,
        task_plan: TaskPlan,
        store: JsonTaskArtifactStore,
    ) -> ArtifactGraph:
        nodes: list[ArtifactGraphNode] = []
        edges: list[ArtifactGraphEdge] = []
        task_ids = {task.task_id for task in task_plan.tasks}

        for task in task_plan.tasks:
            nodes.append(
                ArtifactGraphNode(
                    node_id=f"task:{task.task_id}",
                    node_type="task",
                    task_id=task.task_id,
                )
            )
            for upstream in task.depends_on:
                if upstream in task_ids:
                    edges.append(
                        ArtifactGraphEdge(
                            source=f"task:{upstream}",
                            target=f"task:{task.task_id}",
                            relation="depends_on",
                        )
                    )

        for artifact in store.list():
            artifact_node = f"artifact:{artifact.artifact_id}"
            nodes.append(
                ArtifactGraphNode(
                    node_id=artifact_node,
                    node_type="artifact",
                    task_id=artifact.task_id,
                    artifact_type=artifact.artifact_type,
                )
            )
            edges.append(
                ArtifactGraphEdge(
                    source=f"task:{artifact.task_id}",
                    target=artifact_node,
                    relation="produces",
                )
            )
            for downstream in task_plan.tasks:
                if artifact.task_id in downstream.depends_on:
                    edges.append(
                        ArtifactGraphEdge(
                            source=artifact_node,
                            target=f"task:{downstream.task_id}",
                            relation="consumes",
                        )
                    )
        return ArtifactGraph(nodes=nodes, edges=edges)
