from agent_core.artifacts import JsonTaskArtifactStore, TaskArtifactResolver
from agent_core.change_surface import RepositoryAwareReplanner, ResolvedChangeSurface
from agent_core.sdd.model import TaskDefinition, TaskPlan


def test_replanner_refines_module_hypothesis_to_files(tmp_path):
    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")
    store.put_model(
        task_id="TASK-002",
        artifact_type="change_surface",
        model=ResolvedChangeSurface(
            task_id="TASK-002",
            implementation_targets=[
                "src/views/UsersView.vue",
                "src/stores/users.ts",
            ],
            dependencies=["src/api/users.ts"],
        ),
    )
    plan = TaskPlan(
        summary="demo",
        tasks=[
            TaskDefinition(
                task_id="TASK-002",
                task_type="code_change",
                title="change",
                objective="change",
                source_step_order=2,
                target_modules=["src/views", "src/stores"],
            )
        ],
    )
    replanner = RepositoryAwareReplanner(
        artifact_resolver=TaskArtifactResolver(store=store)
    )
    result = replanner.build(task_plan=plan)
    assert result.tasks[0].resolved_target_files == [
        "src/views/UsersView.vue",
        "src/stores/users.ts",
    ]
