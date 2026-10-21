from agent_core.benchmark.execution_surface import ProductionExecutionSurfacePolicy
from agent_core.change_surface.model import ResolvedChangeSurface


def test_production_execution_surface_makes_tests_read_only() -> None:
    source = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        dependencies=["src/api/users.ts"],
        references=["src/components/UserTable.vue"],
        tests=[
            "tests/e2e/users.spec.ts",
            "tests/unit/usersStore.test.ts",
        ],
        execution_disposition="change_required",
    )

    result = ProductionExecutionSurfacePolicy().apply(source)

    assert result.implementation_targets == [
        "src/stores/users.ts",
        "src/views/UsersView.vue",
    ]
    assert result.tests == []
    assert result.references == [
        "src/components/UserTable.vue",
        "tests/e2e/users.spec.ts",
        "tests/unit/usersStore.test.ts",
    ]
    assert result.allowed_change_files() == {
        "src/stores/users.ts",
        "src/views/UsersView.vue",
    }

    # The repository-aware planning artifact remains unchanged; this is an
    # execution policy projection, not a mutation of planning evidence.
    assert source.tests == [
        "tests/e2e/users.spec.ts",
        "tests/unit/usersStore.test.ts",
    ]


def test_production_execution_surface_can_be_noop_after_test_demotion() -> None:
    source = ResolvedChangeSurface(
        task_id="TASK-TEST-ONLY",
        implementation_targets=[],
        tests=["tests/unit/usersStore.test.ts"],
        execution_disposition="change_required",
    )

    result = ProductionExecutionSurfacePolicy().apply(source)

    assert result.tests == []
    assert result.references == ["tests/unit/usersStore.test.ts"]
    assert result.requires_patch() is False
