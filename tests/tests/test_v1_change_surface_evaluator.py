from agent_core.benchmark.case_model import BenchmarkCase, BenchmarkExpectation
from agent_core.benchmark.change_surface_evaluator import ChangeSurfaceEvaluator
from agent_core.change_surface.model import ResolvedChangeSurface


def test_change_surface_metrics_are_change_focused():
    case = BenchmarkCase(
        schema_version=2,
        case_id="CASE-999",
        title="demo",
        category="cross_file_change",
        difficulty="medium",
        repository_name="demo",
        base_commit_sha="a" * 40,
        requirement="demo",
        expectation=BenchmarkExpectation(
            required_context_files=[
                "src/views/UsersView.vue",
                "src/stores/users.ts",
            ],
            useful_context_files=["src/api/users.ts"],
            expected_change_files=[
                "src/views/UsersView.vue",
                "src/stores/users.ts",
            ],
            allowed_change_files=["tests/unit/usersStore.test.ts"],
            forbidden_change_files=["src/api/users.ts"],
            expected_behaviors=["works"],
            verification_command_ids=["unit"],
        ),
    )
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/views/UsersView.vue",
            "src/stores/users.ts",
        ],
        dependencies=["src/api/users.ts"],
    )
    result = ChangeSurfaceEvaluator().evaluate(case=case, surface=surface)
    assert result.status == "passed"
    assert result.target_recall == 1.0
    assert result.target_precision == 1.0
    assert not result.forbidden_change_hits
