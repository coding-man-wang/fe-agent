from agent_core.change_surface import PatchScopeGuard, ResolvedChangeSurface
from agent_core.coding.patch_model import PatchPlan, PatchTarget


SHA = "a" * 64


def test_scope_guard_allows_target_and_blocks_dependency():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/views/UsersView.vue"],
        dependencies=["src/api/users.ts"],
    )
    guard = PatchScopeGuard()

    good = PatchPlan(
        task_id="TASK-002",
        artifact_id="a1",
        context_hash="c1",
        targets=[
            PatchTarget(
                relative_path="src/views/UsersView.vue",
                expected_sha256=SHA,
                objective="add page size selector",
                proposed_approach="reuse store",
            )
        ],
    )
    assert guard.evaluate_plan(plan=good, surface=surface).allowed

    bad = good.model_copy(
        update={
            "targets": [
                PatchTarget(
                    relative_path="src/api/users.ts",
                    expected_sha256=SHA,
                    objective="change api",
                    proposed_approach="not allowed",
                )
            ]
        }
    )
    result = guard.evaluate_plan(plan=bad, surface=surface)
    assert not result.allowed
    assert "src/api/users.ts" in result.violations[0]
