from agent_core.benchmark.execution_model import (
    FullExecutionTaskRecord,
)
from agent_core.benchmark.final_verification import (
    FinalVerificationPlanner,
)
from agent_core.benchmark.impact_verification import (
    ImpactVerificationPlan,
)


def _task(
    *,
    task_id: str,
    added: list[str],
) -> FullExecutionTaskRecord:

    return FullExecutionTaskRecord(
        task_id=task_id,
        task_type="code_change",
        status="completed",
        verification_status="succeeded",
        planned_verification=(
            ImpactVerificationPlan(
                base_command_ids=[
                    "frontend-lint",
                    "frontend-typecheck",
                    "frontend-unit-test",
                    "frontend-build",
                ],
                added_command_ids=added,
                final_command_ids=[
                    "frontend-lint",
                    "frontend-typecheck",
                    "frontend-unit-test",
                    "frontend-build",
                    *added,
                ],
                reasons=[],
            )
        ),
    )


def test_final_verification_keeps_base_commands():
    planner = FinalVerificationPlanner()

    plan = planner.build(
        base_command_ids=[
            "frontend-lint",
            "frontend-typecheck",
            "frontend-unit-test",
            "frontend-build",
        ],
        task_records=[],
    )

    assert plan.base_command_ids == [
        "frontend-lint",
        "frontend-typecheck",
        "frontend-unit-test",
        "frontend-build",
    ]

    assert plan.added_command_ids == []

    assert plan.final_command_ids == plan.base_command_ids


def test_final_verification_adds_impact_commands():
    planner = FinalVerificationPlanner()

    plan = planner.build(
        base_command_ids=[
            "frontend-lint",
            "frontend-typecheck",
            "frontend-unit-test",
            "frontend-build",
        ],
        task_records=[
            _task(
                task_id="TASK-003",
                added=["frontend-e2e"],
            )
        ],
    )

    assert plan.added_command_ids == ["frontend-e2e"]

    assert plan.final_command_ids == [
        "frontend-lint",
        "frontend-typecheck",
        "frontend-unit-test",
        "frontend-build",
        "frontend-e2e",
    ]


def test_final_verification_dedupes_added_commands():
    planner = FinalVerificationPlanner()

    plan = planner.build(
        base_command_ids=[
            "frontend-unit-test",
        ],
        task_records=[
            _task(
                task_id="TASK-002",
                added=["frontend-e2e"],
            ),
            _task(
                task_id="TASK-003",
                added=["frontend-e2e"],
            ),
        ],
    )

    assert plan.added_command_ids == ["frontend-e2e"]

    assert plan.final_command_ids == [
        "frontend-unit-test",
        "frontend-e2e",
    ]


def test_final_verification_does_not_duplicate_base_command():
    planner = FinalVerificationPlanner()

    plan = planner.build(
        base_command_ids=[
            "frontend-unit-test",
        ],
        task_records=[
            _task(
                task_id="TASK-002",
                added=["frontend-unit-test"],
            )
        ],
    )

    assert plan.added_command_ids == []

    assert plan.final_command_ids == ["frontend-unit-test"]
