from agent_core.benchmark.impact_verification import (
    ImpactAwareVerificationPlanner,
)
from agent_core.codebase.impact import (
    ChangeImpactReport,
)


def _impact(
    *,
    tests: list[str],
    level: str = "medium",
) -> ChangeImpactReport:
    return ChangeImpactReport(
        repository_revision="commit-a",
        changed_files=["src/example.ts"],
        affected_production_files=[],
        affected_test_files=[],
        suggested_test_files=tests,
        unresolved_changed_files=[],
        impact_level=level,
        reasons=[],
    )


def test_existing_unit_command_is_not_duplicated():
    result = ImpactAwareVerificationPlanner().build(
        base_command_ids=[
            "frontend-lint",
            "frontend-unit-test",
        ],
        impact=_impact(tests=["tests/unit/foo.test.ts"]),
    )

    assert result.final_command_ids == [
        "frontend-lint",
        "frontend-unit-test",
    ]

    assert result.added_command_ids == []


def test_e2e_impact_adds_e2e_command():
    result = ImpactAwareVerificationPlanner().build(
        base_command_ids=[
            "frontend-lint",
            "frontend-unit-test",
        ],
        impact=_impact(tests=["tests/e2e/users.spec.ts"]),
    )

    assert "frontend-e2e" in result.final_command_ids

    assert result.added_command_ids == ["frontend-e2e"]


def test_unknown_impact_adds_e2e_conservatively():
    result = ImpactAwareVerificationPlanner().build(
        base_command_ids=[
            "frontend-lint",
        ],
        impact=_impact(
            tests=[],
            level="unknown",
        ),
    )

    assert "frontend-e2e" in result.final_command_ids
