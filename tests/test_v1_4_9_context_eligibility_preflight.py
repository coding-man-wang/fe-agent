import pytest

from agent_core.benchmark.context_eligibility_preflight import (
    analyze_context_eligibility,
)
from agent_core.change_surface.model import ResolvedChangeSurface


def test_case006_has_no_source_drop_candidate():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/stores/users.ts"],
        tests=["tests/unit/usersStore.test.ts"],
        dependencies=[
            "src/types/domain.ts",
            "src/api/users.ts",
            "src/mocks/service.ts",
            "src/views/UsersView.vue",
            "src/components/UserFilters.vue",
        ],
    )

    report = analyze_context_eligibility(
        context_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
            "src/api/users.ts",
            "src/mocks/service.ts",
            "tests/unit/usersStore.test.ts",
            "tests/e2e/users.spec.ts",
        ],
        surface=surface,
    )

    assert report["status"] == "no_source_drop_candidate"
    assert report["possible_source_files"] == []
    assert report["non_source_files"] == ["tests/e2e/users.spec.ts"]
    assert report["applied"] is False


def test_unprotected_source_is_only_a_possible_candidate():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/main.ts"],
    )

    report = analyze_context_eligibility(
        context_files=["src/main.ts", "src/unused.ts"],
        surface=surface,
    )

    assert report["status"] == "candidate_possible"
    assert report["possible_source_files"] == ["src/unused.ts"]
    assert report["requires_shadow_validation"] is True


def test_obligation_evidence_is_protected():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/main.ts"],
    )

    report = analyze_context_eligibility(
        context_files=["src/main.ts", "src/evidence.ts"],
        surface=surface,
        obligation_evidence_files=["src/evidence.ts"],
    )

    assert report["status"] == "no_source_drop_candidate"


def test_unknown_file_is_inconclusive():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/main.ts"],
    )

    report = analyze_context_eligibility(
        context_files=["src/main.ts", "custom/README.md"],
        surface=surface,
    )

    assert report["status"] == "inconclusive"


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["src/main.ts", "src/main.ts"],
    ],
)
def test_missing_or_duplicate_context_cannot_pass(paths):
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/main.ts"],
    )

    report = analyze_context_eligibility(
        context_files=paths,
        surface=surface,
    )

    assert report["status"] == "inconclusive"
