import pytest

from agent_core.benchmark.execution_model import (
    VerificationExecutionAttempt,
)
from agent_core.benchmark.verification_repair_diagnostics import (
    analyze_repair_outcome,
)
from agent_core.coding.verification_model import (
    VerificationCheck,
)


def make_attempt(
    number: int,
    kind: str,
    status: str,
    checks: dict[str, str],
) -> VerificationExecutionAttempt:
    return VerificationExecutionAttempt(
        attempt=number,
        kind=kind,
        status=status,
        proposal_id=str(number) * 64,
        workspace_id=f"workspace-{number}",
        checks=[
            VerificationCheck(
                name=command_id,
                command_id=command_id,
                status=check_status,
            )
            for command_id, check_status in checks.items()
        ],
    )


def initial():
    return make_attempt(
        1,
        "initial",
        "failed",
        {
            "frontend-lint": "failed",
            "frontend-typecheck": "passed",
            "frontend-unit-test": "passed",
            "frontend-e2e": "failed",
        },
    )


def repaired(
    *,
    lint="passed",
    typecheck="passed",
    unit="passed",
    e2e="passed",
):
    checks = {
        "frontend-lint": lint,
        "frontend-typecheck": typecheck,
        "frontend-unit-test": unit,
        "frontend-e2e": e2e,
    }

    status = "succeeded" if all(s == "passed" for s in checks.values()) else "failed"

    return make_attempt(2, "repair", status, checks)


def test_case006_partial_repair():
    report = analyze_repair_outcome(
        [
            initial(),
            repaired(e2e="failed"),
        ]
    )

    assert report["status"] == "partial_repair"
    assert report["initial_failed_checks"] == [
        "frontend-e2e",
        "frontend-lint",
    ]
    assert report["resolved_checks"] == ["frontend-lint"]
    assert report["unresolved_checks"] == ["frontend-e2e"]
    assert report["all_checks_passed"] is False
    assert report["task_completion_allowed"] is False


def test_all_failures_resolved():
    report = analyze_repair_outcome(
        [
            initial(),
            repaired(),
        ]
    )

    assert report["status"] == "all_resolved"
    assert report["all_checks_passed"] is True
    assert report["task_completion_allowed"] is False


def test_no_failure_resolved():
    report = analyze_repair_outcome(
        [
            initial(),
            repaired(lint="failed", e2e="failed"),
        ]
    )

    assert report["status"] == "unresolved"


def test_repair_introduces_regression():
    report = analyze_repair_outcome(
        [
            initial(),
            repaired(
                typecheck="failed",
                e2e="passed",
            ),
        ]
    )

    assert report["status"] == "regressed"
    assert report["regressed_checks"] == ["frontend-typecheck"]


def test_missing_verification_check_is_not_success():
    incomplete = make_attempt(
        2,
        "repair",
        "failed",
        {
            "frontend-lint": "passed",
            "frontend-typecheck": "passed",
            "frontend-unit-test": "passed",
        },
    )

    report = analyze_repair_outcome(
        [
            initial(),
            incomplete,
        ]
    )

    assert report["status"] == "insufficient_evidence"
    assert report["unknown_checks"] == ["frontend-e2e"]
    assert report["all_checks_passed"] is False


def test_unexpected_check_requires_review():
    extra = make_attempt(
        2,
        "repair",
        "failed",
        {
            "frontend-lint": "passed",
            "frontend-typecheck": "passed",
            "frontend-unit-test": "passed",
            "frontend-e2e": "passed",
            "frontend-build": "failed",
        },
    )

    report = analyze_repair_outcome(
        [
            initial(),
            extra,
        ]
    )

    assert report["status"] == "insufficient_evidence"
    assert report["unexpected_checks"] == ["frontend-build"]


def test_no_repair_attempt():
    report = analyze_repair_outcome([initial()])

    assert report["status"] == "insufficient_evidence"
    assert report["reason"] == "missing_repair_attempt"


def test_duplicate_command_id():
    attempt = repaired()
    attempt.checks.append(
        VerificationCheck(
            name="duplicate",
            command_id="frontend-e2e",
            status="passed",
        )
    )

    report = analyze_repair_outcome(
        [
            initial(),
            attempt,
        ]
    )

    assert report["status"] == "insufficient_evidence"
    assert report["reason"] == "duplicate_command_id"


def test_repair_must_use_distinct_workspace():
    attempt = repaired()
    attempt.workspace_id = initial().workspace_id

    report = analyze_repair_outcome(
        [
            initial(),
            attempt,
        ]
    )

    assert report["status"] == "insufficient_evidence"
    assert report["reason"] == "repair_workspace_not_distinct"


from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
    FullExecutionTaskRecord,
)
from apps.agent_cli.verification_repair_diagnostics import (
    inspect_saved_repair,
)


def test_saved_run_repair_diagnostics_is_read_only(tmp_path):
    initial_attempt = initial()
    repaired_attempt = repaired(e2e="failed")

    record = FullExecutionRunRecord(
        run_id="a" * 32,
        baseline_run_id="historical-input",
        case_id="CASE-006",
        base_commit_sha="b" * 40,
        status="failed",
        tasks=[
            FullExecutionTaskRecord(
                task_id="TASK-005",
                task_type="test",
                status="failed",
                verification_status="failed",
                verification_repair_count=1,
                verification_attempts=[
                    initial_attempt,
                    repaired_attempt,
                ],
                verification_checks=list(repaired_attempt.checks),
            )
        ],
    )

    directory = tmp_path / ".fe-agent" / "benchmark" / "full_execution" / "runs"
    directory.mkdir(parents=True)

    path = directory / f"{record.run_id}.json"
    original = record.model_dump_json(indent=2)

    path.write_text(original, encoding="utf-8")

    report = inspect_saved_repair(
        run_id=record.run_id,
        task_id="TASK-005",
        fe_agent_root=tmp_path,
    )

    assert report["case_id"] == "CASE-006"
    assert report["outcome"]["status"] == "partial_repair"
    assert report["outcome"]["resolved_checks"] == ["frontend-lint"]
    assert report["outcome"]["unresolved_checks"] == ["frontend-e2e"]
    assert report["historical_input_coverage"] == ("not_persisted")
    assert report["task_completion_allowed"] is False

    # 不修改历史 Run。
    assert path.read_text(encoding="utf-8") == original
