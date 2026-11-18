"""D4-A: deterministic Verification Repair outcome diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_core.benchmark.execution_model import (
    VerificationExecutionAttempt,
)


def _collect_checks(
    attempt: VerificationExecutionAttempt,
) -> tuple[dict[str, str] | None, str | None]:
    if not attempt.checks:
        return None, "empty_verification_checks"

    result: dict[str, str] = {}

    for check in attempt.checks:
        command_id = check.command_id

        if not command_id:
            return None, "missing_command_id"

        if command_id in result:
            return None, "duplicate_command_id"

        result[command_id] = check.status

    return result, None


def analyze_repair_outcome(
    attempts: Sequence[VerificationExecutionAttempt],
) -> dict[str, Any]:
    """Compare one initial attempt with one bounded repair.

    Report only. Never authorizes task completion.
    """

    def insufficient(reason: str) -> dict[str, Any]:
        return {
            "version": "v1.4.9-P4-D4-A",
            "status": "insufficient_evidence",
            "reason": reason,
            "initial_failed_checks": [],
            "resolved_checks": [],
            "unresolved_checks": [],
            "unknown_checks": [],
            "regressed_checks": [],
            "unexpected_checks": [],
            "all_checks_passed": False,
            "task_completion_allowed": False,
            "advisory_only": True,
        }

    if len(attempts) < 2:
        return insufficient("missing_repair_attempt")

    # Current Runtime only supports one Verification Repair.
    if len(attempts) != 2:
        return insufficient("unexpected_attempt_count")

    initial, repaired = attempts

    if (
        initial.kind != "initial"
        or repaired.kind != "repair"
        or initial.attempt != 1
        or repaired.attempt != 2
    ):
        return insufficient("invalid_attempt_sequence")

    if initial.status != "failed":
        return insufficient("initial_attempt_not_failed")

    if initial.proposal_id == repaired.proposal_id:
        return insufficient("repair_proposal_not_distinct")

    if initial.workspace_id == repaired.workspace_id:
        return insufficient("repair_workspace_not_distinct")

    before, before_error = _collect_checks(initial)
    after, after_error = _collect_checks(repaired)

    if before_error or after_error:
        return insufficient(before_error or after_error or "invalid_checks")

    assert before is not None
    assert after is not None

    initial_failed = {
        command_id for command_id, status in before.items() if status == "failed"
    }

    if not initial_failed:
        return insufficient("no_initial_failed_checks")

    resolved = sorted(
        command_id for command_id in initial_failed if after.get(command_id) == "passed"
    )

    unresolved = sorted(
        command_id for command_id in initial_failed if after.get(command_id) == "failed"
    )

    unknown = sorted(set(before) - set(after))

    regressed = sorted(
        command_id
        for command_id, status in before.items()
        if status == "passed" and after.get(command_id) == "failed"
    )

    unexpected = sorted(set(after) - set(before))

    unexpected_failed = sorted(
        command_id for command_id in unexpected if after[command_id] == "failed"
    )

    checks_all_passed = set(before) == set(after) and all(
        value == "passed" for value in after.values()
    )

    attempt_status_consistent = repaired.status == (
        "succeeded" if checks_all_passed else "failed"
    )

    all_checks_passed = checks_all_passed and attempt_status_consistent

    if unknown or unexpected:
        status = "insufficient_evidence"
        reason = "verification_command_set_changed"
    elif not attempt_status_consistent:
        status = "insufficient_evidence"
        reason = "attempt_status_inconsistent"
    elif regressed or unexpected_failed:
        status = "regressed"
        reason = "previously_passing_check_failed"
    elif all_checks_passed:
        status = "all_resolved"
        reason = "all_checks_passed_after_repair"
    elif resolved:
        status = "partial_repair"
        reason = "some_initial_failures_remain"
    else:
        status = "unresolved"
        reason = "no_initial_failure_resolved"

    return {
        "version": "v1.4.9-P4-D4-A",
        "status": status,
        "reason": reason,
        "initial_failed_checks": sorted(initial_failed),
        "resolved_checks": resolved,
        "unresolved_checks": unresolved,
        "unknown_checks": unknown,
        "regressed_checks": regressed,
        "unexpected_checks": unexpected,
        "unexpected_failed_checks": unexpected_failed,
        "all_checks_passed": all_checks_passed,
        "task_completion_allowed": False,
        "advisory_only": True,
    }
