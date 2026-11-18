"""D4-B: verify coverage of the Repair model input."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)


def inspect_repair_input_coverage(
    *,
    verification: VerificationRunRecord,
    payload: Mapping[str, Any],
    truncate_details: Callable[[str], str],
) -> dict[str, Any]:
    """Validate all failed checks before an LLM call.

    No model calls. Never log raw verification details.
    """

    def result(
        issues: list[str],
        expected_ids: list[str],
        provided_ids: list[str],
        truncated_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "version": "v1.4.9-P4-D4-B",
            "status": ("complete" if not issues else "invalid"),
            "complete": not issues,
            "issues": sorted(set(issues)),
            "expected_failed_command_ids": sorted(expected_ids),
            "provided_failed_command_ids": sorted(provided_ids),
            "truncated_detail_command_ids": sorted(truncated_ids),
            "advisory_only": True,
        }

    if verification.status != "failed" or verification.result is None:
        return result(["invalid_verification_record"], [], [], [])

    expected_checks = [
        check for check in verification.result.checks if check.status == "failed"
    ]

    if not expected_checks:
        return result(["no_failed_checks"], [], [], [])

    expected_ids = [check.command_id or "" for check in expected_checks]

    issues: list[str] = []

    if any(not item for item in expected_ids):
        issues.append("source_missing_command_id")

    if len(expected_ids) != len(set(expected_ids)):
        issues.append("duplicate_source_command_id")

    provided = payload.get("verification_failures")

    if not isinstance(provided, list):
        return result(
            issues + ["missing_verification_failures"],
            expected_ids,
            [],
            [],
        )

    provided_by_id: dict[str, dict] = {}
    provided_ids: list[str] = []

    for item in provided:
        if not isinstance(item, dict):
            issues.append("invalid_payload_check")
            continue

        command_id = item.get("command_id")

        if not isinstance(command_id, str) or not command_id:
            issues.append("payload_missing_command_id")
            continue

        provided_ids.append(command_id)

        if command_id in provided_by_id:
            issues.append("duplicate_payload_command_id")

        provided_by_id[command_id] = item

    expected_set = set(expected_ids)
    provided_set = set(provided_ids)

    if expected_set - provided_set:
        issues.append("failed_checks_missing_from_payload")

    if provided_set - expected_set:
        issues.append("unexpected_payload_checks")

    truncated_ids: list[str] = []

    for check in expected_checks:
        command_id = check.command_id

        if not command_id:
            continue

        supplied = provided_by_id.get(command_id)

        if supplied is None:
            continue

        expected_details = truncate_details(check.details)

        if expected_details != check.details:
            truncated_ids.append(command_id)

        if (
            supplied.get("name") != check.name
            or supplied.get("status") != "failed"
            or supplied.get("details") != expected_details
        ):
            issues.append("verification_evidence_mismatch:" + command_id)

    return result(
        issues,
        expected_ids,
        provided_ids,
        truncated_ids,
    )
