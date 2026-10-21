from __future__ import annotations

from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.execution_model import (
    FullExecutionEvaluation,
    FullExecutionTaskRecord,
)
from agent_core.coding.verification_model import PatchVerificationResult


def _matches_scope(path: str, scope: str) -> bool:
    normalized_path = path.replace("\\", "/").strip("/")
    normalized_scope = scope.replace("\\", "/").strip("/")
    return normalized_path == normalized_scope or normalized_path.startswith(
        normalized_scope + "/"
    )


def _covered(path: str, scopes: list[str]) -> bool:
    return any(_matches_scope(path, scope) for scope in scopes)


class FullExecutionEvaluator:
    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        actual_change_files: list[str],
        task_records: list[FullExecutionTaskRecord],
        hidden_acceptance: PatchVerificationResult | None,
    ) -> FullExecutionEvaluation:
        oracle = case.change_oracle()
        expected = sorted(set(oracle.expected_change_files))
        allowed = sorted(set(oracle.allowed_change_files))
        forbidden = sorted(set(oracle.forbidden_change_files))
        actual = sorted(set(actual_change_files))

        matched = [path for path in expected if any(_matches_scope(item, path) for item in actual)]
        missed = [path for path in expected if path not in matched]
        unexpected = [
            path
            for path in actual
            if not _covered(path, expected) and not _covered(path, allowed)
        ]
        forbidden_hits = [path for path in actual if _covered(path, forbidden)]

        failed_task_ids = sorted(
            record.task_id for record in task_records if record.status == "failed"
        )
        executed = [record for record in task_records if record.status == "completed"]
        verification_passed = bool(executed) and all(
            record.verification_status == "succeeded" for record in executed
        )
        hidden_passed = (
            hidden_acceptance is not None and hidden_acceptance.status == "passed"
        )

        reasons: list[str] = []
        if missed:
            reasons.append("missing_expected_change")
        if unexpected:
            reasons.append("unexpected_change")
        if forbidden_hits:
            reasons.append("forbidden_change")
        if failed_task_ids:
            reasons.append("task_execution_failed")
        if not verification_passed:
            reasons.append("verification_failed")
        if not hidden_passed:
            reasons.append("hidden_acceptance_failed")

        return FullExecutionEvaluation(
            status="passed" if not reasons else "failed",
            expected_change_files=expected,
            allowed_change_files=allowed,
            actual_change_files=actual,
            matched_change_files=matched,
            missed_change_files=missed,
            unexpected_change_files=unexpected,
            forbidden_change_hits=forbidden_hits,
            completed_task_count=len(executed),
            failed_task_ids=failed_task_ids,
            verification_passed=verification_passed,
            hidden_acceptance_passed=hidden_passed,
            failure_reasons=reasons,
        )
