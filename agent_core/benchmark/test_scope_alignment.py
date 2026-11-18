"""Explain Runtime Test Scope vs frozen Benchmark Change Oracle.

Evaluation-only. Never changes permissions, source code or Oracle.
"""

from __future__ import annotations

from typing import Any

from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.execution_evaluator import (
    FullExecutionEvaluator,
)
from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
)


def analyze_test_scope_alignment(
    *,
    case: BenchmarkCase,
    run: FullExecutionRunRecord,
) -> dict[str, Any]:

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "version": "v1.5.0-P1-A",
            "status": "insufficient_evidence",
            "reason": reason,
            "benchmark_passed": False,
            "test_scope_review_required": False,
            "read_only": True,
            "applied": False,
        }

    if case.case_id != run.case_id or case.base_commit_sha != run.base_commit_sha:
        return unavailable("case_or_revision_mismatch")

    if run.status != "completed" or run.evaluation is None:
        return unavailable("execution_or_evaluation_incomplete")

    if run.hidden_acceptance is None:
        return unavailable("hidden_acceptance_missing")

    # 不能盲信保存下来的 evaluation 字段。
    # 使用当前冻结的 Case Oracle 重新计算。
    recomputed = FullExecutionEvaluator().evaluate(
        case=case,
        actual_change_files=run.evaluation.actual_change_files,
        task_records=run.tasks,
        hidden_acceptance=run.hidden_acceptance,
    )

    if recomputed.model_dump(mode="json") != run.evaluation.model_dump(mode="json"):
        return unavailable("evaluation_record_inconsistent")

    issues = []

    for path in recomputed.unexpected_change_files:
        normalized = path.replace("\\", "/")

        origins = []

        for task in run.tasks:
            if task.task_type != "test":
                continue

            proposed = {p.replace("\\", "/") for p in task.proposed_files}

            if normalized not in proposed:
                continue

            strategy = task.verification_strategy or {}
            writable = strategy.get("writable_test_files", [])

            writable_valid = isinstance(writable, list) and all(
                isinstance(p, str) for p in writable
            )

            writable_set = (
                {p.replace("\\", "/") for p in writable} if writable_valid else set()
            )

            origins.append(
                {
                    "task_id": task.task_id,
                    "task_completed": task.status == "completed",
                    "verification_succeeded": (task.verification_status == "succeeded"),
                    "runtime_writable_recorded": (normalized in writable_set),
                }
            )

        expected_test_path = normalized.startswith("tests/") and (
            ".test." in normalized or ".spec." in normalized
        )

        traceable = (
            expected_test_path
            and bool(origins)
            and all(
                origin["task_completed"]
                and origin["verification_succeeded"]
                and origin["runtime_writable_recorded"]
                for origin in origins
            )
        )

        issues.append(
            {
                "path": normalized,
                "classification": (
                    "runtime_test_scope_recorded"
                    if traceable
                    else "unattributed_or_unverified_change"
                ),
                "origins": origins,
                "benchmark_allowed": False,
                "file_creation_verified": False,
                "requires_review": True,
            }
        )

    other_failures = [
        reason for reason in recomputed.failure_reasons if reason != "unexpected_change"
    ]

    if any(
        issue["classification"] == "unattributed_or_unverified_change"
        for issue in issues
    ):
        status = "unattributed_change"
    elif issues and other_failures:
        status = "multiple_evaluation_failures"
    elif issues:
        status = "test_scope_contract_mismatch"
    elif recomputed.status == "passed":
        status = "aligned"
    else:
        status = "other_evaluation_failure"

    return {
        "version": "v1.5.0-P1-A",
        "case_id": case.case_id,
        "run_id": run.run_id,
        "base_commit_sha": run.base_commit_sha,
        "status": status,
        "execution_status": run.status,
        "evaluation_status": recomputed.status,
        "hidden_acceptance_status": run.hidden_acceptance.status,
        "unexpected_change_files": (recomputed.unexpected_change_files),
        "forbidden_change_hits": (recomputed.forbidden_change_hits),
        "other_failure_reasons": other_failures,
        "issues": issues,
        "benchmark_passed": recomputed.status == "passed",
        "test_scope_review_required": bool(issues),
        "review_options": (
            [
                "review_test_file_necessity",
                "review_frozen_benchmark_contract",
                "retain_original_failed_result",
            ]
            if issues
            else []
        ),
        "oracle_visibility": "benchmark_evaluation_only",
        "read_only": True,
        "applied": False,
        "production_ready": False,
    }
