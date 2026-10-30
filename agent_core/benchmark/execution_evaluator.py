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
        impact_oracle = case.impact_oracle()
        expected = sorted(set(oracle.expected_change_files))
        # =====================================
        # v1.2.4
        #
        # 汇总整个 Full Execution Run 中
        # 所有 Task 发现的 Impact。
        #
        # Case Oracle 关注的是：
        #
        # “整个需求执行过程中有没有发现”
        #
        # 而不是要求某个固定 TASK-ID
        # 必须发现某个文件。
        #
        # 因为 Task Plan 本身可能变化。
        # =====================================

        observed_impact_set: set[str] = set()

        observed_impact_verification_set: set[str] = set()

        for task_record in task_records:

            impact = task_record.planned_impact

            if impact is not None:
                observed_impact_set.update(
                    item.relative_path for item in (impact.affected_production_files)
                )

                observed_impact_set.update(
                    item.relative_path for item in (impact.affected_test_files)
                )

                observed_impact_set.update(impact.suggested_test_files)

            # =====================================
            # Impact Verification Coverage
            #
            # 这里验证的是：
            #
            # Impact Analysis 是否让
            # Verification Planner
            # 安排了 Oracle 要求的验证。
            #
            # 所以 Source of Truth 是：
            #
            # planned_verification.final_command_ids
            #
            # 而不是当前生命周期阶段
            # 实际执行了哪些命令。
            #
            # planned != scheduled != executed
            # =====================================

            planned_verification = task_record.planned_verification

            if planned_verification is not None:
                (
                    observed_impact_verification_set.update(
                        planned_verification.final_command_ids
                    )
                )

        allowed = sorted(set(oracle.allowed_change_files))
        forbidden = sorted(set(oracle.forbidden_change_files))
        actual = sorted(set(actual_change_files))

        matched = [
            path
            for path in expected
            if any(_matches_scope(item, path) for item in actual)
        ]
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
        completed = [record for record in task_records if record.status == "completed"]

        verified_tasks = [
            record
            for record in completed
            if record.task_type
            in {
                "code_change",
                "test",
                "verification",
            }
        ]

        verification_passed = bool(verified_tasks) and all(
            record.verification_status == "succeeded" for record in verified_tasks
        )

        hidden_passed = (
            hidden_acceptance is not None and hidden_acceptance.status == "passed"
        )
        expected_impacts = sorted(set(impact_oracle.required_impact_files))

        observed_impacts = sorted(observed_impact_set)

        missed_impacts = [
            path
            for path in expected_impacts
            if not _covered(
                path,
                observed_impacts,
            )
        ]

        required_impact_commands = sorted(
            set(impact_oracle.required_verification_command_ids)
        )

        observed_verification_commands = sorted(observed_impact_verification_set)

        missing_impact_commands = [
            command_id
            for command_id in required_impact_commands
            if command_id not in (observed_impact_verification_set)
        ]

        impact_coverage_passed = not missed_impacts

        impact_verification_passed = not missing_impact_commands

        reasons: list[str] = []
        if missed:
            reasons.append("missing_expected_change")
        if expected_impacts and missed_impacts:
            reasons.append("missing_required_impact")
        if required_impact_commands and missing_impact_commands:
            reasons.append("missing_impact_verification")
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
            completed_task_count=len(completed),
            failed_task_ids=failed_task_ids,
            verification_passed=verification_passed,
            hidden_acceptance_passed=hidden_passed,
            failure_reasons=reasons,
            expected_impact_files=(expected_impacts),
            observed_impact_files=(observed_impacts),
            missed_impact_files=(missed_impacts),
            required_impact_verification_command_ids=(required_impact_commands),
            observed_verification_command_ids=(observed_verification_commands),
            missing_impact_verification_command_ids=(missing_impact_commands),
            impact_coverage_passed=(impact_coverage_passed),
            impact_verification_passed=(impact_verification_passed),
        )
