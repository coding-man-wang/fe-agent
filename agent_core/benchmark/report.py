from agent_core.benchmark.result_model import (
    BenchmarkCaseResult,
)


class BenchmarkTextReportRenderer:

    def render(
        self,
        result: BenchmarkCaseResult,
    ) -> str:

        lines: list[str] = []

        lines.append("=" * 60)

        lines.append(f"Benchmark Case: " f"{result.case_id}")

        lines.append("=" * 60)

        lines.append(f"Run ID: {result.run_id}")

        lines.append(f"Run Status: " f"{result.run_status}")

        lines.append(f"Final Stage: " f"{result.final_stage}")

        lines.append(f"Base Commit: " f"{result.base_commit_sha}")

        # ==================================
        # Planning
        # ==================================

        lines.append("")
        lines.append("[Planning]")

        if result.planning is None:

            lines.append("Not available")

        else:

            planning = result.planning

            lines.append(f"Task Count: " f"{planning.task_count}")

            lines.append(
                "Task Types: "
                f"investigation="
                f"{planning.investigation_tasks}, "
                f"code_change="
                f"{planning.code_change_tasks}, "
                f"test="
                f"{planning.test_tasks}, "
                f"verification="
                f"{planning.verification_tasks}"
            )

        # ==================================
        # Retrieval
        # ==================================

        lines.append("")
        lines.append("[Retrieval]")

        retrieval = result.retrieval

        lines.append(f"Status: " f"{retrieval.status}")

        if retrieval.recall is not None:

            lines.append("Recall: " f"{retrieval.recall:.2%}")

        if retrieval.precision is not None:

            lines.append("Precision: " f"{retrieval.precision:.2%}")

        self._append_list(
            lines,
            "Expected Files",
            retrieval.expected_files,
        )

        self._append_list(
            lines,
            "Matched Files",
            retrieval.matched_files,
        )

        self._append_list(
            lines,
            "Missed Files",
            retrieval.missed_files,
        )

        self._append_list(
            lines,
            "Unexpected Files",
            retrieval.unexpected_files,
        )

        self._append_list(
            lines,
            "Forbidden Hits",
            retrieval.forbidden_hits,
        )

        self._append_list(
            lines,
            "Failure Reasons",
            retrieval.failure_reasons,
        )

        # ==================================
        # Harness Error
        # ==================================

        if result.execution_error:

            lines.append("")
            lines.append("[Execution Error]")

            lines.append(result.execution_error)

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def _append_list(
        lines: list[str],
        title: str,
        values: list[str],
    ) -> None:

        lines.append(f"{title}:")

        if not values:

            lines.append("  - none")

            return

        for value in values:

            lines.append(f"  - {value}")
