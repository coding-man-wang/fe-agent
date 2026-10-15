from agent_core.benchmark.suite_model import (
    BenchmarkSuiteResult,
)


class BenchmarkSuiteTextReportRenderer:

    def render(
        self,
        suite: BenchmarkSuiteResult,
    ) -> str:

        lines: list[str] = []

        lines.append("=" * 88)

        lines.append("Benchmark Suite")

        lines.append("=" * 88)

        lines.append(f"Suite ID: " f"{suite.suite_id}")

        lines.append(f"Cases: " f"{suite.summary.total_cases}")

        lines.append("")

        # -----------------------------------
        # Header
        # -----------------------------------

        lines.append(
            self._row(
                "Case",
                "Run",
                "Tasks",
                "Retrieval",
                "Recall",
                "Precision",
            )
        )

        lines.append("-" * 88)

        # -----------------------------------
        # Case Rows
        # -----------------------------------

        for execution in suite.cases:

            if execution.status == "error" or execution.result is None:

                lines.append(
                    self._row(
                        execution.case_id,
                        "ERROR",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
                )

                continue

            result = execution.result

            planning = result.planning

            retrieval = result.retrieval

            task_count = str(planning.task_count) if planning is not None else "-"

            recall = self._format_percent(retrieval.recall)

            precision = self._format_percent(retrieval.precision)

            lines.append(
                self._row(
                    result.case_id,
                    result.run_status,
                    task_count,
                    retrieval.status,
                    recall,
                    precision,
                )
            )

        # -----------------------------------
        # Summary
        # -----------------------------------

        lines.append("")
        lines.append("[Summary]")

        summary = suite.summary

        lines.append(f"Total Cases: " f"{summary.total_cases}")

        lines.append(f"Completed Cases: " f"{summary.completed_cases}")

        lines.append(f"Harness Errors: " f"{summary.harness_error_cases}")

        lines.append("Retrieval Passed: " f"{summary.retrieval_passed}")

        lines.append("Retrieval Partial: " f"{summary.retrieval_partial}")

        lines.append("Retrieval Failed: " f"{summary.retrieval_failed}")

        lines.append("Retrieval Not Evaluated: " f"{summary.retrieval_not_evaluated}")

        lines.append("Average Recall: " + self._format_percent(summary.average_recall))

        lines.append(
            "Average Precision: " + self._format_percent(summary.average_precision)
        )

        # -----------------------------------
        # Harness Errors
        # -----------------------------------

        errors = [execution for execution in suite.cases if execution.status == "error"]

        if errors:

            lines.append("")
            lines.append("[Harness Errors]")

            for execution in errors:

                lines.append(f"{execution.case_id}: " f"{execution.error}")

        lines.append("")
        lines.append("=" * 88)

        return "\n".join(lines)

    @staticmethod
    def _format_percent(
        value: float | None,
    ) -> str:

        if value is None:
            return "-"

        return f"{value:.2%}"

    @staticmethod
    def _row(
        case: str,
        run: str,
        tasks: str,
        retrieval: str,
        recall: str,
        precision: str,
    ) -> str:

        return (
            f"{case:<12}"
            f"{run:<14}"
            f"{tasks:<8}"
            f"{retrieval:<16}"
            f"{recall:<12}"
            f"{precision:<12}"
        )
