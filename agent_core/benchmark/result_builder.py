from agent_core.benchmark.result_model import (
    BenchmarkCaseResult,
    BenchmarkPlanningSummary,
    BenchmarkRetrievalSummary,
)
from agent_core.benchmark.run_model import (
    BenchmarkRunRecord,
)


class BenchmarkResultBuilder:

    def build(
        self,
        *,
        run: BenchmarkRunRecord,
    ) -> BenchmarkCaseResult:

        planning = self._build_planning_summary(run)

        retrieval = self._build_retrieval_summary(run)

        return BenchmarkCaseResult(
            run_id=run.run_id,
            case_id=run.case_id,
            base_commit_sha=(run.base_commit_sha),
            run_status=run.status,
            final_stage=run.stage,
            planning=planning,
            retrieval=retrieval,
            execution_error=run.error,
        )

    def _build_planning_summary(
        self,
        run: BenchmarkRunRecord,
    ) -> BenchmarkPlanningSummary | None:

        if run.task_plan is None:
            return None

        task_types = [task.task_type for task in run.task_plan.tasks]

        return BenchmarkPlanningSummary(
            task_count=len(task_types),
            investigation_tasks=(task_types.count("investigation")),
            code_change_tasks=(task_types.count("code_change")),
            test_tasks=(task_types.count("test")),
            verification_tasks=(task_types.count("verification")),
        )

    def _build_retrieval_summary(
        self,
        run: BenchmarkRunRecord,
    ) -> BenchmarkRetrievalSummary:

        evaluation = run.retrieval_evaluation

        if evaluation is None:

            return BenchmarkRetrievalSummary(
                status="not_evaluated",
            )

        return BenchmarkRetrievalSummary(
            status=(evaluation.status),
            recall=(evaluation.recall),
            precision=(evaluation.precision),
            expected_files=(list(evaluation.expected_files)),
            retrieved_files=(list(evaluation.retrieved_files)),
            matched_files=(list(evaluation.matched_expected_files)),
            missed_files=(list(evaluation.missed_expected_files)),
            unexpected_files=(list(evaluation.unexpected_files)),
            forbidden_hits=(list(evaluation.forbidden_hits)),
            failure_reasons=(list(evaluation.failure_reasons)),
        )
