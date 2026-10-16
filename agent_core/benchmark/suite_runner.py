import asyncio
from typing import Protocol
from uuid import uuid4

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.result_builder import (
    BenchmarkResultBuilder,
)
from agent_core.benchmark.result_model import (
    BenchmarkCaseResult,
)
from agent_core.benchmark.result_store import (
    BenchmarkResultStore,
)
from agent_core.benchmark.run_model import (
    BenchmarkRunRecord,
    utc_now,
)
from agent_core.benchmark.suite_model import (
    BenchmarkSuiteCaseExecution,
    BenchmarkSuiteResult,
    BenchmarkSuiteSummary,
)


class BenchmarkRunnerProtocol(Protocol):

    async def run(
        self,
        *,
        case_id: str,
    ) -> BenchmarkRunRecord: ...


class BenchmarkSuiteRunner:

    def __init__(
        self,
        *,
        case_loader: BenchmarkCaseLoader,
        benchmark_runner: BenchmarkRunnerProtocol,
        result_builder: BenchmarkResultBuilder,
        result_store: BenchmarkResultStore,
        case_timeout_seconds: float = 300.0,
    ) -> None:

        self.case_loader = case_loader

        self.benchmark_runner = benchmark_runner

        self.result_builder = result_builder

        self.result_store = result_store

        if case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds 必须大于 0")

        self.case_timeout_seconds = case_timeout_seconds

    async def run_all(
        self,
    ) -> BenchmarkSuiteResult:

        case_ids = self.case_loader.list_case_ids()

        return await self.run_cases(case_ids=case_ids)

    async def run_cases(
        self,
        *,
        case_ids: list[str],
    ) -> BenchmarkSuiteResult:

        suite_id = uuid4().hex

        started_at = utc_now()

        executions: list[BenchmarkSuiteCaseExecution] = []

        # -----------------------------------
        # v0.1 故意串行
        #
        # 原因：
        # - LLM Provider 限流
        # - Benchmark 更容易复现
        # - 日志更容易分析
        #
        # 后续再做并行 Suite。
        # -----------------------------------

        for case_id in case_ids:
            print(
                f"[benchmark] " f"{case_id} started",
                flush=True,
            )

            execution = await self._run_case(case_id=case_id)

            print(
                f"[benchmark] " f"{case_id} " f"{execution.status}",
                flush=True,
            )

            executions.append(execution)

        finished_at = utc_now()

        summary = self._build_summary(executions)

        return BenchmarkSuiteResult(
            suite_id=suite_id,
            started_at=started_at,
            finished_at=finished_at,
            cases=executions,
            summary=summary,
        )

    async def _run_case(
        self,
        *,
        case_id: str,
    ) -> BenchmarkSuiteCaseExecution:

        try:

            async with asyncio.timeout(self.case_timeout_seconds):
                run = await self.benchmark_runner.run(case_id=case_id)

            result = self.result_builder.build(run=run)

            self.result_store.save(result)

            return BenchmarkSuiteCaseExecution(
                case_id=case_id,
                status="completed",
                result=result,
                error=None,
            )
        except TimeoutError:

            return BenchmarkSuiteCaseExecution(
                case_id=case_id,
                status="error",
                result=None,
                error=(
                    "Benchmark case timeout " f"after " f"{self.case_timeout_seconds}s"
                ),
            )
        except Exception as exc:

            # 一个 Case 出问题，
            # 不能让整个 Suite 中断。
            return BenchmarkSuiteCaseExecution(
                case_id=case_id,
                status="error",
                result=None,
                error=(f"{type(exc).__name__}: " f"{exc}"),
            )

    @staticmethod
    def _build_summary(
        executions: list[BenchmarkSuiteCaseExecution],
    ) -> BenchmarkSuiteSummary:

        completed = [
            execution
            for execution in executions
            if (execution.status == "completed" and execution.result is not None)
        ]

        harness_errors = [
            execution for execution in executions if execution.status == "error"
        ]

        retrieval_passed = 0
        retrieval_partial = 0
        retrieval_failed = 0
        retrieval_not_evaluated = 0

        recalls: list[float] = []
        precisions: list[float] = []

        for execution in completed:

            result = execution.result

            if result is None:
                continue

            retrieval = result.retrieval

            if retrieval.status == "passed":
                retrieval_passed += 1

            elif retrieval.status == "partial":
                retrieval_partial += 1

            elif retrieval.status == "failed":
                retrieval_failed += 1

            else:
                retrieval_not_evaluated += 1

            if retrieval.recall is not None:
                recalls.append(retrieval.recall)

            if retrieval.precision is not None:
                precisions.append(retrieval.precision)

        average_recall = sum(recalls) / len(recalls) if recalls else None

        average_precision = sum(precisions) / len(precisions) if precisions else None

        return BenchmarkSuiteSummary(
            total_cases=len(executions),
            completed_cases=len(completed),
            harness_error_cases=len(harness_errors),
            retrieval_passed=(retrieval_passed),
            retrieval_partial=(retrieval_partial),
            retrieval_failed=(retrieval_failed),
            retrieval_not_evaluated=(retrieval_not_evaluated),
            average_recall=(average_recall),
            average_precision=(average_precision),
        )
