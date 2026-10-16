import asyncio
from typing import Protocol
from uuid import uuid4

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.case_model import (
    BenchmarkCase,
)
from agent_core.benchmark.evaluation_model import (
    BenchmarkRetrievalEvaluation,
)
from agent_core.benchmark.repository_guard import (
    BenchmarkRepositoryGuard,
)
from agent_core.benchmark.retrieval_model import (
    BenchmarkRetrievalCapture,
)
from agent_core.benchmark.run_model import (
    BenchmarkRunRecord,
    utc_now,
)
from agent_core.benchmark.run_store import (
    BenchmarkRunStore,
)
from agent_core.sdd.model import (
    ImplementationPlan,
    ProjectContext,
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
)


class RequirementAnalyzerProtocol(Protocol):

    async def analyze(
        self,
        requirement: str,
    ) -> RequirementSpec: ...


class ProjectContextProviderProtocol(Protocol):

    def build(
        self,
    ) -> ProjectContext: ...


class TechnicalSpecGeneratorProtocol(Protocol):

    async def generate(
        self,
        requirement: RequirementSpec,
        project_context: ProjectContext,
    ) -> TechnicalSpec: ...


class ImplementationPlanGeneratorProtocol(Protocol):

    async def generate(
        self,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        project_context: ProjectContext,
    ) -> ImplementationPlan: ...


class TaskPlannerProtocol(Protocol):

    async def generate(
        self,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        implementation_plan: ImplementationPlan,
        project_context: ProjectContext,
    ) -> TaskPlan: ...


class RetrievalCaptureProtocol(Protocol):

    async def capture(
        self,
        *,
        task_plan: TaskPlan,
        raw_requirement: str,
        requirement_spec: RequirementSpec,
        repository_revision: str,
    ) -> BenchmarkRetrievalCapture: ...


class RetrievalEvaluatorProtocol(Protocol):

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        capture: BenchmarkRetrievalCapture,
    ) -> BenchmarkRetrievalEvaluation: ...


class BenchmarkRunner:

    def __init__(
        self,
        *,
        case_loader: BenchmarkCaseLoader,
        repository_guard: BenchmarkRepositoryGuard,
        project_context_provider: ProjectContextProviderProtocol,
        requirement_analyzer: RequirementAnalyzerProtocol,
        technical_spec_generator: TechnicalSpecGeneratorProtocol,
        implementation_plan_generator: ImplementationPlanGeneratorProtocol,
        task_planner: TaskPlannerProtocol,
        retrieval_capture_service: RetrievalCaptureProtocol,
        retrieval_evaluator: RetrievalEvaluatorProtocol,
        run_store: BenchmarkRunStore,
    ) -> None:

        self.case_loader = case_loader

        self.repository_guard = repository_guard

        self.project_context_provider = project_context_provider

        self.requirement_analyzer = requirement_analyzer

        self.technical_spec_generator = technical_spec_generator

        self.implementation_plan_generator = implementation_plan_generator

        self.task_planner = task_planner

        self.retrieval_capture_service = retrieval_capture_service

        self.retrieval_evaluator = retrieval_evaluator

        self.run_store = run_store

    async def run(
        self,
        *,
        case_id: str,
    ) -> BenchmarkRunRecord:

        # ===================================
        # 1. Load Case
        # ===================================

        case = self.case_loader.load(case_id)

        # ===================================
        # 2. Baseline Guard
        # ===================================

        state = await self.repository_guard.require_ready(case=case)

        # ===================================
        # 3. Durable Run
        # ===================================

        run_id = uuid4().hex

        record = BenchmarkRunRecord(
            run_id=run_id,
            case_id=case.case_id,
            repository_root=(state.repository_root),
            base_commit_sha=(state.actual_commit_sha),
            status="running",
            stage=("baseline_verified"),
            started_at=utc_now(),
        )

        self.run_store.save(record)

        try:

            # =================================
            # Project Context
            # =================================

            project_context = self.project_context_provider.build()

            record = record.model_copy(
                deep=True,
                update={
                    "project_context": (project_context),
                },
            )

            self.run_store.save(record)

            # =================================
            # Requirement
            #
            # Hidden Oracle 仍然没有进入 Agent。
            # =================================

            requirement_spec = await self.requirement_analyzer.analyze(case.requirement)

            record = record.model_copy(
                deep=True,
                update={
                    "stage": ("requirement_analyzed"),
                    "requirement_spec": (requirement_spec),
                },
            )

            self.run_store.save(record)

            # =================================
            # Technical
            # =================================

            technical_spec = await self.technical_spec_generator.generate(
                requirement_spec,
                project_context,
            )

            record = record.model_copy(
                deep=True,
                update={
                    "stage": ("technical_designed"),
                    "technical_spec": (technical_spec),
                },
            )

            self.run_store.save(record)

            # =================================
            # Implementation Plan
            # =================================

            implementation_plan = await self.implementation_plan_generator.generate(
                requirement_spec,
                technical_spec,
                project_context,
            )

            record = record.model_copy(
                deep=True,
                update={
                    "stage": ("implementation_planned"),
                    "implementation_plan": (implementation_plan),
                },
            )

            self.run_store.save(record)

            # =================================
            # Task Plan
            # =================================

            task_plan = await self.task_planner.generate(
                requirement_spec,
                technical_spec,
                implementation_plan,
                project_context,
            )

            record = record.model_copy(
                deep=True,
                update={
                    "stage": ("task_planned"),
                    "task_plan": (task_plan),
                },
            )

            self.run_store.save(record)

            # =================================
            # Retrieval Capture
            #
            # Agent 运行到这里依然没有看到
            # case.expectation。
            # =================================

            retrieval_capture = await self.retrieval_capture_service.capture(
                task_plan=task_plan,
                # 用户最初输入，
                # 没有经过任何 LLM 改写。
                raw_requirement=(case.requirement),
                # RequirementAnalyzer
                # 结构化之后的需求。
                requirement_spec=(requirement_spec),
                repository_revision=(record.base_commit_sha),
            )

            record = record.model_copy(
                deep=True,
                update={
                    "stage": ("retrieval_captured"),
                    "retrieval_capture": (retrieval_capture),
                },
            )

            self.run_store.save(record)

            # =================================
            # Retrieval Evaluation
            #
            # Trust Boundary:
            #
            # 从这里开始才允许读取
            # Hidden Oracle。
            # =================================

            retrieval_evaluation = self.retrieval_evaluator.evaluate(
                case=case,
                capture=(retrieval_capture),
            )

            record = record.model_copy(
                deep=True,
                update={
                    "status": "completed",
                    "stage": ("retrieval_evaluated"),
                    "retrieval_evaluation": (retrieval_evaluation),
                    "finished_at": (utc_now()),
                    "error": None,
                },
            )

            self.run_store.save(record)

            return record
        except asyncio.CancelledError:

            failed = record.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error": (
                        "CancelledError: " "benchmark run cancelled " "or timed out"
                    ),
                },
            )

            self.run_store.save(failed)

            # CancelledError 一定要继续抛出，
            # 不能吞掉。
            raise
        except Exception as exc:

            failed = record.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "finished_at": (utc_now()),
                    "error": (f"{type(exc).__name__}: " f"{exc}"),
                },
            )

            self.run_store.save(failed)

            return failed
