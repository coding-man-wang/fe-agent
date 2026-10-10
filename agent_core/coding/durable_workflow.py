from typing import Protocol

from agent_core.coding.checkpoint import (
    AnalysisCheckpointStore,
)
from agent_core.coding.handoff import (
    AnalysisArtifactStore,
    calculate_messages_hash,
)
from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from agent_core.coding.orchestrator import (
    ReadOnlyCodingOrchestrator,
)
from agent_core.context.model import (
    RenderedContext,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class DurableContextFactory(Protocol):

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle: ...


class PreparedCodingAnalyzer(Protocol):

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext: ...

    async def analyze_prepared(
        self,
        *,
        bundle: TaskContextBundle,
        messages: RenderedContext,
    ) -> ReadOnlyCodingResult: ...


class DurableReadOnlyWorkflow:

    def __init__(
        self,
        *,
        context_factory: DurableContextFactory,
        analyzer: PreparedCodingAnalyzer,
        checkpoint_store: AnalysisCheckpointStore,
        artifact_store: AnalysisArtifactStore,
    ) -> None:

        self.context_factory = context_factory
        self.analyzer = analyzer

        self.checkpoint_store = checkpoint_store
        self.artifact_store = artifact_store

        # 复用原有 Scheduler 的任务依赖判断逻辑。
        self.scheduler = ReadOnlyCodingOrchestrator(
            context_factory=context_factory,
            analyzer=analyzer,
        )

        checkpoint = self.checkpoint_store.load()

        self.scheduler.records = checkpoint.records

        self._recover_interrupted()

    @property
    def records(
        self,
    ) -> dict[str, TaskAnalysisRecord]:

        return self.scheduler.records

    def _recover_interrupted(
        self,
    ) -> None:

        changed = False

        for record in self.records.values():

            if record.status != "running":
                continue

            record.status = "failed"

            record.error = (
                "InterruptedExecution: " "上一次运行在分析完成前中断，" "需要显式重试"
            )

            record.result = None
            record.artifact_id = None
            record.context_hash = None

            changed = True

        if changed:
            self.checkpoint_store.save(self.records)

    async def run_next(
        self,
        *,
        tasks: list[TaskDefinition],
        completed_task_ids: set[str],
    ) -> TaskAnalysisRecord | None:

        ready = self.scheduler.get_ready_tasks(
            tasks=tasks,
            completed_task_ids=completed_task_ids,
        )

        if not ready:
            return None

        task = ready[0]

        previous = self.records.get(task.task_id)

        attempts = previous.attempts + 1 if previous is not None else 1

        record = TaskAnalysisRecord(
            task_id=task.task_id,
            status="running",
            attempts=attempts,
        )

        self.records[task.task_id] = record

        # 第一处持久化边界：
        # 模型调用之前，先记录正在执行。
        self.checkpoint_store.save(self.records)

        try:
            # 每次 Attempt 只构建一次 Context。
            bundle = await self.context_factory.build(task)

            if bundle.task.task_id != task.task_id:
                raise ValueError("ContextFactory 返回了错误的 Task")

            # 只准备一次实际模型消息。
            messages = self.analyzer.prepare_messages(bundle)

            context_hash = calculate_messages_hash(messages)

            result = await self.analyzer.analyze_prepared(
                bundle=bundle,
                messages=messages,
            )

            if result.task_id != task.task_id:
                raise ValueError("分析结果的 Task ID 不一致")

            if result.execution_status != "not_executed":
                raise ValueError("只读分析不能声明任务已经执行")

            # 先准备成功记录，但不提前提交 Checkpoint。
            success_record = record.model_copy(
                deep=True,
                update={
                    "status": "succeeded",
                    "result": result,
                    "error": None,
                },
            )

            # 第二处持久化边界：
            # 先保存实际分析产物。
            artifact, _ = self.artifact_store.publish(
                bundle=bundle,
                record=success_record,
                context_hash=context_hash,
            )

            success_record.artifact_id = artifact.artifact_id

            success_record.context_hash = context_hash

            # 第三处持久化边界：
            # Artifact 已存在，才提交成功状态。
            self.records[task.task_id] = success_record

            self.checkpoint_store.save(self.records)

            return success_record

        except Exception as exc:

            failed_record = record.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "result": None,
                    "artifact_id": None,
                    "context_hash": None,
                    "error": (f"{type(exc).__name__}: {exc}"),
                },
            )

            self.records[task.task_id] = failed_record

            self.checkpoint_store.save(self.records)

            return failed_record

    def reset_failed(
        self,
        task_id: str,
    ) -> None:

        # 复用旧状态机的显式重试规则。
        self.scheduler.reset_failed(task_id)

        # reset 也属于状态变更，必须保存。
        self.checkpoint_store.save(self.records)
