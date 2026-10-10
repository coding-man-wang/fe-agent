from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from agent_core.coding.protocols import (
    CodingAnalyzer,
    ContextFactory,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class AnalysisOrchestrationError(ValueError):
    """分析任务编排错误。"""


class ReadOnlyCodingOrchestrator:
    SUPPORTED_TASK_TYPES = {
        "investigation",
        "code_change",
    }

    def __init__(
        self,
        *,
        context_factory: ContextFactory,
        analyzer: CodingAnalyzer,
    ) -> None:

        self.context_factory = context_factory
        self.analyzer = analyzer

        self.records: dict[
            str,
            TaskAnalysisRecord,
        ] = {}

    def _validate_tasks(
        self,
        tasks: list[TaskDefinition],
    ) -> None:

        task_ids = [task.task_id for task in tasks]

        if len(task_ids) != len(set(task_ids)):
            raise AnalysisOrchestrationError("TaskPlan 存在重复 task_id")

        known_ids = set(task_ids)

        for task in tasks:

            for dependency in task.depends_on:

                if dependency not in known_ids:
                    raise AnalysisOrchestrationError(
                        f"{task.task_id} 引用了未知依赖: " f"{dependency}"
                    )

    def get_ready_tasks(
        self,
        *,
        tasks: list[TaskDefinition],
        completed_task_ids: set[str],
    ) -> list[TaskDefinition]:

        self._validate_tasks(tasks)

        known_ids = {task.task_id for task in tasks}

        unknown_completed = completed_task_ids - known_ids

        if unknown_completed:
            raise AnalysisOrchestrationError(
                "completed_task_ids 包含未知任务: " f"{sorted(unknown_completed)}"
            )

        ready: list[TaskDefinition] = []

        for task in tasks:

            if task.task_type not in (self.SUPPORTED_TASK_TYPES):
                continue

            # 真正完成的任务不需要重新分析。
            if task.task_id in completed_task_ids:
                continue

            record = self.records.get(task.task_id)

            # 已成功分析，不重复执行。
            # 已失败的任务，也不会未经允许自动重试。
            if record is not None:
                if record.status in {
                    "running",
                    "succeeded",
                    "failed",
                }:
                    continue

            dependencies_satisfied = all(
                dependency in completed_task_ids for dependency in task.depends_on
            )

            if dependencies_satisfied:
                ready.append(task)

        # 稳定顺序：不依赖调用方传入列表的排列顺序。
        ready.sort(key=lambda task: task.task_id)

        return ready

    async def run_next(
        self,
        *,
        tasks: list[TaskDefinition],
        completed_task_ids: set[str],
    ) -> TaskAnalysisRecord | None:

        ready = self.get_ready_tasks(
            tasks=tasks,
            completed_task_ids=completed_task_ids,
        )

        if not ready:
            return None

        task = ready[0]

        existing = self.records.get(task.task_id)

        record = TaskAnalysisRecord(
            task_id=task.task_id,
            status="running",
            attempts=(existing.attempts + 1 if existing is not None else 1),
        )

        self.records[task.task_id] = record

        try:
            bundle = await self.context_factory.build(task)

            if bundle.task.task_id != task.task_id:
                raise AnalysisOrchestrationError(
                    "ContextFactory 返回了错误的 Task Context"
                )

            result = await self.analyzer.analyze(bundle)

            if result.task_id != task.task_id:
                raise AnalysisOrchestrationError("CodingAnalyzer 返回了错误的 Task ID")

            if result.execution_status != "not_executed":
                raise AnalysisOrchestrationError("只读分析不能声明任务已经执行")

            record.status = "succeeded"
            record.result = result
            record.error = None

        except Exception as exc:

            record.status = "failed"
            record.result = None

            record.error = f"{type(exc).__name__}: {exc}"

        return record

    def reset_failed(
        self,
        task_id: str,
    ) -> None:

        record = self.records.get(task_id)

        if record is None:
            raise AnalysisOrchestrationError(f"不存在分析记录: {task_id}")

        if record.status != "failed":
            raise AnalysisOrchestrationError("只能重置 failed 状态的分析任务")

        record.status = "pending"
        record.error = None
