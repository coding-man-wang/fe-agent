from pathlib import Path

import pytest

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.coding.orchestrator import (
    AnalysisOrchestrationError,
    ReadOnlyCodingOrchestrator,
)
from agent_core.context.model import (
    TaskContextBundle,
)
from agent_core.sdd.model import (
    TaskDefinition,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


def create_task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
    task_type: str = "code_change",
) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        task_type=task_type,
        title=f"分析任务 {task_id}",
        objective="分析相关代码并提出修改方案",
        source_step_order=1,
        depends_on=(depends_on if depends_on is not None else []),
        inputs=["RequirementSpec"],
        target_modules=["src/api"],
        expected_outputs=["代码分析提案"],
        verification=["检查分析结果"],
        acceptance_criteria_refs=[],
    )


class FakeContextFactory:

    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root
        self.calls: list[str] = []

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle:
        self.calls.append(task.task_id)

        bundle = create_test_bundle(self.root)

        # 复用真实 fixture，但替换成调度器指定的 Task。
        return bundle.model_copy(update={"task": task})


class FakeCodingAnalyzer:

    def __init__(
        self,
        *,
        should_fail: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.calls: list[str] = []

    async def analyze(
        self,
        bundle: TaskContextBundle,
    ) -> ReadOnlyCodingResult:
        self.calls.append(bundle.task.task_id)

        if self.should_fail:
            raise RuntimeError("模拟分析失败")

        return ReadOnlyCodingResult(
            task_id=bundle.task.task_id,
            summary="已完成只读代码分析",
            findings=[],
            proposed_changes=[],
            verification_plan=["后续执行真实代码验证"],
            missing_information=[],
            execution_status="not_executed",
        )


@pytest.mark.asyncio
async def test_runs_ready_task(
    tmp_path: Path,
):
    factory = FakeContextFactory(tmp_path)

    analyzer = FakeCodingAnalyzer()

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=factory,
        analyzer=analyzer,
    )

    tasks = [create_task("TASK-001")]

    record = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert record is not None
    assert record.task_id == "TASK-001"
    assert record.status == "succeeded"
    assert record.attempts == 1

    assert record.result is not None

    assert record.result.execution_status == "not_executed"

    assert factory.calls == ["TASK-001"]
    assert analyzer.calls == ["TASK-001"]


@pytest.mark.asyncio
async def test_dependency_requires_real_completion(
    tmp_path: Path,
):
    factory = FakeContextFactory(tmp_path)

    analyzer = FakeCodingAnalyzer()

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=factory,
        analyzer=analyzer,
    )

    tasks = [
        create_task("TASK-001"),
        create_task(
            "TASK-002",
            depends_on=["TASK-001"],
        ),
    ]

    first = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert first is not None
    assert first.task_id == "TASK-001"
    assert first.status == "succeeded"

    # TASK-001 只是分析成功，没有真正完成。
    blocked = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert blocked is None
    assert analyzer.calls == ["TASK-001"]

    # 只有外部真正确认 TASK-001 完成，
    # TASK-002 才能进入分析。
    second = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids={"TASK-001"},
    )

    assert second is not None
    assert second.task_id == "TASK-002"
    assert second.status == "succeeded"


@pytest.mark.asyncio
async def test_succeeded_analysis_is_not_repeated(
    tmp_path: Path,
):
    analyzer = FakeCodingAnalyzer()

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=FakeContextFactory(tmp_path),
        analyzer=analyzer,
    )

    tasks = [create_task("TASK-001")]

    await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    second = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert second is None
    assert analyzer.calls == ["TASK-001"]


@pytest.mark.asyncio
async def test_failure_is_recorded(
    tmp_path: Path,
):
    analyzer = FakeCodingAnalyzer(should_fail=True)

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=FakeContextFactory(tmp_path),
        analyzer=analyzer,
    )

    tasks = [create_task("TASK-001")]

    record = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert record is not None
    assert record.status == "failed"
    assert record.result is None
    assert "模拟分析失败" in (record.error or "")

    # 不允许未经明确重置就无限重试。
    second = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert second is None
    assert analyzer.calls == ["TASK-001"]


@pytest.mark.asyncio
async def test_explicit_retry_preserves_attempts(
    tmp_path: Path,
):
    analyzer = FakeCodingAnalyzer(should_fail=True)

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=FakeContextFactory(tmp_path),
        analyzer=analyzer,
    )

    tasks = [create_task("TASK-001")]

    first = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert first is not None
    assert first.status == "failed"
    assert first.attempts == 1

    orchestrator.reset_failed("TASK-001")

    analyzer.should_fail = False

    second = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert second is not None
    assert second.status == "succeeded"
    assert second.attempts == 2


@pytest.mark.asyncio
async def test_unknown_dependency_is_rejected(
    tmp_path: Path,
):
    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=FakeContextFactory(tmp_path),
        analyzer=FakeCodingAnalyzer(),
    )

    tasks = [
        create_task(
            "TASK-001",
            depends_on=["TASK-999"],
        )
    ]

    with pytest.raises(
        AnalysisOrchestrationError,
        match="未知依赖",
    ):
        await orchestrator.run_next(
            tasks=tasks,
            completed_task_ids=set(),
        )


@pytest.mark.asyncio
async def test_unsupported_task_type_is_not_executed(
    tmp_path: Path,
):
    analyzer = FakeCodingAnalyzer()

    orchestrator = ReadOnlyCodingOrchestrator(
        context_factory=FakeContextFactory(tmp_path),
        analyzer=analyzer,
    )

    tasks = [
        create_task(
            "TASK-001",
            task_type="test",
        )
    ]

    result = await orchestrator.run_next(
        tasks=tasks,
        completed_task_ids=set(),
    )

    assert result is None
    assert analyzer.calls == []
