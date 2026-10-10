from pathlib import Path

import pytest

from agent_core.coding.checkpoint import (
    AnalysisCheckpointStore,
)
from agent_core.coding.durable_workflow import (
    DurableReadOnlyWorkflow,
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
from agent_core.context.model import (
    RenderedContext,
    TaskContextBundle,
)
from agent_core.context.renderer import (
    ContextRenderer,
)
from agent_core.sdd.model import (
    TaskDefinition,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


class FakeContextFactory:

    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root
        self.calls = 0

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle:
        self.calls += 1

        bundle = create_test_bundle(self.root)

        return bundle.model_copy(update={"task": task})


class FakePreparedAnalyzer:

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.analyze_calls = 0

        self.last_messages: RenderedContext | None = None

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext:
        self.prepare_calls += 1

        rendered = ContextRenderer().render(bundle)

        return RenderedContext(
            system_message=(rendered.system_message + "\n\n只读分析输出规范"),
            user_message=rendered.user_message,
        )

    async def analyze_prepared(
        self,
        *,
        bundle: TaskContextBundle,
        messages: RenderedContext,
    ) -> ReadOnlyCodingResult:
        self.analyze_calls += 1

        self.last_messages = messages

        return ReadOnlyCodingResult(
            task_id=bundle.task.task_id,
            summary="已分析相关代码",
            findings=[],
            proposed_changes=[],
            verification_plan=["后续执行验证"],
            missing_information=[],
            execution_status="not_executed",
        )


def create_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户列表查询",
        objective="支持 status 参数",
        source_step_order=2,
        depends_on=[],
        inputs=["RequirementSpec"],
        target_modules=["src/api"],
        expected_outputs=["修改提案"],
        verification=["检查提案"],
        acceptance_criteria_refs=[],
    )


def create_workflow(
    root: Path,
    *,
    factory: FakeContextFactory,
    analyzer: FakePreparedAnalyzer,
) -> DurableReadOnlyWorkflow:
    return DurableReadOnlyWorkflow(
        context_factory=factory,
        analyzer=analyzer,
        checkpoint_store=(AnalysisCheckpointStore(root)),
        artifact_store=(AnalysisArtifactStore(root)),
    )


@pytest.mark.asyncio
async def test_success_uses_one_context(
    tmp_path: Path,
):
    factory = FakeContextFactory(tmp_path)

    analyzer = FakePreparedAnalyzer()

    workflow = create_workflow(
        tmp_path,
        factory=factory,
        analyzer=analyzer,
    )

    record = await workflow.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert record is not None
    assert record.status == "succeeded", f"Analysis failed: {record.error}"

    assert factory.calls == 1
    assert analyzer.prepare_calls == 1
    assert analyzer.analyze_calls == 1

    assert record.artifact_id is not None
    assert record.context_hash is not None


@pytest.mark.asyncio
async def test_hash_matches_actual_messages(
    tmp_path: Path,
):
    analyzer = FakePreparedAnalyzer()

    workflow = create_workflow(
        tmp_path,
        factory=FakeContextFactory(tmp_path),
        analyzer=analyzer,
    )

    record = await workflow.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert record is not None
    assert record.status == "succeeded"

    assert analyzer.last_messages is not None

    expected_hash = calculate_messages_hash(analyzer.last_messages)

    assert record.context_hash == expected_hash

    assert record.artifact_id is not None

    artifact = AnalysisArtifactStore(tmp_path).load(record.artifact_id)

    assert artifact.context_hash == expected_hash

    assert artifact.result == record.result


@pytest.mark.asyncio
async def test_restart_does_not_repeat_success(
    tmp_path: Path,
):
    first_analyzer = FakePreparedAnalyzer()

    first = create_workflow(
        tmp_path,
        factory=FakeContextFactory(tmp_path),
        analyzer=first_analyzer,
    )

    initial = await first.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert initial is not None
    assert initial.status == "succeeded"

    # 创建完全新的 Workflow，模拟重启。
    second_analyzer = FakePreparedAnalyzer()

    second = create_workflow(
        tmp_path,
        factory=FakeContextFactory(tmp_path),
        analyzer=second_analyzer,
    )

    restored = second.records["TASK-003"]

    assert restored.status == "succeeded"
    assert restored.artifact_id == initial.artifact_id

    next_record = await second.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert next_record is None
    assert second_analyzer.analyze_calls == 0


@pytest.mark.asyncio
async def test_interrupted_running_requires_retry(
    tmp_path: Path,
):
    store = AnalysisCheckpointStore(tmp_path)

    store.save(
        {
            "TASK-003": TaskAnalysisRecord(
                task_id="TASK-003",
                status="running",
                attempts=2,
            )
        }
    )

    workflow = create_workflow(
        tmp_path,
        factory=FakeContextFactory(tmp_path),
        analyzer=FakePreparedAnalyzer(),
    )

    restored = workflow.records["TASK-003"]

    assert restored.status == "failed"
    assert restored.attempts == 2
    assert "InterruptedExecution" in (restored.error or "")

    # 未重置时不能自动重试。
    blocked = await workflow.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert blocked is None

    workflow.reset_failed("TASK-003")

    retried = await workflow.run_next(
        tasks=[create_task()],
        completed_task_ids=set(),
    )

    assert retried is not None
    assert retried.status == "succeeded"
    assert retried.attempts == 3
