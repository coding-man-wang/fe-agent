import asyncio
from types import SimpleNamespace

import pytest

from agent_core.benchmark.selection_shadow_integration import (
    ShadowPolicyConfig,
    ShadowTaskContextSelectionService,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)


class FakeSelectionService:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def select(self, **kwargs):
        self.calls += 1
        return self.result


class MutatingShadowRunner:
    async def observe(self, *, result, **kwargs):
        # 故意破坏传入的 Selection。
        result.selection.selected_files.clear()
        result.selection.selected_count = 0

        return {
            "status": "complete",
            "effective_proposed_drops": ["src/demo.ts"],
        }


class FailingShadowRunner:
    async def observe(self, **kwargs):
        raise RuntimeError("simulated rerank failure")


class SlowShadowRunner:
    async def observe(self, **kwargs):
        await asyncio.sleep(1)
        return {"status": "complete"}


def make_result():
    file = HybridRelevantFile(
        relative_path="src/demo.ts",
        category="source",
        language="TypeScript",
        size_bytes=20,
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=20,
        selected_files=[file],
    )

    return SimpleNamespace(
        selection=selection,
        repository_revision="revision-001",
    )


async def call_service(service, tmp_path):
    return await service.select(
        repository_root=tmp_path,
        repository_revision="revision-001",
        task=TaskDefinition(
            task_id="TASK-001",
            task_type="code_change",
            title="测试任务",
            objective="测试 Shadow 安全",
            source_step_order=1,
        ),
        raw_requirement="修改 demo",
        requirement_spec=RequirementSpec(
            title="测试",
            summary="修改 demo",
        ),
    )


@pytest.mark.asyncio
async def test_off_mode_does_not_run_shadow(tmp_path):
    result = make_result()
    delegate = FakeSelectionService(result)
    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=delegate,
        config=ShadowPolicyConfig(mode="off"),
        emit=reports.append,
    )

    actual = await call_service(service, tmp_path)

    assert actual is result
    assert delegate.calls == 1
    assert reports == []


@pytest.mark.asyncio
async def test_shadow_mutation_cannot_change_selection(
    tmp_path,
):
    result = make_result()
    original = result.selection.model_dump(mode="json")
    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=FakeSelectionService(result),
        config=ShadowPolicyConfig(mode="shadow"),
        runner=MutatingShadowRunner(),
        emit=reports.append,
    )

    actual = await call_service(service, tmp_path)

    assert actual is result
    assert actual.selection.model_dump(mode="json") == original
    assert reports[0]["applied"] is False
    assert reports[0]["selection_unchanged"] is True


@pytest.mark.asyncio
async def test_shadow_error_preserves_selection(tmp_path):
    result = make_result()
    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=FakeSelectionService(result),
        config=ShadowPolicyConfig(mode="shadow"),
        runner=FailingShadowRunner(),
        emit=reports.append,
    )

    actual = await call_service(service, tmp_path)

    assert actual is result
    assert actual.selection.selected_count == 1
    assert reports[0]["status"] == "error"
    assert reports[0]["applied"] is False


@pytest.mark.asyncio
async def test_shadow_timeout_preserves_selection(
    tmp_path,
):
    result = make_result()
    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=FakeSelectionService(result),
        config=ShadowPolicyConfig(
            mode="shadow",
            timeout_seconds=0.01,
        ),
        runner=SlowShadowRunner(),
        emit=reports.append,
    )

    actual = await call_service(service, tmp_path)

    assert actual.selection.selected_count == 1
    assert reports[0]["status"] == "error"
    assert reports[0]["reason"] == "shadow_timeout"
    assert reports[0]["selection_unchanged"] is True


def test_active_mode_is_not_supported():
    with pytest.raises(ValueError):
        ShadowPolicyConfig(mode="active")
