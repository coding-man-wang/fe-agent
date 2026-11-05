import asyncio
from types import SimpleNamespace

import pytest
import json
from pathlib import Path

from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)

from agent_core.benchmark.shadow_audit import (
    JsonlShadowAuditStore,
)
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
    # P1: Shadow 会在运行前采集真实源码证据。
    # 因此必须先准备 Selection 中引用的文件。
    prepare_source(tmp_path)

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

    assert actual is result
    assert actual.selection.selected_count == 1

    assert len(reports) == 1
    assert reports[0]["status"] == "error"
    assert reports[0]["reason"] == "shadow_timeout"
    assert reports[0]["applied"] is False
    assert reports[0]["selection_unchanged"] is True


def test_active_mode_is_not_supported():
    with pytest.raises(ValueError):
        ShadowPolicyConfig(mode="active")


def prepare_source(root: Path) -> Path:
    path = root / "src" / "demo.ts"
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "export const value = 1\n",
        encoding="utf-8",
    )
    return path


class StableShadowRunner:
    async def observe(self, **kwargs):
        return {
            "status": "complete",
            "graph_proposed_drops": [],
            "rerank_proposed_drops": [],
            "effective_proposed_drops": [],
        }


class SourceChangingShadowRunner:
    async def observe(
        self,
        *,
        repository_root,
        **kwargs,
    ):
        path = Path(repository_root) / "src" / "demo.ts"

        path.write_text(
            "export const value = 2\n",
            encoding="utf-8",
        )

        return {
            "status": "complete",
            "graph_proposed_drops": ["src/demo.ts"],
            "rerank_proposed_drops": [],
            "effective_proposed_drops": ["src/demo.ts"],
        }


@pytest.mark.asyncio
async def test_p1_source_unchanged_accepts_observation(
    tmp_path,
):
    prepare_source(tmp_path)

    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=FakeSelectionService(make_result()),
        config=ShadowPolicyConfig(mode="shadow"),
        runner=StableShadowRunner(),
        emit=reports.append,
    )

    result = await call_service(
        service,
        tmp_path,
    )

    report = reports[0]
    binding = report["evidence_binding"]

    assert result.selection.selected_count == 1
    assert report["status"] == "complete"
    assert report["applied"] is False

    assert binding["source_matched"] is True
    assert binding["selection_matched"] is True

    assert binding["source_sha256_before"] == binding["source_sha256_after"]

    assert binding["changed_paths"] == []
    assert report["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_p1_source_change_invalidates_proposal(
    tmp_path,
):
    prepare_source(tmp_path)

    reports = []

    service = ShadowTaskContextSelectionService(
        delegate=FakeSelectionService(make_result()),
        config=ShadowPolicyConfig(mode="shadow"),
        runner=SourceChangingShadowRunner(),
        emit=reports.append,
    )

    result = await call_service(
        service,
        tmp_path,
    )

    report = reports[0]
    binding = report["evidence_binding"]

    assert result.selection.selected_count == 1

    assert report["status"] == "abstained"
    assert report["reason"] == ("evidence_changed_during_shadow")

    assert binding["source_matched"] is False
    assert binding["selection_matched"] is True
    assert binding["changed_paths"] == ["src/demo.ts"]

    assert report["effective_proposed_drops"] == []
    assert report["applied"] is False
    assert report["selection_unchanged"] is True


def test_p1_selection_metadata_changes_fingerprint(
    tmp_path,
):
    prepare_source(tmp_path)
    result = make_result()

    before = capture_selection_evidence(
        repository_root=tmp_path,
        selection=result.selection,
    )

    result.selection.selected_files[0].retrieval_score = 0.5

    after = capture_selection_evidence(
        repository_root=tmp_path,
        selection=result.selection,
    )

    # 文件内容未变。
    assert before.source_sha256 == after.source_sha256

    # 选择元数据发生变化。
    assert before.selection_sha256 != after.selection_sha256


def test_p1_audit_store_persists_evidence_binding(
    tmp_path,
):
    audit_path = tmp_path / "audit" / "shadow.jsonl"

    store = JsonlShadowAuditStore(
        path=audit_path,
    )

    store.append(
        {
            "policy_version": "v1.4.9-P1",
            "mode": "shadow",
            "task_id": "TEST-001",
            "repository_revision": "revision-001",
            "requirement_sha256": "requirement-hash",
            "status": "complete",
            "applied": False,
            "elapsed_ms": 12.5,
            "effective_proposed_drops": [],
            "evidence_binding": {
                "selection_sha256": "selection-hash",
                "source_sha256_before": "source-hash",
                "source_sha256_after": "source-hash",
                "source_matched": True,
                "selection_matched": True,
                "changed_paths": [],
                "secret": "SHOULD_NOT_STORE",
            },
            "api_key": "SECRET_API_KEY",
            "raw_requirement": "PRIVATE_PROMPT",
            "source_code": "CONFIDENTIAL_SOURCE",
        }
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["task_id"] == "TEST-001"
    assert record["elapsed_ms"] == 12.5
    assert record["evidence_binding"]["source_matched"] is True

    assert record["evidence_binding"]["selection_sha256"] == "selection-hash"

    content = lines[0]

    assert "SECRET_API_KEY" not in content
    assert "PRIVATE_PROMPT" not in content
    assert "CONFIDENTIAL_SOURCE" not in content
    assert "SHOULD_NOT_STORE" not in content
