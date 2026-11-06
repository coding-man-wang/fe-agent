import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_core.benchmark.full_execution import (
    FullExecutionBenchmarkService,
)
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)

REVISION = "test-revision-001"
REQUIREMENT = "修改 demo 功能"


@pytest.fixture
def shadow_case(tmp_path):
    path = tmp_path / "src" / "demo.ts"
    path.parent.mkdir(parents=True)

    path.write_text(
        "export const demo = 1\n",
        encoding="utf-8",
    )

    item = HybridRelevantFile(
        relative_path="src/demo.ts",
        category="source",
        language="TypeScript",
        size_bytes=path.stat().st_size,
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=item.size_bytes,
        selected_files=[item],
    )

    task = TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="修改 demo",
        objective=REQUIREMENT,
        source_step_order=1,
    )

    requirement = RequirementSpec(
        title="修改 demo",
        summary=REQUIREMENT,
    )

    technical_spec = TechnicalSpec(
        summary="修改 demo 常量",
    )

    repository = RepositoryProfile(
        project_name="shadow-test",
        root_path=str(tmp_path),
        language="TypeScript",
    )

    bundle = ProjectContextBuilder().build(
        repository_root=tmp_path,
        requirement=requirement,
        technical_spec=technical_spec,
        task=task,
        repository=repository,
        selection=selection,
        max_files=8,
        max_total_chars=40_000,
        max_chars_per_file=10_000,
    )

    assert len(bundle.code_context) == 1

    return {
        "root": tmp_path,
        "path": path,
        "selection": selection,
        "result": SimpleNamespace(
            repository_revision=REVISION,
            selection=selection,
        ),
        "bundle": bundle,
    }


class SuccessfulRunner:
    def __init__(self):
        self.calls = 0

    async def observe(self, *, result, **kwargs):
        self.calls += 1

        # 修改传入的副本，验证原始 Selection
        # 不受 Shadow 副作用影响。
        result.selection.selected_files[0].retrieval_score = 0.1

        return {
            "status": "complete",
            "baseline_file_count": 1,
            "proposed_file_count": 1,
            "graph_proposed_drops": [],
            "rerank_proposed_drops": [],
            "effective_proposed_drops": [],
            "rerank_status": "complete",
            "rerank_filter_status": "no_change",
        }


class SlowRunner:
    async def observe(self, **kwargs):
        await asyncio.sleep(0.2)
        return {"status": "complete"}


class ErrorRunner:
    async def observe(self, **kwargs):
        raise RuntimeError("simulated rerank error")


class ChangingSourceRunner:
    async def observe(self, *, repository_root, **kwargs):
        path = repository_root / "src" / "demo.ts"

        path.write_text(
            "export const demo = 999\n",
            encoding="utf-8",
        )

        return {"status": "complete"}


async def observe(case, runner, timeout=1.0):
    return await FullExecutionContextShadowObserver(
        runner=runner,
        timeout_seconds=timeout,
    ).observe(
        repository_root=case["root"],
        repository_revision=REVISION,
        raw_requirement=REQUIREMENT,
        selection_result=case["result"],
        bundle=case["bundle"],
    )


@pytest.mark.asyncio
async def test_shadow_uses_existing_selection(
    shadow_case,
):
    runner = SuccessfulRunner()

    original_selection = shadow_case["selection"].model_dump(mode="json")

    original_bundle = shadow_case["bundle"].model_dump(mode="json")

    report = await observe(shadow_case, runner)

    assert runner.calls == 1
    assert report["status"] == "complete"
    assert report["applied"] is False
    assert report["selection_unchanged"] is True
    assert report["context_unchanged"] is True
    assert report["bundle_source_matched"] is True

    assert shadow_case["selection"].model_dump(mode="json") == original_selection

    assert shadow_case["bundle"].model_dump(mode="json") == original_bundle


@pytest.mark.asyncio
async def test_timeout_preserves_context(
    shadow_case,
):
    report = await observe(
        shadow_case,
        SlowRunner(),
        timeout=0.01,
    )

    assert report["status"] == "error"
    assert report["reason"] == "shadow_timeout"
    assert report["applied"] is False
    assert report["context_unchanged"] is True
    assert report["selection_unchanged"] is True


@pytest.mark.asyncio
async def test_exception_preserves_context(
    shadow_case,
):
    report = await observe(
        shadow_case,
        ErrorRunner(),
    )

    assert report["status"] == "error"
    assert report["reason"] == ("shadow_exception:RuntimeError")
    assert report["applied"] is False
    assert report["context_unchanged"] is True


@pytest.mark.asyncio
async def test_source_change_abstains(
    shadow_case,
):
    report = await observe(
        shadow_case,
        ChangingSourceRunner(),
    )

    assert report["status"] == "abstained"
    assert report["reason"] == ("final_context_evidence_changed")

    assert report["bundle_source_matched"] is False
    assert report["effective_proposed_drops"] == []
    assert report["applied"] is False
    assert report["selection_unchanged"] is True


@pytest.mark.asyncio
async def test_bundle_mutation_is_detected(
    shadow_case,
):
    bundle = shadow_case["bundle"]

    class MutatingBundleRunner:
        async def observe(self, **kwargs):
            bundle.code_context[0].content = "unexpected mutation"
            return {"status": "complete"}

    report = await observe(
        shadow_case,
        MutatingBundleRunner(),
    )

    assert report["status"] == "abstained"
    assert report["context_unchanged"] is False
    assert report["effective_proposed_drops"] == []


@pytest.mark.asyncio
async def test_revision_mismatch_abstains(
    shadow_case,
):
    shadow_case["result"].repository_revision = "another-revision"

    runner = SuccessfulRunner()

    report = await observe(
        shadow_case,
        runner,
    )

    assert report["status"] == "abstained"
    assert report["reason"] == ("repository_revision_mismatch")
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_full_execution_hook_writes_audit(
    shadow_case,
    tmp_path,
):
    runner = SuccessfulRunner()

    # 独立测试 FullExecution 的接入方法，
    # 不启动真正的 Patch Apply。
    service = object.__new__(FullExecutionBenchmarkService)

    service.run_root = tmp_path / "run"
    service.context_shadow_observer = FullExecutionContextShadowObserver(
        runner=runner,
    )

    report = await service._observe_context_shadow(
        repository_root=shadow_case["root"],
        repository_revision=REVISION,
        raw_requirement=REQUIREMENT,
        selection_result=shadow_case["result"],
        bundle=shadow_case["bundle"],
    )

    audit_path = service.run_root / "context_shadow.jsonl"

    lines = audit_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert report["status"] == "complete"
    assert record["mode"] == "shadow"
    assert record["applied"] is False
    assert record["task_id"] == "TASK-001"
    assert record["bundle_source_matched"] is True
    assert record["bundle_sha256_before"] == record["bundle_sha256_after"]


@pytest.mark.asyncio
async def test_audit_failure_does_not_block(
    shadow_case,
    tmp_path,
):
    service = object.__new__(FullExecutionBenchmarkService)

    # 让审计目录创建失败。
    blocker = tmp_path / "blocker"
    blocker.write_text(
        "not a directory",
        encoding="utf-8",
    )

    service.run_root = blocker
    service.context_shadow_observer = FullExecutionContextShadowObserver(
        runner=SuccessfulRunner()
    )

    report = await service._observe_context_shadow(
        repository_root=shadow_case["root"],
        repository_revision=REVISION,
        raw_requirement=REQUIREMENT,
        selection_result=shadow_case["result"],
        bundle=shadow_case["bundle"],
    )

    assert report["status"] == "complete"
    assert report["applied"] is False
    assert report["context_unchanged"] is True
