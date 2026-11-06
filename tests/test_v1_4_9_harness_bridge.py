from types import SimpleNamespace

import pytest

from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.harness_context_policy_bridge import (
    HarnessContextPolicyBridge,
)
from agent_core.change_surface.context_policy import (
    RoleAwareContextPolicy,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    FileRecord,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.harness.trace import JsonlHarnessTraceStore
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


@pytest.fixture
def bridge_case(tmp_path):
    src = tmp_path / "src"
    src.mkdir()

    files = {
        "src/a.ts": "export const a = 1\n",
        "src/b.ts": "export const b = 2\n",
    }

    for relative, content in files.items():
        path = tmp_path / relative
        path.write_text(content, encoding="utf-8")

    records = []

    for relative in files:
        path = tmp_path / relative
        records.append(
            FileRecord(
                relative_path=relative,
                name=path.name,
                suffix=".ts",
                size_bytes=path.stat().st_size,
                category="source",
                language="TypeScript",
                is_indexable=True,
            )
        )

    index = RepositoryFileIndex(
        root_path=str(tmp_path),
        total_files=2,
        indexable_files=2,
        files=records,
        category_counts={"source": 2},
    )

    # Retrieval 只选中了 a.ts。
    selected = HybridRelevantFile(
        relative_path="src/a.ts",
        category="source",
        language="TypeScript",
        size_bytes=(tmp_path / "src/a.ts").stat().st_size,
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=selected.size_bytes,
        selected_files=[selected],
    )

    task = TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="修改 a",
        objective="修改 a，保持 b",
        source_step_order=1,
    )

    requirement = RequirementSpec(
        title="修改 a",
        summary="修改 a，保持 b",
    )

    technical_spec = TechnicalSpec(
        summary="验证 Context Policy",
    )

    repository = RepositoryProfile(
        project_name="demo",
        root_path=str(tmp_path),
        framework="Vue3",
        language="TypeScript",
    )

    original_bundle = ProjectContextBuilder().build(
        repository_root=tmp_path,
        requirement=requirement,
        technical_spec=technical_spec,
        task=task,
        repository=repository,
        selection=selection,
        max_files=8,
    )

    # Change Surface 要求阅读 b.ts，
    # 但它没有进入 Retrieval Selection。
    surface = ResolvedChangeSurface(
        task_id=task.task_id,
        implementation_targets=["src/a.ts"],
        dependencies=["src/b.ts"],
    )

    trace = JsonlHarnessTraceStore(
        path=tmp_path / "trace.jsonl",
    )

    coordinator = SimpleNamespace(
        artifact_resolver=TaskArtifactResolver(
            store=JsonTaskArtifactStore(
                root=tmp_path / "artifacts",
            )
        ),
        context_policy=RoleAwareContextPolicy(),
        trace_store=trace,
    )

    bridge = HarnessContextPolicyBridge(
        coordinator=coordinator,
        consumer=GuardedContextPolicyConsumer(config=ContextRolloutConfig(mode="off")),
    )

    result = SimpleNamespace(
        selection=selection,
        retrieval_selection=selection,
        repository_revision="revision-001",
    )

    return {
        "root": tmp_path,
        "task": task,
        "requirement": requirement,
        "technical_spec": technical_spec,
        "repository": repository,
        "index": index,
        "bundle": original_bundle,
        "surface": surface,
        "selection_result": result,
        "trace": trace,
        "bridge": bridge,
    }


def test_surface_dependency_is_added_to_real_context(
    bridge_case,
):
    case = bridge_case

    actual = case["bridge"].build_resolved_context(
        repository_root=case["root"],
        requirement=case["requirement"],
        technical_spec=case["technical_spec"],
        repository=case["repository"],
        index=case["index"],
        original_bundle=case["bundle"],
        selection_result=case["selection_result"],
        surface=case["surface"],
        raw_requirement=case["requirement"].summary,
        report={},
        get_current_revision=lambda: "revision-001",
    )

    paths = {item.relative_path for item in actual.bundle.code_context}

    assert paths == {"src/a.ts", "src/b.ts"}
    assert actual.decision.status == "baseline"
    assert actual.bundle.task.task_id == "TASK-001"

    # 只能改变阅读上下文，不能扩大 Change Surface。
    assert case["surface"].implementation_targets == ["src/a.ts"]
    assert case["surface"].dependencies == ["src/b.ts"]

    events = case["trace"].read_all()

    assert len(events) == 1
    assert events[0].event_type == "context_refined"
    assert set(events[0].data["context_files"]) == paths
    assert events[0].data["policy_status"] == "baseline"


def test_patch_plan_still_delegates_to_scope_guard():
    calls = []

    coordinator = SimpleNamespace(
        guard_patch_plan=lambda **kwargs: (calls.append(kwargs) or "guarded")
    )

    bridge = HarnessContextPolicyBridge(
        coordinator=coordinator,
        consumer=GuardedContextPolicyConsumer(config=ContextRolloutConfig(mode="off")),
    )

    context = SimpleNamespace(
        bundle=SimpleNamespace(task=SimpleNamespace(task_id="TASK-001"))
    )
    plan = SimpleNamespace(task_id="TASK-001")
    surface = ResolvedChangeSurface(task_id="TASK-001")

    result = bridge.guard_patch_plan(
        context=context,
        plan=plan,
        surface=surface,
    )

    assert result == "guarded"
    assert len(calls) == 1
    assert calls[0]["surface"] is surface


def test_completion_rejects_wrong_task():
    context = SimpleNamespace(
        bundle=SimpleNamespace(task=SimpleNamespace(task_id="TASK-A"))
    )

    class FakeCompletionGate:
        def complete(self, **kwargs):
            raise AssertionError("不应进入 Completion Gate")

    with pytest.raises(ValueError, match="Task ID"):
        HarnessContextPolicyBridge.complete_task(
            context=context,
            completion_gate=FakeCompletionGate(),
            proposal=SimpleNamespace(task_id="TASK-B"),
            journal=None,
            verification=None,
        )


def test_completion_delegates_when_identity_matches():
    calls = []

    class FakeCompletionGate:
        def complete(self, **kwargs):
            calls.append(kwargs)
            return "completed"

    context = SimpleNamespace(
        bundle=SimpleNamespace(task=SimpleNamespace(task_id="TASK-001"))
    )

    proposal = SimpleNamespace(task_id="TASK-001")

    result = HarnessContextPolicyBridge.complete_task(
        context=context,
        completion_gate=FakeCompletionGate(),
        proposal=proposal,
        journal=None,
        verification=None,
    )

    assert result == "completed"
    assert len(calls) == 1
    assert calls[0]["proposal"] is proposal
