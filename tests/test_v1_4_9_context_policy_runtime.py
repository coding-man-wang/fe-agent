from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
    StaleContextEvidenceError,
)
from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
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

REVISION = "revision-001"
RAW_REQUIREMENT = "修改现有页面"


@pytest.fixture
def policy_case(tmp_path):
    src = tmp_path / "src"
    src.mkdir()

    files = []

    for n in range(1, 7):
        path = src / f"f{n}.ts"
        path.write_text(
            f"export const value{n} = {n}\n",
            encoding="utf-8",
        )

        files.append(
            HybridRelevantFile(
                relative_path=f"src/f{n}.ts",
                category="source",
                language="TypeScript",
                size_bytes=path.stat().st_size,
                retrieval_score=1.0 / n,
                retrieval_rank=n,
                policy_score=0,
                selection_score=1.0 / n,
                reasons=(["Graph dependent: synthetic"] if n == 6 else []),
            )
        )

    selection = HybridRelevantFileSelection(
        candidate_count=6,
        eligible_count=6,
        selected_count=6,
        total_selected_bytes=sum(file.size_bytes for file in files),
        selected_files=files,
    )

    graph = RepositoryDependencyGraph(
        root_path=str(tmp_path),
        nodes=[],
        edges=[],
    )

    result = SimpleNamespace(
        repository_revision=REVISION,
        retrieval_selection=selection,
        selection=selection,
        graph=graph,
    )

    task = TaskDefinition(
        task_id="CASE-001",
        task_type="code_change",
        title="示例",
        objective=RAW_REQUIREMENT,
        source_step_order=1,
    )

    requirement = RequirementSpec(
        title="示例",
        summary=RAW_REQUIREMENT,
    )
    technical_spec = TechnicalSpec(
        summary="示例实现",
    )

    repository = RepositoryProfile(
        project_name="test-repo",
        root_path=str(tmp_path),
        language="TypeScript",
    )

    def build_context(current_selection):
        return ProjectContextBuilder().build(
            repository_root=tmp_path,
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            selection=current_selection,
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )

    evidence = capture_selection_evidence(
        repository_root=tmp_path,
        selection=selection,
    )

    report = {
        "policy_version": "v1.4.9-P1",
        "mode": "shadow",
        "task_id": task.task_id,
        "repository_revision": REVISION,
        "requirement_sha256": hashlib.sha256(
            RAW_REQUIREMENT.encode("utf-8")
        ).hexdigest(),
        "status": "complete",
        "applied": False,
        "selection_unchanged": True,
        "baseline_file_count": 6,
        "proposed_file_count": 5,
        "graph_proposed_drops": ["src/f6.ts"],
        "rerank_proposed_drops": [],
        "effective_proposed_drops": ["src/f6.ts"],
        "protected_paths": [f"src/f{n}.ts" for n in range(1, 6)],
        "evidence_binding": {
            "selection_sha256": evidence.selection_sha256,
            "source_sha256_before": evidence.source_sha256,
            "source_sha256_after": evidence.source_sha256,
            "source_matched": True,
            "selection_matched": True,
            "changed_paths": [],
        },
    }

    return {
        "root": tmp_path,
        "task": task,
        "result": result,
        "report": report,
        "build_context": build_context,
    }


def consume(
    case,
    *,
    config=None,
    audit_sink=None,
    build_context=None,
    report=None,
    extra=None,
    current_revision=REVISION,
    fresh=True,
):
    consumer = GuardedContextPolicyConsumer(
        config=config or ContextRolloutConfig(),
        audit_sink=audit_sink,
    )

    return consumer.consume(
        repository_root=case["root"],
        task=case["task"],
        raw_requirement=RAW_REQUIREMENT,
        selection_result=case["result"],
        report=report or case["report"],
        get_current_revision=lambda: current_revision,
        build_context=(build_context or case["build_context"]),
        additional_source_paths=extra,
        fresh_in_process_report=fresh,
    )


def test_shadow_validates_candidate_without_applying(
    policy_case,
):
    result = consume(policy_case)

    assert result.decision.status == "shadow"
    assert len(result.bundle.code_context) == 6
    assert result.decision.proposed_drops == ("src/f6.ts",)


def test_active_requires_explicit_approval(policy_case):
    audits = []

    config = ContextRolloutConfig(
        mode="active",
        active_enabled=True,
        canary_passed=True,
        approved_revision=REVISION,
        task_allowlist=frozenset({"CASE-001"}),
    )

    result = consume(
        policy_case,
        config=config,
        audit_sink=audits.append,
    )

    assert result.decision.status == "active"
    assert len(result.bundle.code_context) == 5
    assert "src/f6.ts" not in {
        item.relative_path for item in result.bundle.code_context
    }
    assert audits[0]["applied"] is True


def test_active_without_canary_falls_back(policy_case):
    audits = []

    config = ContextRolloutConfig(
        mode="active",
        active_enabled=True,
        canary_passed=False,
        approved_revision=REVISION,
        task_allowlist=frozenset({"CASE-001"}),
    )

    result = consume(
        policy_case,
        config=config,
        audit_sink=audits.append,
    )

    assert result.decision.status == "fallback"
    assert result.decision.reason == "active_not_authorized"
    assert len(result.bundle.code_context) == 6


def test_protected_file_cannot_be_dropped(policy_case):
    report = dict(policy_case["report"])
    report.update(
        {
            "graph_proposed_drops": ["src/f1.ts"],
            "effective_proposed_drops": ["src/f1.ts"],
        }
    )

    result = consume(
        policy_case,
        report=report,
    )

    assert result.decision.status == "fallback"
    assert result.decision.reason == ("protected_evidence_drop")
    assert len(result.bundle.code_context) == 6


def test_stale_source_requires_new_retrieval(policy_case):
    path = policy_case["root"] / "src" / "f1.ts"

    path.write_text(
        "export const value1 = 999\n",
        encoding="utf-8",
    )

    with pytest.raises(StaleContextEvidenceError):
        consume(policy_case)


def test_repository_revision_change_is_rejected(
    policy_case,
):
    with pytest.raises(StaleContextEvidenceError):
        consume(
            policy_case,
            current_revision="new-revision",
        )


def test_file_change_during_context_build_is_rejected(
    policy_case,
):
    path = policy_case["root"] / "src" / "f2.ts"
    original_build = policy_case["build_context"]

    calls = 0

    def mutating_build(selection):
        nonlocal calls
        calls += 1

        if calls == 1:
            path.write_text(
                "export const value2 = 200\n",
                encoding="utf-8",
            )

        return original_build(selection)

    with pytest.raises(
        StaleContextEvidenceError,
        match="source_changed_during_context_build",
    ):
        consume(
            policy_case,
            build_context=mutating_build,
        )


def test_additional_surface_file_change_is_detected(
    policy_case,
):
    extra_path = policy_case["root"] / "src" / "extra.ts"

    extra_path.write_text(
        "export const extra = 1\n",
        encoding="utf-8",
    )

    original_build = policy_case["build_context"]

    calls = 0

    def mutating_build(selection):
        nonlocal calls
        calls += 1

        if calls == 1:
            extra_path.write_text(
                "export const extra = 2\n",
                encoding="utf-8",
            )

        return original_build(selection)

    with pytest.raises(StaleContextEvidenceError):
        consume(
            policy_case,
            build_context=mutating_build,
            extra={"src/extra.ts"},
        )


def test_active_audit_failure_falls_back(policy_case):
    def failing_sink(_):
        raise OSError("audit unavailable")

    config = ContextRolloutConfig(
        mode="active",
        active_enabled=True,
        canary_passed=True,
        approved_revision=REVISION,
        task_allowlist=frozenset({"CASE-001"}),
    )

    result = consume(
        policy_case,
        config=config,
        audit_sink=failing_sink,
    )

    assert result.decision.status == "fallback"
    assert result.decision.reason == ("active_audit_unavailable")
    assert len(result.bundle.code_context) == 6


def test_off_mode_does_not_run_snapshot_gate(policy_case):
    result = consume(
        policy_case,
        config=ContextRolloutConfig(mode="off"),
        report={},
    )

    assert result.decision.status == "baseline"
    assert len(result.bundle.code_context) == 6
