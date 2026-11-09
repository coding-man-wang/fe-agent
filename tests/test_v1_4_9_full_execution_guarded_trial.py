from dataclasses import replace
from types import SimpleNamespace

import hashlib
import pytest

from agent_core.benchmark.context_policy_runtime import (
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.full_execution_active_preflight import (
    ActivePreflightDecision,
)
from agent_core.benchmark.full_execution_active_preview import (
    ActiveCandidatePreview,
)
from agent_core.benchmark.full_execution_guarded_trial import (
    run_guarded_active_trial,
)
from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
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

REVISION = "a" * 40
RAW_REQUIREMENT = "修改页面并保持原有功能"


@pytest.fixture
def trial_case(tmp_path):
    src = tmp_path / "src"
    src.mkdir()

    selected = []

    for number in range(1, 8):
        path = src / f"f{number}.ts"

        path.write_text(
            f"export const value{number} = {number}\n",
            encoding="utf-8",
        )

        selected.append(
            HybridRelevantFile(
                relative_path=f"src/f{number}.ts",
                category="source",
                language="TypeScript",
                size_bytes=path.stat().st_size,
                retrieval_score=1.0 / number,
                retrieval_rank=number,
                policy_score=0,
                selection_score=1.0 / number,
                reasons=(["Graph dependent: synthetic"] if number == 6 else []),
            )
        )

    selection = HybridRelevantFileSelection(
        candidate_count=6,
        eligible_count=6,
        selected_count=6,
        total_selected_bytes=sum(item.size_bytes for item in selected),
        selected_files=selected,
    )

    task = TaskDefinition(
        task_id="TASK-002",
        task_type="code_change",
        title="修改页面",
        objective=RAW_REQUIREMENT,
        source_step_order=1,
    )

    requirement = RequirementSpec(
        title="修改页面",
        summary=RAW_REQUIREMENT,
    )
    technical_spec = TechnicalSpec(
        summary="保持兼容",
    )
    repository = RepositoryProfile(
        project_name="trial",
        root_path=str(tmp_path),
        language="TypeScript",
    )

    surface = ResolvedChangeSurface(
        task_id=task.task_id,
        implementation_targets=["src/f1.ts"],
    )

    def build_final_context(current_selection):
        # 当前 Fixture 没有 Execution Obligations，
        # 因此 Enrichment 是 no-op；
        # code_change 不需要 Verification Strategy。
        bundle = ProjectContextBuilder().build(
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
        return bundle.model_copy(
            deep=True,
            update={"change_surface": surface.model_dump(mode="json")},
        )

    baseline = build_final_context(selection)

    graph = RepositoryDependencyGraph(
        root_path=str(tmp_path),
        nodes=[],
        edges=[],
    )

    selection_result = SimpleNamespace(
        repository_revision=REVISION,
        selection=selection,
        retrieval_selection=selection,
        graph=graph,
    )

    fingerprint = capture_selection_evidence(
        repository_root=tmp_path,
        selection=selection,
    )

    p1_report = {
        "policy_version": "v1.4.9-P1",
        "mode": "shadow",
        "status": "complete",
        "applied": False,
        "task_id": task.task_id,
        "repository_revision": REVISION,
        "requirement_sha256": hashlib.sha256(
            RAW_REQUIREMENT.encode("utf-8")
        ).hexdigest(),
        "selection_unchanged": True,
        "baseline_file_count": 7,
        "proposed_file_count": 6,
        "graph_proposed_drops": ["src/f6.ts"],
        "rerank_proposed_drops": [],
        "effective_proposed_drops": ["src/f6.ts"],
        "evidence_binding": {
            "selection_sha256": (fingerprint.selection_sha256),
            "source_sha256_before": (fingerprint.source_sha256),
            "source_sha256_after": (fingerprint.source_sha256),
            "source_matched": True,
            "selection_matched": True,
            "changed_paths": [],
        },
    }

    preflight = ActivePreflightDecision(
        version="v1.4.9-P4-B-D2-A",
        status="passed",
        reason="active_preflight_valid",
        case_id="CASE-006",
        task_id=task.task_id,
        approved_revision=REVISION,
        current_revision=REVISION,
        canary_sha256="b" * 64,
    )

    preview = ActiveCandidatePreview(
        status="preview_valid",
        reason="candidate_snapshot_and_protection_valid",
        proposed_selection_drops=("src/f6.ts",),
        drops_in_final_context=("src/f6.ts",),
        p1_report=p1_report,
    )

    return {
        "repository_root": tmp_path,
        "raw_requirement": RAW_REQUIREMENT,
        "selection_result": selection_result,
        "final_baseline_bundle": baseline,
        "execution_surface": surface,
        "obligation_evidence_files": [],
        "preflight": preflight,
        "preview": preview,
        "get_current_revision": lambda: REVISION,
        "build_final_context": build_final_context,
    }


def test_valid_trial_does_not_apply_context(trial_case):
    baseline = trial_case["final_baseline_bundle"]
    original = baseline.model_dump(mode="json")

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "candidate_valid"
    assert result.removed_files == ("src/f6.ts",)
    assert result.candidate_chars < result.baseline_chars

    assert result.applied is False
    assert result.production_ready is False
    assert baseline.model_dump(mode="json") == original


def test_unapproved_preflight_falls_back(trial_case):
    trial_case["preflight"] = replace(
        trial_case["preflight"],
        status="denied",
    )

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "fallback"
    assert result.reason == "preflight_not_eligible"


def test_invalid_preview_falls_back(trial_case):
    trial_case["preview"] = replace(
        trial_case["preview"],
        status="abstained",
        p1_report=None,
    )

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "fallback"
    assert result.reason == "candidate_preview_invalid"


def test_obligation_evidence_cannot_be_dropped(trial_case):
    # 将原本可移除的 f6 标记为必要履约证据。
    trial_case["obligation_evidence_files"] = ["src/f6.ts"]

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "fallback"
    assert result.reason == "mandatory_evidence_drop"


def test_stale_source_blocks_trial(trial_case):
    path = trial_case["repository_root"] / "src/f6.ts"

    path.write_text(
        "export const value6 = 999\n",
        encoding="utf-8",
    )

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "blocked"
    assert result.reason == "stale_context_evidence"
    assert result.applied is False


def test_rebuild_must_match_final_baseline(trial_case):
    original_builder = trial_case["build_final_context"]

    def incorrect_builder(selection):
        bundle = original_builder(selection)
        return bundle.model_copy(
            deep=True,
            update={
                "engineering_rules": ["unexpected rule"],
            },
        )

    trial_case["build_final_context"] = incorrect_builder

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "fallback"
    assert result.reason == "baseline_rebuild_mismatch"


def test_candidate_must_not_drop_extra_files(trial_case):
    original_builder = trial_case["build_final_context"]

    def incorrect_builder(selection):
        bundle = original_builder(selection)

        # 仅在候选中额外删除 f5。
        if len(selection.selected_files) == 7:
            return bundle

        items = [
            item for item in bundle.code_context if item.relative_path != "src/f7.ts"
        ]

        budget = bundle.budget.model_copy(
            deep=True,
            update={
                "used_files": len(items),
                "used_chars": sum(len(item.content) for item in items),
            },
        )

        return bundle.model_copy(
            deep=True,
            update={
                "code_context": items,
                "budget": budget,
            },
        )

    trial_case["build_final_context"] = incorrect_builder

    result = run_guarded_active_trial(**trial_case)

    assert result.status == "fallback"
    assert result.reason == "candidate_drop_mismatch"
    assert result.applied is False
