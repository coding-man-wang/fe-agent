"""Real builder and service-boundary tests; no external LLM or patch execution."""

from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

import agent_core.benchmark.full_execution as execution
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.case_model import BenchmarkCase, BenchmarkExpectation
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.obligation_evidence import ObligationEvidenceEnricher
from agent_core.benchmark.selection_evidence_binding import json_digest
from agent_core.benchmark.verification_strategy import VerificationStrategyResolver
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.change_surface.context_policy import RoleAwareContextPolicy, RoleBudget
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.model import HybridRelevantFile, HybridRelevantFileSelection
from agent_core.codebase.task_context_selection import TaskContextSelectionResult
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.sdd.model import TaskPlan
from test_obligation_evidence import create_bundle, create_repository
from test_v1_4_9_full_execution_guarded_trial import trial_case  # noqa: F401


def coordinator_for(root):
    return HarnessV1Coordinator(
        store=JsonTaskArtifactStore(root=root / "artifacts"),
        investigation_builder=object(),
        surface_resolver=object(),
    )


@pytest.fixture
def final_builder_case(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    create_repository(root)
    base = create_bundle(root)
    path = root / "src/api/orders.ts"
    selected = HybridRelevantFile(
        relative_path="src/api/orders.ts",
        category="source",
        language="TypeScript",
        size_bytes=path.stat().st_size,
        retrieval_score=1,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1,
    )
    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=selected.size_bytes,
        selected_files=[selected],
    )
    return dict(
        coordinator=coordinator_for(tmp_path),
        repository_root=root,
        requirement=base.requirement,
        technical_spec=base.technical_spec,
        repository=base.repository,
        index=RepositoryFileIndexer().build(root),
        base_bundle=base,
        selection=selection,
        execution_surface=ResolvedChangeSurface.model_validate(base.change_surface),
        engineering_rules=["Preserve existing public behavior"],
        task_plan=TaskPlan(summary="Order filtering", tasks=[base.task]),
        obligation_enricher=ObligationEvidenceEnricher(max_added_files=1),
        strategy_resolver=VerificationStrategyResolver(),
    )


def test_shared_final_builder_adds_real_obligation_evidence_without_write_permission(
    final_builder_case,
):
    args = final_builder_case
    original_selection = args["selection"].model_dump(mode="json")
    original_surface = args["execution_surface"].model_dump(mode="json")
    first, enrichment, strategy = execution._build_final_execution_context(**args)
    rebuilt, _, _ = execution._build_final_execution_context(**args)

    assert enrichment.added_files == ["src/api/users.ts"]
    assert {item.relative_path for item in first.code_context} == {
        "src/api/orders.ts",
        "src/api/users.ts",
    }
    assert first.change_surface == original_surface
    assert "src/api/users.ts" not in first.change_surface["implementation_targets"]
    assert strategy is None
    assert first.model_dump(mode="json") == rebuilt.model_dump(mode="json")
    assert args["selection"].model_dump(mode="json") == original_selection


def test_shared_final_builder_resolves_real_strategy_after_enrichment(
    final_builder_case,
):
    args = final_builder_case
    task = args["base_bundle"].task.model_copy(
        deep=True,
        update={"task_type": "test", "verification": ["Verify remote error behavior"]},
    )
    args["base_bundle"] = args["base_bundle"].model_copy(
        deep=True, update={"task": task}
    )
    args["task_plan"] = TaskPlan(summary="Verify API behavior", tasks=[task])
    args["execution_surface"] = args["execution_surface"].model_copy(
        deep=True,
        update={
            "tests": ["tests/unit/ordersApi.test.ts"],
            "create_files": ["tests/unit/ordersApi.test.ts"],
        },
    )
    seen = []
    real_resolver = VerificationStrategyResolver()

    class RecordingResolver:
        def resolve(self, **kwargs):
            seen.append([item.relative_path for item in kwargs["code_context"]])
            return real_resolver.resolve(**kwargs)

    args["strategy_resolver"] = RecordingResolver()
    bundle, enrichment, strategy = execution._build_final_execution_context(**args)

    assert enrichment.added_files == ["src/api/users.ts"]
    assert "src/api/users.ts" in seen[0]
    assert strategy is not None
    assert strategy.writable_test_files == ["tests/unit/ordersApi.test.ts"]
    assert bundle.verification_strategy == strategy.model_dump(mode="json")
    assert bundle.change_surface == args["execution_surface"].model_dump(mode="json")


def test_final_builder_failure_propagates_instead_of_emitting_incomplete_context(
    final_builder_case,
):
    class BrokenEnricher:
        def enrich(self, **kwargs):
            raise RuntimeError("required evidence could not be built")

    final_builder_case["obligation_enricher"] = BrokenEnricher()
    with pytest.raises(RuntimeError, match="required evidence"):
        execution._build_final_execution_context(**final_builder_case)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def committed_repository(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "tracked.ts").write_text("export const value = 1;\n", encoding="utf-8")
    git(tmp_path, "add", "tracked.ts")
    git(
        tmp_path,
        "-c",
        "user.name=Preview Test",
        "-c",
        "user.email=preview@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "fixture",
    )
    return tmp_path


def test_live_revision_reads_committed_head(committed_repository):
    assert execution._read_clean_snapshot_revision(committed_repository) == git(
        committed_repository, "rev-parse", "HEAD"
    )


@pytest.mark.parametrize("path", ["tracked.ts", "new-evidence.ts"])
def test_live_revision_rejects_tracked_and_untracked_changes(
    committed_repository, path
):
    (committed_repository / path).write_text(
        "export const value = 2;\n", encoding="utf-8"
    )
    with pytest.raises(execution.FullExecutionBenchmarkError, match="dirty"):
        execution._read_clean_snapshot_revision(committed_repository)


def test_live_revision_rejects_head_race(monkeypatch, tmp_path):
    outputs = iter(["a" * 40, "", "b" * 40])
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    with pytest.raises(execution.FullExecutionBenchmarkError, match="revision changed"):
        execution._read_clean_snapshot_revision(tmp_path)


def test_service_defaults_leave_guarded_preview_disabled(tmp_path):
    service = execution.FullExecutionBenchmarkService(
        fe_agent_root=tmp_path,
        benchmark_repository=tmp_path,
        baseline_run_id="fixture",
        llm=object(),
        coordinator=object(),
        context_selection_service=object(),
    )
    assert service.context_guarded_preflight is None
    assert service.context_shadow_observer is None


def test_service_rejects_guarded_preview_without_observer(trial_case):
    with pytest.raises(ValueError, match="Shadow observer"):
        execution.FullExecutionBenchmarkService(
            fe_agent_root=trial_case["repository_root"],
            benchmark_repository=trial_case["repository_root"],
            baseline_run_id="fixture",
            llm=object(),
            coordinator=object(),
            context_selection_service=object(),
            context_guarded_preflight=trial_case["preflight"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["case_id", "base_commit_sha"])
async def test_run_identity_rejected_before_planning_storage_or_model(
    trial_case,
    monkeypatch,
    mismatch,
):
    baseline = trial_case["final_baseline_bundle"]
    revision = trial_case["selection_result"].repository_revision
    case = BenchmarkCase(
        case_id="CASE-006",
        title="Identity gate",
        category="single_file_change",
        difficulty="easy",
        repository_name="fixture",
        base_commit_sha=revision,
        requirement=trial_case["raw_requirement"],
        expectation=BenchmarkExpectation(
            expected_files=["src/f1.ts"],
            expected_behaviors=["Preserve behavior"],
            verification_command_ids=["test:unit"],
        ),
    ).model_copy(
        update={
            mismatch: "CASE-005" if mismatch == "case_id" else "c" * 40,
        }
    )

    def unexpected_work(*args, **kwargs):
        raise AssertionError("Rejected identity must not start execution")

    service = object.__new__(execution.FullExecutionBenchmarkService)
    service.context_guarded_preflight = trial_case["preflight"]
    service.store = SimpleNamespace(save=unexpected_work)
    service._create_run_scoped_coordinator = unexpected_work
    monkeypatch.setattr(execution, "ExecutionPlanCompressor", unexpected_work)
    monkeypatch.setattr(execution, "ReadOnlyCodingAgent", unexpected_work)
    with pytest.raises(
        execution.FullExecutionBenchmarkError, match="does not match this run"
    ):
        await service.run(
            case=case,
            task_plan=TaskPlan(summary="Fixture", tasks=[baseline.task]),
            requirement=baseline.requirement,
            technical_spec=baseline.technical_spec,
            repository=baseline.repository,
            engineering_rules=[],
            task_surfaces={baseline.task.task_id: trial_case["execution_surface"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["off", "preview", "source_changed", "message_bundle_changed"]
)
async def test_real_service_loop_preserves_baseline_or_blocks_before_model(
    trial_case,
    monkeypatch,
    mode,
):
    """Exercise run(), stopping deliberately at the first model call.

    Retrieval and snapshot materialization are fixtures. The final context
    builder, observer, Preview, Trial, audit and service branch are real.
    This tests the consumption boundary, not Patch/Verification/Commit E2E.
    """
    root = trial_case["repository_root"]
    baseline = trial_case["final_baseline_bundle"]
    surface = trial_case["execution_surface"]
    revision = trial_case["selection_result"].repository_revision
    selected = trial_case["selection_result"]
    selection_result = TaskContextSelectionResult(
        repository_revision=revision,
        index=RepositoryFileIndexer().build(root),
        retrieval_selection=selected.retrieval_selection,
        graph=selected.graph,
        selection=selected.selection,
        retrieval_mode="lexical_only",
        semantic_error=None,
    )
    counts = {"retrieval": 0, "observer": 0, "model": 0, "removed": 0}
    consumed = []

    class SelectionService:
        async def select(self, **kwargs):
            counts["retrieval"] += 1
            return deepcopy(selection_result)

    class SnapshotManager:
        def __init__(self, **kwargs):
            pass

        async def create(self, **kwargs):
            return root

        async def remove(self, *args, **kwargs):
            counts["removed"] += 1

    class Runner:
        async def observe(self, **kwargs):
            counts["observer"] += 1
            if mode == "source_changed":
                (root / "src/f6.ts").write_text(
                    "export const changed = 99;\n", encoding="utf-8"
                )
            return {
                "status": "complete",
                "baseline_file_count": 7,
                "proposed_file_count": 6,
                "graph_proposed_drops": ["src/f6.ts"],
                "rerank_proposed_drops": [],
                "effective_proposed_drops": ["src/f6.ts"],
            }

    async def stop_at_model(self, *, bundle, messages):
        counts["model"] += 1
        consumed.append(bundle.model_copy(deep=True))
        raise RuntimeError("MODEL_BOUNDARY_REACHED")

    agent_root = root.parent / (root.name + "-agent")
    agent_root.mkdir()
    coordinator = coordinator_for(agent_root)
    # Permit the seven-file fixture to reach the real final builder so the
    # removable sixth item remains present at the consumption boundary.
    coordinator.context_policy = RoleAwareContextPolicy(
        default_budget=RoleBudget(context_only=6)
    )

    async def resolve_surface(**kwargs):
        return surface.model_copy(
            deep=True, update={"execution_disposition": "change_required"}
        )

    monkeypatch.setattr(coordinator, "resolve_change_surface", resolve_surface)
    monkeypatch.setattr(execution, "CommittedSnapshotWorktreeManager", SnapshotManager)
    monkeypatch.setattr(execution, "_read_clean_snapshot_revision", lambda _: revision)
    monkeypatch.setattr(
        execution.ReadOnlyCodingAgent, "analyze_prepared", stop_at_model
    )
    if mode == "message_bundle_changed":
        original_prepare = execution.ReadOnlyCodingAgent.prepare_messages

        def prepare_then_mutate(self, bundle):
            messages = original_prepare(self, bundle)
            bundle.engineering_rules.append("Unexpected rule after rendering")
            return messages

        monkeypatch.setattr(
            execution.ReadOnlyCodingAgent,
            "prepare_messages",
            prepare_then_mutate,
        )
    service = execution.FullExecutionBenchmarkService(
        fe_agent_root=agent_root,
        benchmark_repository=root,
        baseline_run_id="fixture",
        llm=object(),
        coordinator=coordinator,
        context_selection_service=SelectionService(),
        context_shadow_observer=(
            FullExecutionContextShadowObserver(runner=Runner())
            if mode != "off"
            else None
        ),
        context_guarded_preflight=trial_case["preflight"] if mode != "off" else None,
    )
    monkeypatch.setattr(service, "_create_run_scoped_coordinator", lambda: coordinator)
    case = BenchmarkCase(
        case_id="CASE-006",
        title="Guarded integration",
        category="single_file_change",
        difficulty="easy",
        repository_name="fixture",
        base_commit_sha=revision,
        requirement=trial_case["raw_requirement"],
        expectation=BenchmarkExpectation(
            expected_files=["src/f1.ts"],
            expected_behaviors=["Preserve behavior"],
            verification_command_ids=["test:unit"],
        ),
    )
    record = await service.run(
        case=case,
        task_plan=TaskPlan(summary="Fixture task", tasks=[baseline.task]),
        requirement=baseline.requirement,
        technical_spec=baseline.technical_spec,
        repository=baseline.repository,
        engineering_rules=[],
        task_surfaces={baseline.task.task_id: surface},
    )
    assert counts["retrieval"] == 1
    assert counts["removed"] == 1
    assert record.status == "failed"  # Intentional stop, never a claimed E2E success.
    if mode == "source_changed":
        assert counts["model"] == 0, record.error
        assert "Guarded Preview blocked" in record.error
        assert record.tasks[0].context_guarded_preview["status"] == "blocked"
    elif mode == "message_bundle_changed":
        assert counts["model"] == 0, record.error
        assert "changed before model consumption" in record.error
        report = record.tasks[0].context_guarded_preview
        assert report["status"] == "blocked"
        assert report["reason"] == "changed_before_model_consumption"
        assert report["snapshot_matched"] is False
        assert report["audit_recorded"] is True
        events = [
            json.loads(line)
            for line in (service.run_root / "context_guarded_preview.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert events[0]["status"] == "candidate_valid"
        assert events[-1]["status"] == "blocked"
        assert events[-1]["reason"] == "changed_before_model_consumption"
        assert events[-1]["applied"] is False
    else:
        assert counts["model"] == 1, record.error
        assert "MODEL_BOUNDARY_REACHED" in record.error
        assert "src/f6.ts" in {item.relative_path for item in consumed[0].code_context}
        if mode == "preview":
            report = record.tasks[0].context_guarded_preview
            assert report["status"] == "candidate_valid"
            assert report["removed_files"] == ("src/f6.ts",)
            assert report["applied"] is False
            assert report["active_execution_enabled"] is False
            assert (
                json_digest(consumed[0].model_dump(mode="json"))
                == report["baseline_bundle_sha256"]
            )
            events = [
                json.loads(line)
                for line in (service.run_root / "context_guarded_preview.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert events[-1]["applied"] is False
        else:
            assert counts["observer"] == 0
            assert record.tasks[0].context_guarded_preview is None
            assert not (service.run_root / "context_guarded_preview.jsonl").exists()
