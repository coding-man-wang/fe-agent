"""Exercise experimental Active through FullExecutionBenchmarkService.run.

The real context/consumer/audit path reaches an intercepted model boundary.
These tests do not claim real model, patch, verification or commit acceptance.
"""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

import agent_core.benchmark.full_execution as execution
from agent_core.benchmark.case_model import BenchmarkCase, BenchmarkExpectation
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.selection_evidence_binding import json_digest
from agent_core.change_surface.context_policy import RoleAwareContextPolicy, RoleBudget
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.task_context_selection import TaskContextSelectionResult
from agent_core.sdd.model import TaskPlan
from test_v1_4_9_full_execution_guarded_integration import coordinator_for
from test_v1_4_9_full_execution_guarded_trial import trial_case  # noqa: F401


def make_case(trial_case):
    return BenchmarkCase(
        case_id="CASE-006",
        title="Active integration",
        category="single_file_change",
        difficulty="easy",
        repository_name="fixture",
        base_commit_sha=trial_case["selection_result"].repository_revision,
        requirement=trial_case["raw_requirement"],
        expectation=BenchmarkExpectation(
            expected_files=["src/f1.ts"],
            expected_behaviors=["Preserve existing behavior"],
            verification_command_ids=["test:unit"],
        ),
    )


def identity_for(case, preflight):
    return {
        "case_sha256": json_digest(case.model_dump(mode="json")),
        "prepared_input_sha256": "c" * 64,
        "upstream_artifacts_sha256": "d" * 64,
        "model_config_sha256": "e" * 64,
        "canary_sha256": preflight.canary_sha256,
        "approved_revision": preflight.approved_revision,
    }


def run_arguments(trial_case, case):
    bundle = trial_case["final_baseline_bundle"]
    return dict(
        case=case,
        task_plan=TaskPlan(summary="Fixture", tasks=[bundle.task]),
        requirement=bundle.requirement,
        technical_spec=bundle.technical_spec,
        repository=bundle.repository,
        engineering_rules=[],
        task_surfaces={bundle.task.task_id: trial_case["execution_surface"]},
    )


def test_active_requires_explicit_preflight_before_runtime_initialization():
    with pytest.raises(ValueError, match="explicit preflight"):
        execution.FullExecutionBenchmarkService(
            fe_agent_root="unused",
            benchmark_repository="unused",
            baseline_run_id="unused",
            llm=object(),
            coordinator=object(),
            context_selection_service=object(),
            experimental_context_active=True,
        )


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None, [], {}])
def test_active_constructor_rejects_non_boolean_flags(flag):
    # Supply the other dependencies so truthy values cannot pass merely because
    # an unrelated missing-preflight check happens to reject them.
    with pytest.raises((TypeError, ValueError)):
        execution.FullExecutionBenchmarkService(
            fe_agent_root="unused",
            benchmark_repository="unused",
            baseline_run_id="unused",
            llm=object(),
            coordinator=object(),
            context_selection_service=object(),
            experimental_context_active=flag,
            context_guarded_preflight=SimpleNamespace(status="passed"),
            context_shadow_observer=object(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "case_id",
        "base_commit_sha",
        "preflight",
        "case_sha256",
        "canary_sha256",
        "approved_revision",
        "missing_identity",
    ],
)
async def test_active_run_rejects_wrong_identity_before_any_work(
    trial_case, monkeypatch, invalid
):
    case = make_case(trial_case)
    preflight = trial_case["preflight"]
    identity = identity_for(case, preflight)
    if invalid in {"case_id", "base_commit_sha"}:
        case = case.model_copy(
            update={invalid: "CASE-005" if invalid == "case_id" else "f" * 40}
        )
    elif invalid == "preflight":
        preflight = replace(preflight, status="denied")
    elif invalid == "missing_identity":
        identity.pop("prepared_input_sha256")
    else:
        identity[invalid] = "0" * (40 if invalid == "approved_revision" else 64)

    def unexpected_work(*args, **kwargs):
        raise AssertionError("A rejected experiment must not initialize execution")

    service = object.__new__(execution.FullExecutionBenchmarkService)
    service.context_guarded_preflight = preflight
    service.experimental_context_active = True
    service.context_experiment_identity = identity
    service.store = SimpleNamespace(save=unexpected_work)
    monkeypatch.setattr(execution, "ExecutionPlanCompressor", unexpected_work)
    monkeypatch.setattr(execution, "ReadOnlyCodingAgent", unexpected_work)
    with pytest.raises(
        execution.FullExecutionBenchmarkError, match="does not match this run"
    ):
        await service.run(**run_arguments(trial_case, case))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "active",
        "default",
        "audit_failure",
        "source_changed",
        "render_mutation",
        "review_source_changed",
    ],
)
async def test_service_consumes_only_authorized_fresh_candidate(
    trial_case, monkeypatch, mode
):
    root = trial_case["repository_root"]
    base = trial_case["final_baseline_bundle"]
    selected = trial_case["selection_result"]
    revision = selected.repository_revision
    case = make_case(trial_case)
    result = TaskContextSelectionResult(
        repository_revision=revision,
        index=RepositoryFileIndexer().build(root),
        retrieval_selection=selected.retrieval_selection,
        graph=selected.graph,
        selection=selected.selection,
        retrieval_mode="lexical_only",
        semantic_error=None,
    )
    calls = {"retrieval": 0, "observer": 0, "model": 0, "review": 0, "removed": 0}
    consumed = []

    class Selection:
        async def select(self, **kwargs):
            calls["retrieval"] += 1
            return deepcopy(result)

    class Snapshot:
        def __init__(self, **kwargs):
            pass

        async def create(self, **kwargs):
            return root

        async def remove(self, *args, **kwargs):
            calls["removed"] += 1

    class Runner:
        async def observe(self, **kwargs):
            calls["observer"] += 1
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
        calls["model"] += 1
        consumed.append(bundle.model_copy(deep=True))
        if mode == "review_source_changed":
            (root / "src/f6.ts").write_text(
                "export const changed_after_analysis = 99;\n", encoding="utf-8"
            )
            return SimpleNamespace(proposed_changes=["src/f1.ts"])
        raise RuntimeError("MODEL_BOUNDARY_REACHED")

    agent_root = root.parent / (root.name + "-active-agent")
    agent_root.mkdir()
    coordinator = coordinator_for(agent_root)
    coordinator.context_policy = RoleAwareContextPolicy(
        default_budget=RoleBudget(context_only=6)
    )

    async def resolve_surface(**kwargs):
        return trial_case["execution_surface"].model_copy(
            deep=True, update={"execution_disposition": "change_required"}
        )

    monkeypatch.setattr(coordinator, "resolve_change_surface", resolve_surface)
    monkeypatch.setattr(execution, "CommittedSnapshotWorktreeManager", Snapshot)
    # Actual Git cleanliness/HEAD checks have independent integration tests.
    monkeypatch.setattr(execution, "_read_clean_snapshot_revision", lambda _: revision)
    monkeypatch.setattr(
        execution.ReadOnlyCodingAgent, "analyze_prepared", stop_at_model
    )
    if mode == "review_source_changed":
        # Isolate the follow-up model boundary; guard enforcement remains real.
        monkeypatch.setattr(
            execution.RuntimeExecutionDecisionPolicy,
            "decide",
            lambda *args, **kwargs: SimpleNamespace(action="continue"),
        )
        monkeypatch.setattr(
            execution.ExecutionObligationGate,
            "evaluate",
            lambda *args, **kwargs: SimpleNamespace(
                allowed=False, unresolved_obligation_ids=["OBL-001"]
            ),
        )
        monkeypatch.setattr(
            execution.ReadOnlyCodingAgent,
            "prepare_execution_obligation_review_messages",
            lambda self, **kwargs: kwargs["messages"],
        )

        async def forbidden_review(self, **kwargs):
            calls["review"] += 1
            raise AssertionError("Stale context must not reach obligation review")

        monkeypatch.setattr(
            execution.ReadOnlyCodingAgent,
            "review_execution_obligations",
            forbidden_review,
        )
    if mode == "render_mutation":
        prepare = execution.ReadOnlyCodingAgent.prepare_messages

        def prepare_then_mutate(self, bundle):
            messages = prepare(self, bundle)
            bundle.code_context[0].content += "\n// unexpected mutation"
            return messages

        monkeypatch.setattr(
            execution.ReadOnlyCodingAgent, "prepare_messages", prepare_then_mutate
        )
    if mode == "audit_failure":

        def fail_audit(self, report):
            raise OSError("audit store unavailable")

        monkeypatch.setattr(execution.JsonlShadowAuditStore, "append", fail_audit)

    enabled = mode != "default"
    service = execution.FullExecutionBenchmarkService(
        fe_agent_root=agent_root,
        benchmark_repository=root,
        baseline_run_id="fixture",
        llm=object(),
        coordinator=coordinator,
        context_selection_service=Selection(),
        context_shadow_observer=(
            FullExecutionContextShadowObserver(runner=Runner()) if enabled else None
        ),
        context_guarded_preflight=trial_case["preflight"] if enabled else None,
        experimental_context_active=enabled,
        context_experiment_identity=(
            identity_for(case, trial_case["preflight"]) if enabled else None
        ),
    )
    monkeypatch.setattr(service, "_create_run_scoped_coordinator", lambda: coordinator)
    record = await service.run(**run_arguments(trial_case, case))

    assert calls["retrieval"] == calls["removed"] == 1
    assert record.status == "failed"  # An intentional interception, not an E2E pass.
    task_record = record.tasks[0]
    report = task_record.context_guarded_active
    if mode in {"source_changed", "render_mutation", "review_source_changed"}:
        assert calls["model"] == (
            1 if mode == "review_source_changed" else 0
        ), record.error
        assert calls["review"] == 0
        assert report["status"] == "blocked"
        assert report["applied"] is False
        events = [
            json.loads(line)
            for line in (service.run_root / "context_guarded_active.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert events[-1]["status"] == "blocked"
        assert events[-1]["applied"] is False
        if mode in {"render_mutation", "review_source_changed"}:
            assert events[0]["status"] == "active"
            assert report["reason"] == "changed_before_model_consumption"
        return

    assert calls["model"] == 1, record.error
    assert "MODEL_BOUNDARY_REACHED" in record.error
    paths = [item.relative_path for item in consumed[0].code_context]
    assert task_record.selected_context_files == paths
    assert consumed[0].change_surface["implementation_targets"] == ["src/f1.ts"]
    if mode == "active":
        assert calls["observer"] == 1
        assert report["status"] == "active"
        assert report["applied"] is True
        assert report["production_ready"] is False
        assert "src/f6.ts" not in paths and len(paths) == 6
        assert report["consumed_bundle_sha256"] == json_digest(
            consumed[0].model_dump(mode="json")
        )
        assert record.context_policy_mode == "active_experiment"
        assert record.context_experiment_identity == identity_for(
            case, trial_case["preflight"]
        )
    else:
        assert "src/f6.ts" in paths and len(paths) == 7
        if mode == "audit_failure":
            assert report["status"] == "fallback"
            assert report["applied"] is False
            assert report["reason"] == "active_audit_unavailable"
        else:
            assert report is None
            assert service.experimental_context_active is False
            assert calls["observer"] == 0
            assert record.context_policy_mode == "baseline"
