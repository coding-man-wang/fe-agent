from dataclasses import replace

import pytest

from agent_core.benchmark.context_policy_runtime import StaleContextEvidenceError
from agent_core.benchmark.full_execution_guarded_active import (
    run_full_execution_guarded_active,
)
from agent_core.benchmark.full_execution_guarded_trial import run_guarded_active_trial
from agent_core.benchmark.selection_evidence_binding import json_digest
from test_v1_4_9_full_execution_guarded_preview import (
    bridge_case,
    change_source,
)  # noqa: F401
from test_v1_4_9_full_execution_guarded_trial import trial_case  # noqa: F401


@pytest.fixture
def active_case(bridge_case):
    args, runner, events = bridge_case
    args["experimental_active_enabled"] = True
    return args, runner, events


@pytest.mark.asyncio
async def test_active_consumes_once_after_audit_and_keeps_original(active_case):
    args, runner, events = active_case
    original = args["final_baseline_bundle"].model_dump(mode="json")
    builder = args["build_final_context"]
    builds = []

    def build(selection):
        builds.append(len(selection.selected_files))
        return builder(selection)

    args["build_final_context"] = build
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "active"
    assert result.report.applied and result.report.active_execution_enabled
    assert result.report.production_ready is False
    assert result.report.audit_recorded
    assert result.report.snapshot_matched
    assert all(
        (
            result.report.mandatory_context_complete,
            result.report.retained_symbol_evidence,
            result.report.metadata_unchanged,
            result.report.scope_unchanged,
        )
    )
    assert result.bundle is not args["final_baseline_bundle"]
    assert "src/f6.ts" not in [
        item.relative_path for item in result.bundle.code_context
    ]
    assert args["final_baseline_bundle"].model_dump(mode="json") == original
    assert builds == [7, 6]
    assert runner.calls == 1
    assert result.report.consumed_bundle_sha256 == json_digest(
        result.bundle.model_dump(mode="json")
    )
    assert result.report.consumed_bundle_sha256 != result.report.baseline_bundle_sha256
    assert events[-1]["applied"] is True
    assert events[-1]["production_ready"] is False
    assert "export const" not in str(events)
    result.assert_fresh()


@pytest.mark.asyncio
async def test_active_requires_explicit_opt_in(active_case):
    args, runner, _ = active_case
    args.pop("experimental_active_enabled")
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "fallback"
    assert result.report.reason == "experimental_active_disabled"
    assert result.bundle is args["final_baseline_bundle"]
    assert not result.report.applied
    assert runner.calls == 0
    result.assert_fresh()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"case_id": "CASE-001"},
        {"task_id": "TASK-003"},
        {"approved_revision": "b" * 40},
        {"current_revision": "b" * 40},
        {"canary_sha256": None},
        {"status": "denied"},
        {"production_ready": True},
        {"applied": True},
        {"active_execution_enabled": True},
    ],
)
async def test_preflight_cannot_authorize_other_scope(active_case, changed):
    args, runner, _ = active_case
    args["preflight"] = replace(args["preflight"], **changed)
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "fallback"
    assert result.report.reason == "preflight_not_eligible"
    assert result.bundle is args["final_baseline_bundle"]
    assert not result.report.applied
    assert runner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "error", "async", "false"])
async def test_no_candidate_release_without_synchronous_audit(active_case, kind):
    args, _, _ = active_case

    def fail(event):
        raise OSError("disk full")

    async def deferred(event):
        raise AssertionError("must not schedule detached audit")

    args["audit_sink"] = {
        "missing": None,
        "error": fail,
        "async": deferred,
        "false": lambda event: False,
    }[kind]
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "fallback"
    assert result.report.reason == "active_audit_unavailable"
    assert result.bundle is args["final_baseline_bundle"]
    assert result.report.applied is result.report.active_execution_enabled is False
    assert result.report.audit_recorded is False
    result.assert_fresh()


@pytest.mark.asyncio
async def test_audit_source_mutation_blocks_instead_of_old_baseline(active_case):
    args, _, _ = active_case

    def audit(event):
        change_source(args)
        raise OSError("write failed")

    args["audit_sink"] = audit
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "blocked"
    assert result.bundle is None
    assert result.report.applied is False
    with pytest.raises(StaleContextEvidenceError):
        result.assert_fresh()


@pytest.mark.asyncio
async def test_audit_cannot_mutate_validated_candidate_via_builder_reference(
    active_case,
):
    args, _, _ = active_case
    builder = args["build_final_context"]
    built = []

    def build(selection):
        bundle = builder(selection)
        built.append(bundle)
        return bundle

    args["build_final_context"] = build
    args["audit_sink"] = lambda event: built[-1].engineering_rules.append("changed")
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "blocked"
    assert result.bundle is None
    assert not result.report.applied


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["source", "baseline", "selection", "surface", "candidate", "revision"]
)
async def test_consume_boundary_rechecks_all_evidence(active_case, target):
    args, _, _ = active_case
    revision = [args["selection_result"].repository_revision]
    args["get_current_revision"] = lambda: revision[0]
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "active"
    if target == "source":
        # The removed file is still required for the selection proof.
        change_source(args)
    elif target == "baseline":
        args["final_baseline_bundle"].engineering_rules.append("changed")
    elif target == "selection":
        args["selection_result"].selection.selected_files.pop()
    elif target == "surface":
        args["execution_surface"].references.append("src/f7.ts")
    elif target == "candidate":
        result.bundle.code_context[0].content = "changed"
    else:
        revision[0] = "b" * 40
    with pytest.raises(StaleContextEvidenceError):
        result.assert_fresh()


@pytest.mark.asyncio
async def test_missing_existing_required_context_blocks_before_observer(active_case):
    args, runner, _ = active_case
    change_source(args, "src/extra.ts")
    args["execution_surface"].dependencies.append("src/extra.ts")
    args["final_baseline_bundle"].change_surface = args["execution_surface"].model_dump(
        mode="json"
    )
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "blocked"
    assert result.report.reason == "mandatory_context_missing"
    assert result.bundle is None
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_authorized_create_file_may_be_absent(active_case):
    args, _, _ = active_case
    surface = args["execution_surface"]
    surface.tests.append("src/new.test.ts")
    surface.create_files.append("src/new.test.ts")
    args["final_baseline_bundle"].change_surface = surface.model_dump(mode="json")
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "active"
    assert result.report.mandatory_context_complete
    assert not (args["repository_root"] / "src/new.test.ts").exists()
    result.assert_fresh()


@pytest.mark.asyncio
async def test_absent_unapproved_required_file_blocks(active_case):
    args, runner, _ = active_case
    args["execution_surface"].references.append("src/missing.ts")
    args["final_baseline_bundle"].change_surface = args["execution_surface"].model_dump(
        mode="json"
    )
    result = await run_full_execution_guarded_active(**args)
    assert result.report.status == "blocked"
    assert result.bundle is None
    assert runner.calls == 0


def test_public_trial_remains_report_only(trial_case):
    report = run_guarded_active_trial(**trial_case)
    assert report.status == "candidate_valid"
    assert not hasattr(report, "candidate")
    assert not hasattr(report, "bundle")
    assert report.applied is False


def test_trial_early_rebuild_failure_still_checks_sources(trial_case):
    builder = trial_case["build_final_context"]

    def rebuild(selection):
        bundle = builder(selection)
        change_source(trial_case)
        bundle.engineering_rules.append("mismatch")
        return bundle

    trial_case["build_final_context"] = rebuild
    report = run_guarded_active_trial(**trial_case)
    assert report.status == "blocked"
    assert report.applied is False
