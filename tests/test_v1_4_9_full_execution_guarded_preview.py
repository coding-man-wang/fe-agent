from dataclasses import replace

import pytest

from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.full_execution_guarded_preview import (
    run_full_execution_guarded_preview,
)
from test_v1_4_9_full_execution_guarded_trial import trial_case  # noqa: F401


class ProposalRunner:
    def __init__(self, callback=None):
        self.calls = 0
        self.callback = callback

    async def observe(self, **kwargs):
        self.calls += 1
        if self.callback is not None:
            self.callback()
        return {
            "status": "complete",
            "baseline_file_count": 7,
            "proposed_file_count": 6,
            "graph_proposed_drops": ["src/f6.ts"],
            "rerank_proposed_drops": [],
            "effective_proposed_drops": ["src/f6.ts"],
        }


@pytest.fixture
def bridge_case(trial_case):
    args = dict(trial_case)
    args.pop("preview")
    events = []
    runner = ProposalRunner()
    args["observer"] = FullExecutionContextShadowObserver(runner=runner)
    args["audit_sink"] = events.append
    return args, runner, events


def change_source(args, path="src/f6.ts"):
    (args["repository_root"] / path).write_text("export const changed = 99;\n")


@pytest.mark.asyncio
async def test_fresh_pipeline_returns_only_summary_and_keeps_baseline(bridge_case):
    args, runner, events = bridge_case
    baseline = args["final_baseline_bundle"]
    original = baseline.model_dump(mode="json")
    builder = args["build_final_context"]
    builds = []

    def build(selection):
        builds.append(len(selection.selected_files))
        return builder(selection)

    args["build_final_context"] = build
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "candidate_valid"
    assert result.removed_files == ("src/f6.ts",)
    assert result.candidate_chars < result.baseline_chars
    assert runner.calls == 1
    assert builds == [7, 6]
    assert baseline.model_dump(mode="json") == original
    assert result.snapshot_matched and result.audit_recorded
    assert len(result.baseline_bundle_sha256) == 64
    assert len(result.selection_sha256) == len(result.source_sha256) == 64
    assert (
        result.applied
        is result.active_execution_enabled
        is result.production_ready
        is False
    )
    assert not hasattr(result, "bundle")
    assert events[-1]["applied"] is False
    assert "export const" not in str(events)


@pytest.mark.asyncio
async def test_denied_preflight_skips_observer_but_still_audits(bridge_case):
    args, runner, events = bridge_case
    args["preflight"] = replace(args["preflight"], status="denied")
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "fallback"
    assert result.reason == "preflight_not_eligible"
    assert result.snapshot_matched
    assert runner.calls == 0
    assert events[-1]["status"] == "fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["applied", "production_ready", "active_execution_enabled"]
)
async def test_preflight_cannot_enable_active(bridge_case, field):
    args, runner, _ = bridge_case
    args["preflight"] = replace(args["preflight"], **{field: True})
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "fallback"
    assert runner.calls == 0
    assert result.applied is result.active_execution_enabled is False


@pytest.mark.asyncio
async def test_observer_error_uses_unchanged_baseline(bridge_case):
    args, runner, _ = bridge_case

    def fail():
        raise RuntimeError("diagnostic unavailable")

    runner.callback = fail
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "fallback"
    assert result.reason == "preview:invalid_shadow_provenance"
    assert result.snapshot_matched


@pytest.mark.asyncio
async def test_source_change_on_observer_error_blocks_baseline(bridge_case):
    args, runner, _ = bridge_case

    def fail():
        change_source(args)
        raise RuntimeError("diagnostic failed after mutation")

    runner.callback = fail
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"
    assert not result.snapshot_matched


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_audit_failure_and_mutation_checked_after_append(bridge_case, failure):
    args, _, _ = bridge_case

    def audit(event):
        change_source(args)
        if failure:
            raise OSError("audit storage unavailable")

    args["audit_sink"] = audit
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"
    assert result.reason == "final_snapshot_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_audit_unavailable_discards_success(bridge_case, missing):
    args, _, _ = bridge_case

    def audit(event):
        raise OSError("audit storage unavailable")

    args["audit_sink"] = None if missing else audit
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "fallback"
    assert result.reason == "preview_audit_unavailable"
    assert result.audit_recorded is False
    assert result.snapshot_matched
    assert result.applied is False


@pytest.mark.asyncio
async def test_audit_payload_mutation_cannot_enable_active(bridge_case):
    args, _, _ = bridge_case

    def audit(event):
        event.update(applied=True, active_execution_enabled=True, production_ready=True)

    args["audit_sink"] = audit
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "candidate_valid"
    assert (
        result.applied
        is result.production_ready
        is result.active_execution_enabled
        is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutate", [False, True])
async def test_builder_exception_requires_safe_baseline(bridge_case, mutate):
    args, _, _ = bridge_case

    def build(selection):
        if mutate:
            change_source(args)
        raise RuntimeError("build failed")

    args["build_final_context"] = build
    result = await run_full_execution_guarded_preview(**args)
    # C1 classifies a baseline reconstruction failure as blocked, never success.
    assert result.status == "blocked"
    assert result.snapshot_matched is (not mutate)


@pytest.mark.asyncio
async def test_obligation_evidence_is_protected(bridge_case):
    args, _, _ = bridge_case
    args["obligation_evidence_files"] = ["src/f6.ts"]
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "fallback"
    assert result.preview_reason == "final_context_protected_drop"
    assert result.trial_reason == "not_run"


@pytest.mark.asyncio
async def test_missing_obligation_evidence_blocks_even_before_observer(bridge_case):
    args, runner, _ = bridge_case
    args["obligation_evidence_files"] = ["src/not-in-bundle.ts"]
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"
    assert result.reason == "mandatory_context_missing"
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_source_outside_selection_and_context_is_protected(bridge_case):
    args, runner, _ = bridge_case
    surface = args["execution_surface"]
    surface.dependencies.append("src/extra.ts")
    change_source(args, "src/extra.ts")
    args["final_baseline_bundle"].change_surface = surface.model_dump(mode="json")
    runner.callback = lambda: (args["repository_root"] / "src/extra.ts").unlink()
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"


@pytest.mark.asyncio
async def test_creation_of_missing_future_target_is_detected(bridge_case):
    args, runner, _ = bridge_case
    surface = args["execution_surface"]
    surface.implementation_targets.append("src/new.ts")
    args["final_baseline_bundle"].change_surface = surface.model_dump(mode="json")
    runner.callback = lambda: change_source(args, "src/new.ts")
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["bundle", "selection", "surface", "obligations", "graph"]
)
async def test_mutation_of_shared_inputs_blocks_even_on_fallback(bridge_case, target):
    args, runner, _ = bridge_case

    def mutate():
        if target == "bundle":
            args["final_baseline_bundle"].engineering_rules.append("changed")
        elif target == "selection":
            args["selection_result"].selection.selected_files.pop()
        elif target == "surface":
            args["execution_surface"].references.append("src/new.ts")
        elif target == "obligations":
            args["obligation_evidence_files"].append("src/f1.ts")
        else:
            args["selection_result"].graph.root_path = "changed"

    runner.callback = mutate
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"
    assert not result.snapshot_matched


@pytest.mark.asyncio
async def test_revision_change_after_audit_blocks(bridge_case):
    args, _, _ = bridge_case
    revision = [args["selection_result"].repository_revision]
    args["get_current_revision"] = lambda: revision[0]
    args["audit_sink"] = lambda event: revision.__setitem__(0, "b" * 40)
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"


@pytest.mark.asyncio
async def test_early_fallback_still_checks_audit_side_effects(bridge_case):
    args, runner, _ = bridge_case
    args["preflight"] = replace(args["preflight"], status="denied")
    args["audit_sink"] = lambda event: change_source(args)
    result = await run_full_execution_guarded_preview(**args)
    assert result.status == "blocked"
    assert runner.calls == 0
