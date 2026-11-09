from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from agent_core.benchmark.context_policy_runtime import _additional_source_hashes
from agent_core.benchmark.full_execution_active_preflight import ActivePreflightDecision
from agent_core.benchmark.full_execution_active_preview import (
    preview_full_execution_active_candidate,
)
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
    capture_final_context_source_digest,
)
from agent_core.benchmark.full_execution_guarded_trial import run_guarded_active_trial
from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
    json_digest,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.codebase.task_context_selection import TaskContextSelectionResult
from agent_core.context.model import TaskContextBundle


@dataclass(frozen=True)
class GuardedPreviewReport:
    status: Literal["candidate_valid", "fallback", "blocked"]
    reason: str
    task_id: str
    repository_revision: str
    case_id: str
    canary_sha256: str | None
    baseline_files: tuple[str, ...] = ()
    candidate_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()
    baseline_chars: int = 0
    candidate_chars: int = 0
    baseline_bundle_sha256: str = ""
    selection_sha256: str = ""
    source_sha256: str = ""
    snapshot_matched: bool = False
    preview_reason: str = "not_run"
    trial_reason: str = "not_run"
    audit_recorded: bool = False
    policy_version: str = "v1.4.9-P4-B-D2"
    event_type: str = "full_execution_guarded_preview"
    mode: str = "preview"
    applied: bool = False
    production_ready: bool = False
    active_execution_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_full_execution_guarded_preview(
    *,
    repository_root: str | Path,
    raw_requirement: str,
    selection_result: TaskContextSelectionResult,
    final_baseline_bundle: TaskContextBundle,
    execution_surface: ResolvedChangeSurface,
    obligation_evidence_files: list[str],
    preflight: ActivePreflightDecision,
    get_current_revision: Callable[[], str],
    build_final_context: Callable[[HybridRelevantFileSelection], TaskContextBundle],
    observer: FullExecutionContextShadowObserver,
    audit_sink: Callable[[dict[str, Any]], None] | None = None,
) -> GuardedPreviewReport:
    """Run D1 -> D2-B -> C1 at the real Full Execution consumption boundary.

    The caller supplies the already retrieved selection and the SAME final build
    pipeline used by Full Execution, including obligation/strategy enrichment.
    No candidate bundle escapes this function. Only the original baseline may
    reach Coding Agent. A ``blocked`` result MUST stop execution; a fallback may
    use the unchanged baseline because it passes the same final snapshot gate.
    """
    baseline = final_baseline_bundle
    revision = selection_result.repository_revision
    paths = tuple(item.relative_path for item in baseline.code_context)
    chars = sum(len(item.content) for item in baseline.code_context)
    result = GuardedPreviewReport(
        status="fallback",
        reason="not_run",
        task_id=baseline.task.task_id,
        repository_revision=revision,
        case_id=preflight.case_id,
        canary_sha256=preflight.canary_sha256,
        baseline_files=paths,
        candidate_files=paths,
        baseline_chars=chars,
        candidate_chars=chars,
        baseline_bundle_sha256=json_digest(baseline.model_dump(mode="json")),
    )
    # Freeze both the consumed inputs and all file paths before any observer,
    # builder or audit callback can run. Missing future targets are fingerprinted
    # too, so creation/deletion outside the selection cannot slip through.
    frozen_bundle = deepcopy(baseline)
    frozen_selection = deepcopy(selection_result)
    frozen_surface = deepcopy(execution_surface)
    frozen_obligations = list(obligation_evidence_files)
    protected = (
        set(execution_surface.implementation_targets)
        | set(execution_surface.tests)
        | set(execution_surface.dependencies)
        | set(execution_surface.references)
        | set(frozen_obligations)
    )
    strategy = baseline.verification_strategy
    strategy_valid = strategy is None or (
        isinstance(strategy, dict)
        and isinstance(strategy.get("writable_test_files"), list)
        and all(isinstance(path, str) for path in strategy["writable_test_files"])
    )
    if strategy_valid and strategy is not None:
        protected.update(strategy["writable_test_files"])

    def snapshot() -> tuple[str, str]:
        if get_current_revision() != revision:
            raise ValueError("revision_changed")
        if (
            baseline != frozen_bundle
            or selection_result != frozen_selection
            or execution_surface != frozen_surface
            or obligation_evidence_files != frozen_obligations
        ):
            raise ValueError("execution_inputs_changed")
        selection = capture_selection_evidence(
            repository_root=repository_root, selection=frozen_selection.selection
        )
        source = json_digest(
            {
                "selection": selection.source_sha256,
                "context": capture_final_context_source_digest(
                    repository_root=repository_root, bundle=frozen_bundle
                ),
                "protected": _additional_source_hashes(
                    Path(repository_root).resolve(strict=True), protected
                ),
            }
        )
        if get_current_revision() != revision:
            raise ValueError("revision_changed")
        return selection.selection_sha256, source

    try:
        initial_snapshot = snapshot()
    except Exception:
        # Baseline validity cannot be established, so even an early abstention
        # must not allow the model to consume this bundle.
        result = replace(
            result, status="blocked", reason="initial_snapshot_unavailable"
        )
        initial_snapshot = None

    if initial_snapshot is not None:
        result = replace(
            result,
            selection_sha256=initial_snapshot[0],
            source_sha256=initial_snapshot[1],
        )
        eligible = (
            preflight.status == "passed"
            and preflight.case_id == "CASE-006"
            and preflight.task_id == baseline.task.task_id == "TASK-002"
            and preflight.approved_revision == preflight.current_revision == revision
            and bool(preflight.canary_sha256)
            and preflight.applied is False
            and preflight.production_ready is False
            and preflight.active_execution_enabled is False
        )
        try:
            if not eligible:
                result = replace(result, reason="preflight_not_eligible")
            elif not strategy_valid:
                result = replace(result, reason="strategy_schema_invalid")
            elif set(frozen_obligations) - set(paths):
                result = replace(
                    result, status="blocked", reason="mandatory_context_missing"
                )
            else:
                shadow = await observer.observe(
                    repository_root=repository_root,
                    repository_revision=revision,
                    raw_requirement=raw_requirement,
                    selection_result=deepcopy(frozen_selection),
                    bundle=deepcopy(frozen_bundle),
                )
                preview = preview_full_execution_active_candidate(
                    repository_root=repository_root,
                    raw_requirement=raw_requirement,
                    selection_result=deepcopy(frozen_selection),
                    final_bundle=deepcopy(frozen_bundle),
                    execution_surface=deepcopy(frozen_surface),
                    obligation_evidence_files=list(frozen_obligations),
                    shadow_report=shadow,
                    get_current_revision=get_current_revision,
                    fresh_in_process_report=True,
                )
                result = replace(result, preview_reason=preview.reason)
                if preview.status != "preview_valid":
                    result = replace(result, reason="preview:" + preview.reason)
                else:
                    trial = run_guarded_active_trial(
                        repository_root=repository_root,
                        raw_requirement=raw_requirement,
                        selection_result=deepcopy(frozen_selection),
                        final_baseline_bundle=deepcopy(frozen_bundle),
                        execution_surface=deepcopy(frozen_surface),
                        obligation_evidence_files=list(frozen_obligations),
                        preflight=preflight,
                        preview=preview,
                        get_current_revision=get_current_revision,
                        build_final_context=build_final_context,
                    )
                    result = replace(
                        result,
                        status=trial.status,
                        reason="trial:" + trial.reason,
                        trial_reason=trial.reason,
                        candidate_files=trial.candidate_files,
                        removed_files=trial.removed_files,
                        candidate_chars=trial.candidate_chars,
                    )
        except Exception as exc:
            result = replace(result, reason="preview_error:" + type(exc).__name__)

    def final_guard(current: GuardedPreviewReport) -> GuardedPreviewReport:
        if initial_snapshot is None:
            return replace(current, status="blocked", snapshot_matched=False)
        try:
            matched = snapshot() == initial_snapshot
        except Exception:
            matched = False
        if not matched:
            return replace(
                current,
                status="blocked",
                reason="final_snapshot_changed",
                snapshot_matched=False,
            )
        return replace(current, snapshot_matched=True)

    result = final_guard(result)
    if audit_sink is None:
        if result.status != "blocked":
            result = replace(
                result, status="fallback", reason="preview_audit_unavailable"
            )
    else:
        try:
            # A successful synchronous append is required. The sink receives a
            # disposable value and cannot turn this preview into Active.
            audit_sink(replace(result, audit_recorded=True).to_dict())
            result = replace(result, audit_recorded=True)
        except Exception:
            if result.status != "blocked":
                result = replace(
                    result, status="fallback", reason="preview_audit_unavailable"
                )

    # Check again AFTER audit on every path, including preflight rejection,
    # malformed proposals, builder exceptions and audit failures.
    final = final_guard(result)
    if (
        final.status == "blocked"
        and result.status != "blocked"
        and audit_sink is not None
    ):
        try:
            audit_sink(replace(final, audit_recorded=True).to_dict())
            final = replace(final, audit_recorded=True)
        except Exception:
            final = replace(final, audit_recorded=False)
    return final
