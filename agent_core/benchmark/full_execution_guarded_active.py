from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_core.benchmark.full_execution_active_preflight import ActivePreflightDecision
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.full_execution_guarded_preview import (
    GuardedPreviewReport,
    _run_full_execution_guarded_context,
)
from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.codebase.task_context_selection import TaskContextSelectionResult
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.context.model import TaskContextBundle


@dataclass(frozen=True)
class GuardedActiveResult:
    bundle: TaskContextBundle | None
    report: GuardedPreviewReport
    _assert_fresh: Callable[[], None] = field(repr=False, compare=False)

    def assert_fresh(self) -> None:
        """Recheck ALL original evidence and released bundle immediately before LLM.

        Raises StaleContextEvidenceError instead of returning a stale baseline.
        The caller must stop execution if this check fails.
        """
        self._assert_fresh()


async def run_full_execution_guarded_active(
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
    experimental_active_enabled: bool = False,
) -> GuardedActiveResult:
    """Experimental Full Execution handoff, NEVER a production enablement.

    Requires explicit opt-in, current CASE-006/TASK-002 preflight, fresh observer,
    shared C1 proof and a synchronous durable audit sink. The revision callback
    must verify a clean committed workspace, including extra builder inputs.
    Active returns the verified candidate; fallback returns the original baseline;
    blocked returns None. Call assert_fresh() after prepare_messages and before LLM.
    """
    result = await _run_full_execution_guarded_context(
        repository_root=repository_root,
        raw_requirement=raw_requirement,
        selection_result=selection_result,
        final_baseline_bundle=final_baseline_bundle,
        execution_surface=execution_surface,
        obligation_evidence_files=obligation_evidence_files,
        preflight=preflight,
        get_current_revision=get_current_revision,
        build_final_context=build_final_context,
        observer=observer,
        audit_sink=audit_sink,
        mode="active",
        experimental_active_enabled=experimental_active_enabled,
    )
    return GuardedActiveResult(result.bundle, result.report, result.assert_fresh)
