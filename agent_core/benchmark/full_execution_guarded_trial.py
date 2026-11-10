from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
    MissingMandatoryContextError,
    StaleContextEvidenceError,
    _additional_source_hashes,
)
from agent_core.benchmark.full_execution_active_preflight import (
    ActivePreflightDecision,
)
from agent_core.benchmark.full_execution_active_preview import (
    ActiveCandidatePreview,
)
from agent_core.benchmark.full_execution_context_shadow import (
    capture_final_context_source_digest,
)
from agent_core.benchmark.selection_evidence_binding import (
    json_digest,
    capture_selection_evidence,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle


@dataclass(frozen=True)
class GuardedTrialReport:
    status: Literal["candidate_valid", "fallback", "blocked"]
    reason: str
    task_id: str
    repository_revision: str

    baseline_files: tuple[str, ...] = ()
    candidate_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()

    baseline_chars: int = 0
    candidate_chars: int = 0

    # 试运行从不改变正式 Agent 的 Context。
    applied: bool = False
    production_ready: bool = False


@dataclass(frozen=True)
class _GuardedCandidateResult:
    report: GuardedTrialReport
    candidate: TaskContextBundle | None = None


def _build_guarded_candidate(
    *,
    repository_root: str | Path,
    raw_requirement: str,
    selection_result: TaskContextSelectionResult,
    final_baseline_bundle: TaskContextBundle,
    execution_surface: ResolvedChangeSurface,
    obligation_evidence_files: list[str],
    preflight: ActivePreflightDecision,
    preview: ActiveCandidatePreview,
    get_current_revision: Callable[[], str],
    build_final_context: Callable[
        [HybridRelevantFileSelection],
        TaskContextBundle,
    ],
) -> _GuardedCandidateResult:
    """
    D2-C1：在内存中试运行 P3 Active Consumer。

    build_final_context 必须复用正式链路：
      Resolved Context
      -> Obligation Evidence Enrichment
      -> Verification Strategy

    Shared internal validator for report-only C1 and experimental handoff.
    Only a fully validated candidate is exposed to the guarded outer pipeline.
    Does not invoke Coding Agent or Patch.
    """
    baseline = final_baseline_bundle
    task_id = baseline.task.task_id
    revision = selection_result.repository_revision

    frozen_bundle = deepcopy(baseline)
    frozen_selection = deepcopy(selection_result)
    frozen_surface = deepcopy(execution_surface)
    frozen_obligations = list(obligation_evidence_files)
    validated_candidate_digest = None
    protected = (
        set(execution_surface.implementation_targets)
        | set(execution_surface.tests)
        | set(execution_surface.dependencies)
        | set(execution_surface.references)
        | set(obligation_evidence_files)
    )
    strategy = baseline.verification_strategy
    strategy_valid = strategy is None or (
        isinstance(strategy, dict)
        and isinstance(strategy.get("writable_test_files"), list)
        and all(isinstance(path, str) for path in strategy["writable_test_files"])
    )
    if strategy_valid and strategy is not None:
        protected.update(strategy["writable_test_files"])

    def snapshot():
        if (
            get_current_revision() != revision
            or baseline != frozen_bundle
            or selection_result != frozen_selection
            or execution_surface != frozen_surface
            or obligation_evidence_files != frozen_obligations
        ):
            raise StaleContextEvidenceError("trial_inputs_changed")
        evidence = capture_selection_evidence(
            repository_root=repository_root,
            selection=frozen_selection.selection,
        )
        context_digest = capture_final_context_source_digest(
            repository_root=repository_root,
            bundle=frozen_bundle,
        )
        sources = _additional_source_hashes(
            Path(repository_root).resolve(strict=True),
            protected,
        )
        if (
            get_current_revision() != revision
            or baseline != frozen_bundle
            or selection_result != frozen_selection
            or execution_surface != frozen_surface
            or obligation_evidence_files != frozen_obligations
        ):
            raise StaleContextEvidenceError("trial_inputs_changed")
        return evidence, context_digest, sources

    try:
        initial_snapshot = snapshot()
    except Exception:
        initial_snapshot = None

    before_paths = tuple(item.relative_path for item in baseline.code_context)
    before_chars = sum(len(item.content) for item in baseline.code_context)

    def report(status, reason, *, bundle=None):
        current = bundle or baseline
        current_paths = tuple(item.relative_path for item in current.code_context)
        removed = tuple(path for path in before_paths if path not in current_paths)
        result = GuardedTrialReport(
            status=status,
            reason=reason,
            task_id=task_id,
            repository_revision=revision,
            baseline_files=before_paths,
            candidate_files=current_paths,
            removed_files=removed,
            baseline_chars=before_chars,
            candidate_chars=sum(len(item.content) for item in current.code_context),
        )
        try:
            unchanged = initial_snapshot is not None and snapshot() == initial_snapshot
        except Exception:
            unchanged = False
        if not unchanged:
            result = replace(result, status="blocked", reason="final_snapshot_changed")
        if status == "candidate_valid" and (
            bundle is None
            or validated_candidate_digest is None
            or json_digest(bundle.model_dump(mode="json")) != validated_candidate_digest
        ):
            result = replace(
                result, status="blocked", reason="validated_candidate_changed"
            )
        return _GuardedCandidateResult(
            report=result,
            candidate=bundle if result.status == "candidate_valid" else None,
        )

    if initial_snapshot is None:
        return report("blocked", "baseline_source_unavailable")

    # 1. D2-A 准入状态必须与本次 Task/Revision 绑定。
    if not (
        preflight.status == "passed"
        and preflight.case_id == "CASE-006"
        and preflight.task_id == task_id == "TASK-002"
        and preflight.approved_revision == revision
        and preflight.current_revision == revision
        and preflight.canary_sha256
        and preflight.applied is False
        and preflight.production_ready is False
        and preflight.active_execution_enabled is False
    ):
        return report("fallback", "preflight_not_eligible")

    # 2. D2-B 候选必须有效。
    if (
        preview.status != "preview_valid"
        or not isinstance(preview.p1_report, dict)
        or preview.applied
    ):
        return report("fallback", "candidate_preview_invalid")

    if (
        preview.p1_report.get("task_id") != task_id
        or preview.p1_report.get("repository_revision") != revision
    ):
        return report("fallback", "preview_identity_mismatch")

    if (
        execution_surface.task_id != task_id
        or baseline.change_surface != execution_surface.model_dump(mode="json")
    ):
        return report("fallback", "surface_mismatch")

    baseline_digest = json_digest(baseline.model_dump(mode="json"))

    try:
        baseline_source_digest = capture_final_context_source_digest(
            repository_root=repository_root,
            bundle=baseline,
        )
    except (OSError, ValueError, UnicodeError):
        return report("blocked", "baseline_source_unavailable")

    # Existing mandatory evidence must already be present. Only authorized
    # create_files may be absent; intersection-only protection hides omissions.
    if not strategy_valid:
        return report("fallback", "strategy_schema_invalid")
    mandatory = protected
    current_paths = set(before_paths)
    mandatory_sources = initial_snapshot[2]
    allowed_missing = set(execution_surface.create_files)
    required_existing = {
        path
        for path in mandatory
        if mandatory_sources[path.replace("\\", "/")] != "__missing__"
    }
    absent_required = {
        path
        for path in mandatory
        if mandatory_sources[path.replace("\\", "/")] == "__missing__"
        and path not in allowed_missing
    }
    if required_existing - current_paths or absent_required:
        return report("blocked", "mandatory_context_missing")
    mandatory_in_context = mandatory & current_paths

    # 即使 D2-B 存在遗漏，D2-C 仍独立拒绝。
    if mandatory_in_context & set(preview.proposed_selection_drops):
        return report("fallback", "mandatory_evidence_drop")

    # 4. 通过真实 Consumer 进行候选构建。
    # Audit Sink 仅存在于内存中。
    # 内部 applied=True 表示 Consumer 在模拟中
    # 选择 Candidate，绝不代表正式 Agent 已应用。
    simulated_audit: list[dict] = []
    reconstructed_baselines: list[TaskContextBundle] = []

    original_selection_digest = json_digest(
        selection_result.selection.model_dump(mode="json")
    )

    def build_checked(selection):
        selection_copy = selection.model_copy(deep=True)
        built = build_final_context(selection_copy)

        # 明确识别一次使用原 Selection 的重建结果。
        if json_digest(selection.model_dump(mode="json")) == original_selection_digest:
            reconstructed_baselines.append(built)

        return built

    consumer = GuardedContextPolicyConsumer(
        config=ContextRolloutConfig(
            mode="active",
            active_enabled=True,
            canary_passed=True,
            approved_revision=revision,
            task_allowlist=frozenset({task_id}),
            max_drops=2,
            min_files=2,
        ),
        audit_sink=simulated_audit.append,
    )

    try:
        consumed = consumer.consume(
            repository_root=repository_root,
            task=baseline.task,
            raw_requirement=raw_requirement,
            selection_result=selection_result,
            report=preview.p1_report,
            get_current_revision=get_current_revision,
            build_context=build_checked,
            required_context_paths=mandatory_in_context,
            additional_source_paths=(current_paths | mandatory),
            fresh_in_process_report=True,
        )

    except StaleContextEvidenceError:
        # 源码或版本已经变化，不能继续使用旧 Context。
        return report("blocked", "stale_context_evidence")

    except MissingMandatoryContextError:
        return report("blocked", "mandatory_context_missing")

    except (OSError, ValueError, RuntimeError, TypeError):
        return report("blocked", "consumer_execution_error")

    # 5. Consumer 回退时不能使用旧的候选结果。
    if consumed.decision.status != "active":
        return report(
            "fallback",
            "consumer:" + consumed.decision.reason,
        )

    if not simulated_audit or not (simulated_audit[-1].get("applied") is True):
        return report("fallback", "consumer_audit_missing")

    # 6. 用同样的构建流程重建 Baseline，
    # 必须与正式 Agent 的原始 Bundle 完全一致。
    if len(reconstructed_baselines) != 1:
        return report("fallback", "baseline_rebuild_missing")

    reconstructed = reconstructed_baselines[0]

    if json_digest(reconstructed.model_dump(mode="json")) != baseline_digest:
        return report("fallback", "baseline_rebuild_mismatch")

    candidate = consumed.bundle

    # 7. 除 Code Context 和 Budget 外，
    # 所有任务、策略、权限、Artifact 都必须相同。
    baseline_data = baseline.model_dump(mode="json")
    candidate_data = candidate.model_dump(mode="json")

    for key in ("code_context", "budget"):
        baseline_data.pop(key)
        candidate_data.pop(key)

    if baseline_data != candidate_data:
        return report("fallback", "context_metadata_changed")

    candidate_paths = tuple(item.relative_path for item in candidate.code_context)

    if len(set(before_paths)) != len(before_paths) or len(set(candidate_paths)) != len(
        candidate_paths
    ):
        return report("fallback", "duplicate_context_path")

    # 本阶段只允许删除文件，不允许插入文件或调整文件顺序。
    candidate_set = set(candidate_paths)

    expected_order = tuple(path for path in before_paths if path in candidate_set)

    if candidate_paths != expected_order:
        return report("fallback", "unexpected_context_reorder")

    removed = set(before_paths) - candidate_set

    if removed != set(preview.drops_in_final_context):
        return report("fallback", "candidate_drop_mismatch")

    if not removed:
        return report("fallback", "candidate_no_effect")

    if mandatory_in_context & removed:
        return report("fallback", "mandatory_context_removed")

    before_by_path = {item.relative_path: item for item in baseline.code_context}

    # 8. 保留文件的内容和 Symbol 元数据必须完全相同。
    # 目前不接受候选重排或额外改变窗口内容。
    for item in candidate.code_context:
        original_item = before_by_path[item.relative_path]

        if original_item.model_dump(mode="json") != item.model_dump(mode="json"):
            return report(
                "fallback",
                "retained_context_changed",
            )

    for path in removed:
        removed_item = before_by_path[path]

        if (
            removed_item.symbol_candidates
            or removed_item.covered_symbol_ids
            or removed_item.localized_symbol_ids
        ):
            return report(
                "fallback",
                "symbol_evidence_removed",
            )

    candidate_chars = sum(len(item.content) for item in candidate.code_context)

    if (
        candidate_chars >= before_chars
        or candidate.budget.used_chars > baseline.budget.used_chars
        or candidate.budget.used_files > baseline.budget.used_files
    ):
        return report(
            "fallback",
            "context_budget_not_improved",
        )

    validated_candidate_digest = json_digest(candidate.model_dump(mode="json"))

    # 9. 最后一次检查原始 Bundle 和源码没有变化。
    try:
        final_revision = get_current_revision()
        final_source_digest = capture_final_context_source_digest(
            repository_root=repository_root,
            bundle=baseline,
        )
    except (OSError, ValueError, UnicodeError, RuntimeError):
        return report("blocked", "final_snapshot_unavailable")

    if (
        final_revision != revision
        or final_source_digest != baseline_source_digest
        or json_digest(baseline.model_dump(mode="json")) != baseline_digest
    ):
        return report("blocked", "final_snapshot_changed")

    return report(
        "candidate_valid",
        "consumer_candidate_verified",
        bundle=candidate,
    )


def run_guarded_active_trial(
    *,
    repository_root: str | Path,
    raw_requirement: str,
    selection_result: TaskContextSelectionResult,
    final_baseline_bundle: TaskContextBundle,
    execution_surface: ResolvedChangeSurface,
    obligation_evidence_files: list[str],
    preflight: ActivePreflightDecision,
    preview: ActiveCandidatePreview,
    get_current_revision: Callable[[], str],
    build_final_context: Callable[[HybridRelevantFileSelection], TaskContextBundle],
) -> GuardedTrialReport:
    """Public C1 remains report-only; no candidate or Active authorization escapes."""
    return _build_guarded_candidate(
        repository_root=repository_root,
        raw_requirement=raw_requirement,
        selection_result=selection_result,
        final_baseline_bundle=final_baseline_bundle,
        execution_surface=execution_surface,
        obligation_evidence_files=obligation_evidence_files,
        preflight=preflight,
        preview=preview,
        get_current_revision=get_current_revision,
        build_final_context=build_final_context,
    ).report
