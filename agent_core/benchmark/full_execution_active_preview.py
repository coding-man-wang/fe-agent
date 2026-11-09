from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from agent_core.benchmark.full_execution_context_shadow import (
    capture_final_context_source_digest,
)
from agent_core.benchmark.selection_evidence_binding import (
    json_digest,
)
from agent_core.benchmark.selection_snapshot_gate import (
    ShadowSnapshotGate,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle


@dataclass(frozen=True)
class ActiveCandidatePreview:
    status: Literal["preview_valid", "abstained"]
    reason: str
    proposed_selection_drops: tuple[str, ...] = ()
    drops_in_final_context: tuple[str, ...] = ()
    protected_context_paths: tuple[str, ...] = ()
    p1_report: dict[str, Any] | None = None

    # Preview 永远不应用优化。
    applied: bool = False
    production_ready: bool = False


def preview_full_execution_active_candidate(
    *,
    repository_root: str | Path,
    raw_requirement: str,
    selection_result: TaskContextSelectionResult,
    final_bundle: TaskContextBundle,
    execution_surface: ResolvedChangeSurface,
    obligation_evidence_files: list[str],
    shadow_report: Mapping[str, Any],
    get_current_revision: Callable[[], str],
    fresh_in_process_report: bool,
    max_drops: int = 2,
    min_files: int = 2,
) -> ActiveCandidatePreview:
    """
    v1.4.9-P4-B-D2-B

    验证 Full Execution 的候选删除建议。

    不调用 Retrieval / Rerank。
    不重建或修改 Context。
    不授予写权限。
    不调用 Coding Agent。
    """

    def abstain(reason: str) -> ActiveCandidatePreview:
        return ActiveCandidatePreview(
            status="abstained",
            reason=reason,
        )

    if not fresh_in_process_report:
        return abstain("not_fresh_in_process")

    # 必须来自本轮 D1 Observer，不接受外部伪造
    # 的 P1 版本标记。
    if (
        shadow_report.get("policy_version") != "v1.4.9-P4-B-D1"
        or shadow_report.get("mode") != "shadow"
        or shadow_report.get("status") != "complete"
        or shadow_report.get("applied") is not False
    ):
        return abstain("invalid_shadow_provenance")

    if any(
        shadow_report.get(key) is not True
        for key in (
            "selection_unchanged",
            "context_unchanged",
            "bundle_source_matched",
        )
    ):
        return abstain("shadow_evidence_invalid")

    task_id = final_bundle.task.task_id
    revision = selection_result.repository_revision

    if execution_surface.task_id != task_id or shadow_report.get("task_id") != task_id:
        return abstain("task_mismatch")

    if shadow_report.get("repository_revision") != revision:
        return abstain("revision_mismatch")

    # 确认模型即将消费的 Surface
    # 就是当前 Execution Surface。
    if final_bundle.change_surface != (execution_surface.model_dump(mode="json")):
        return abstain("change_surface_mismatch")

    strategy = final_bundle.verification_strategy

    if strategy is not None and not isinstance(strategy, dict):
        return abstain("invalid_verification_strategy")

    # 检查最终 Bundle 的结构和内容。
    current_bundle_digest = json_digest(final_bundle.model_dump(mode="json"))

    if (
        shadow_report.get("bundle_sha256_before") != current_bundle_digest
        or shadow_report.get("bundle_sha256_after") != current_bundle_digest
    ):
        return abstain("final_bundle_changed")

    # 检查最终 Context 所引用的真实源码，
    # 不局限于 Retrieval Selection。
    try:
        current_source_digest = capture_final_context_source_digest(
            repository_root=repository_root,
            bundle=final_bundle,
        )
    except (OSError, ValueError, UnicodeError):
        return abstain("final_source_unavailable")

    if (
        shadow_report.get("bundle_source_sha256_before") != current_source_digest
        or shadow_report.get("bundle_source_sha256_after") != current_source_digest
    ):
        return abstain("final_source_changed")

    # D1 -> P1 的显式兼容转换。
    # 不直接修改原报告的 policy_version。
    binding = shadow_report.get("evidence_binding")

    if not isinstance(binding, dict):
        return abstain("missing_selection_binding")

    p1_report = {
        "policy_version": "v1.4.9-P1",
        "mode": "shadow",
        "status": "complete",
        "applied": False,
        "task_id": task_id,
        "repository_revision": revision,
        "requirement_sha256": shadow_report.get("requirement_sha256"),
        "selection_unchanged": True,
        "baseline_file_count": shadow_report.get("baseline_file_count"),
        "proposed_file_count": shadow_report.get("proposed_file_count"),
        "graph_proposed_drops": shadow_report.get("graph_proposed_drops"),
        "rerank_proposed_drops": shadow_report.get("rerank_proposed_drops"),
        "effective_proposed_drops": shadow_report.get("effective_proposed_drops"),
        "evidence_binding": dict(binding),
    }

    try:
        snapshot = ShadowSnapshotGate().validate(
            report=p1_report,
            repository_root=repository_root,
            selection=selection_result.selection,
            task_id=task_id,
            raw_requirement=raw_requirement,
            expected_revision=revision,
            current_revision=get_current_revision(),
        )
    except (OSError, ValueError, RuntimeError):
        return abstain("snapshot_check_failed")

    if not snapshot.allowed:
        return abstain("snapshot_gate:" + snapshot.reason)

    drops = p1_report["effective_proposed_drops"]
    graph_drops = p1_report["graph_proposed_drops"]
    rerank_drops = p1_report["rerank_proposed_drops"]

    if any(
        not isinstance(group, list) or any(not isinstance(path, str) for path in group)
        for group in (drops, graph_drops, rerank_drops)
    ):
        return abstain("invalid_proposal_schema")

    if (
        len(drops) != len(set(drops))
        or len(drops) > max_drops
        or set(drops) != set(graph_drops) | set(rerank_drops)
    ):
        return abstain("invalid_drop_set")

    if not drops:
        return abstain("no_change")

    baseline_count = len(selection_result.selection.selected_files)

    if baseline_count - len(drops) < min_files:
        return abstain("minimum_files_violated")

    context_paths = {item.relative_path for item in final_bundle.code_context}

    if len(context_paths) != len(final_bundle.code_context):
        return abstain("duplicate_context_paths")

    # 保护当前真实任务所需证据。
    # 不允许 Context 优化修改这些角色或权限。
    protected = (
        set(execution_surface.implementation_targets)
        | set(execution_surface.tests)
        | set(execution_surface.dependencies)
        | set(execution_surface.references)
        | set(obligation_evidence_files)
    )

    if strategy is not None:
        writable_tests = strategy.get("writable_test_files", [])

        if not isinstance(writable_tests, list) or any(
            not isinstance(path, str) for path in writable_tests
        ):
            return abstain("invalid_strategy_file_list")

        protected.update(writable_tests)

    # 只需要保护最终实际存在于 Context 的文件。
    protected &= context_paths

    if protected & set(drops):
        return abstain("final_context_protected_drop")

    effective_final_drops = tuple(path for path in drops if path in context_paths)

    if not effective_final_drops:
        return abstain("candidate_not_in_final_context")

    return ActiveCandidatePreview(
        status="preview_valid",
        reason="candidate_snapshot_and_protection_valid",
        proposed_selection_drops=tuple(drops),
        drops_in_final_context=effective_final_drops,
        protected_context_paths=tuple(sorted(protected)),
        p1_report=p1_report,
    )
