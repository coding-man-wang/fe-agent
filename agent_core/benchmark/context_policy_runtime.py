from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Literal, Mapping

from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.benchmark.selection_snapshot_gate import (
    ShadowSnapshotGate,
)
from agent_core.codebase.graph.anchor_dependency_protection import (
    AnchorDependencyProtection,
)
from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle
from agent_core.sdd.model import TaskDefinition

logger = logging.getLogger(__name__)

PolicyMode = Literal["off", "shadow", "active"]


class StaleContextEvidenceError(RuntimeError):
    """源码、Selection 或仓库版本已变化，必须重新检索。"""


class MissingMandatoryContextError(RuntimeError):
    """必要上下文在 Baseline 中就已缺失。"""


@dataclass(frozen=True)
class ContextRolloutConfig:
    mode: PolicyMode = "shadow"

    # Active 必须同时满足多个显式门禁。
    active_enabled: bool = False
    canary_passed: bool = False
    approved_revision: str | None = None
    task_allowlist: frozenset[str] = frozenset()

    max_drops: int = 2
    min_files: int = 2
    anchor_count: int = 5
    dependency_depth: int = 2

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("非法 Context Policy Mode")
        if self.max_drops < 1 or self.min_files < 1:
            raise ValueError("文件预算参数必须 >= 1")
        if self.anchor_count < 1 or self.dependency_depth < 1:
            raise ValueError("依赖保护参数必须 >= 1")


@dataclass(frozen=True)
class ContextPolicyDecision:
    task_id: str
    mode: PolicyMode
    status: Literal["baseline", "shadow", "active", "fallback"]
    reason: str
    snapshot_gate_reason: str
    proposed_drops: tuple[str, ...]
    context_files_before: tuple[str, ...]
    context_files_after: tuple[str, ...]


@dataclass(frozen=True)
class ContextConsumptionResult:
    bundle: TaskContextBundle
    decision: ContextPolicyDecision


def _context_paths(bundle: TaskContextBundle) -> set[str]:
    return {item.relative_path for item in bundle.code_context}


def _additional_source_hashes(
    root: Path,
    paths: set[str],
) -> dict[str, str]:
    """
    捕获 Selection 外、但下游可能读取的其他源码。

    例如 ResolvedChangeSurface 补入的文件。
    不存在的待创建文件使用固定标记。
    """
    result = {}

    for raw in sorted(paths):
        normalized = raw.replace("\\", "/")
        pure = PurePosixPath(normalized)

        if (
            not pure.parts
            or pure.is_absolute()
            or PureWindowsPath(raw).drive
            or ".." in pure.parts
        ):
            raise ValueError("非法补充源码路径")

        source = (root / Path(*pure.parts)).resolve(strict=False)

        if not source.is_relative_to(root):
            raise ValueError("补充文件超出仓库")

        if not source.exists():
            result[normalized] = "__missing__"
            continue

        if not source.is_file():
            raise ValueError("补充源码不是普通文件")

        result[normalized] = hashlib.sha256(source.read_bytes()).hexdigest()

    return result


class GuardedContextPolicyConsumer:
    """
    v1.4.9 P2-B / P3

    接收 P1 Shadow Report，基于 P2-A Gate 判断
    Candidate 是否可以用于构建 Context。

    Shadow: 验证候选，但返回 Baseline。
    Active: 显式授权、保护规则和审计成功后，
            才返回 Candidate Context。

    不修改 Change Surface 或 Write Permission。
    """

    STALE_REASONS = frozenset(
        {
            "repository_revision_changed",
            "source_snapshot_changed",
            "selection_snapshot_changed",
            "source_snapshot_unavailable",
        }
    )

    PROTECTED_ROLES = frozenset(
        {
            "implementation_target",
            "test",
            "policy",
            "reference",
        }
    )

    def __init__(
        self,
        *,
        config: ContextRolloutConfig,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.audit_sink = audit_sink
        self.snapshot_gate = ShadowSnapshotGate()

    def _protected_paths(
        self,
        result: TaskContextSelectionResult,
    ) -> set[str]:
        baseline = result.selection
        selected = {item.relative_path for item in baseline.selected_files}

        # 高优先级 Retrieval Anchor 的依赖闭包。
        protected = AnchorDependencyProtection(
            anchor_count=self.config.anchor_count,
            max_depth=self.config.dependency_depth,
        ).protected_paths(
            selection=result.retrieval_selection,
            graph=result.graph,
        )

        # 保留当前候选集合中的直接导入依赖。
        for edge in result.graph.edges:
            if edge.source_path in selected and edge.target_path in selected:
                protected.add(edge.target_path)

        # 强证据和明确职责不能被过滤。
        for item in baseline.selected_files:
            if (
                item.context_role in self.PROTECTED_ROLES
                or item.target_match_type != "none"
                or bool(item.symbol_candidates)
                or item.lexical_hit_count > 0
            ):
                protected.add(item.relative_path)

        return protected

    def _build_candidate(
        self,
        *,
        result: TaskContextSelectionResult,
        report: Mapping[str, Any],
    ) -> tuple[HybridRelevantFileSelection | None, str]:

        baseline = result.selection
        paths = {item.relative_path for item in baseline.selected_files}

        drops = report.get("effective_proposed_drops")
        graph_drops = report.get("graph_proposed_drops")
        rerank_drops = report.get("rerank_proposed_drops")

        groups = (drops, graph_drops, rerank_drops)

        if any(
            not isinstance(group, list) or any(not isinstance(x, str) for x in group)
            for group in groups
        ):
            return None, "invalid_proposal_schema"

        if (
            len(drops) != len(set(drops))
            or len(drops) > self.config.max_drops
            or not set(drops).issubset(paths)
        ):
            return None, "invalid_drop_set"

        # 有效删除集必须与两阶段输出一致，
        # 防止审计报告中的字段互相矛盾。
        if set(drops) != set(graph_drops) | set(rerank_drops):
            return None, "inconsistent_proposal"

        if not drops:
            return None, "no_change"

        protected = self._protected_paths(result)

        if protected & set(drops):
            return None, "protected_evidence_drop"

        # 与当前 Graph / Rerank 的候选范围保持一致：
        # 仅允许弱 Graph 或 Semantic Evidence 文件被删除。
        for item in baseline.selected_files:
            if item.relative_path not in drops:
                continue

            graph_evidence = any(
                reason.startswith("Graph dependency:")
                or reason.startswith("Graph dependent:")
                for reason in item.reasons
            )

            semantic_evidence = item.semantic_hit_count > 0

            if not (graph_evidence or semantic_evidence):
                return None, "unsupported_drop_evidence"

        kept = [
            item.model_copy(deep=True)
            for item in baseline.selected_files
            if item.relative_path not in drops
        ]

        if len(kept) < self.config.min_files:
            return None, "minimum_files_violated"

        candidate = baseline.model_copy(
            deep=True,
            update={
                "selected_files": kept,
                "selected_count": len(kept),
                "total_selected_bytes": sum(item.size_bytes for item in kept),
            },
        )

        return candidate, "candidate_valid"

    def consume(
        self,
        *,
        repository_root: str | Path,
        task: TaskDefinition,
        raw_requirement: str,
        selection_result: TaskContextSelectionResult,
        report: Mapping[str, Any],
        get_current_revision: Callable[[], str],
        build_context: Callable[[HybridRelevantFileSelection], TaskContextBundle],
        required_context_paths: set[str] | None = None,
        additional_source_paths: set[str] | None = None,
        fresh_in_process_report: bool = False,
    ) -> ContextConsumptionResult:

        root = Path(repository_root).resolve(strict=True)
        original = selection_result.selection
        revision = selection_result.repository_revision

        required = set(required_context_paths or ())
        additional = set(additional_source_paths or ())

        gate_reason = "not_checked"
        candidate = None
        candidate_reason = "policy_off"

        if self.config.mode != "off":
            gate = self.snapshot_gate.validate(
                report=report,
                repository_root=root,
                selection=original,
                task_id=task.task_id,
                raw_requirement=raw_requirement,
                expected_revision=revision,
                current_revision=get_current_revision(),
            )

            gate_reason = gate.reason

            if not gate.allowed:
                if gate.reason in self.STALE_REASONS:
                    raise StaleContextEvidenceError(gate.reason)

                candidate_reason = f"snapshot_gate:{gate.reason}"

            else:
                candidate, candidate_reason = self._build_candidate(
                    result=selection_result,
                    report=report,
                )

        # Context Build 之前再次保存源码状态。
        before = capture_selection_evidence(
            repository_root=root,
            selection=original,
        )
        extra_before = _additional_source_hashes(
            root,
            additional,
        )

        baseline_bundle = build_context(original.model_copy(deep=True))

        if baseline_bundle.task.task_id != task.task_id:
            raise MissingMandatoryContextError("Context Task ID 不一致")

        base_paths = _context_paths(baseline_bundle)

        if required - base_paths:
            raise MissingMandatoryContextError(
                "Baseline 已缺少必要文件: " + ", ".join(sorted(required - base_paths))
            )

        optimized_bundle = None

        if candidate is not None:
            try:
                proposed_bundle = build_context(candidate.model_copy(deep=True))

                candidate_paths = _context_paths(proposed_bundle)

                # 不能丢失 Baseline 实际提供的强证据。
                mandatory = (
                    self._protected_paths(selection_result) & base_paths
                ) | required

                if mandatory - candidate_paths:
                    candidate_reason = "context_evidence_regression"

                elif (
                    proposed_bundle.budget.used_chars
                    > baseline_bundle.budget.used_chars
                    or proposed_bundle.budget.used_files
                    > baseline_bundle.budget.used_files
                ):
                    candidate_reason = "context_budget_regression"

                else:
                    optimized_bundle = proposed_bundle

            except Exception as exc:
                # Candidate 构建失败，不能损坏 Baseline。
                candidate_reason = "candidate_build_failed:" + type(exc).__name__

        # Context Build 之后进行最终快照检查。
        after = capture_selection_evidence(
            repository_root=root,
            selection=original,
        )
        extra_after = _additional_source_hashes(
            root,
            additional,
        )

        if (
            get_current_revision() != revision
            or before != after
            or extra_before != extra_after
        ):
            # 源码变化不能靠返回旧 Baseline 来掩盖。
            # 必须交给上层重新 Retrieval。
            raise StaleContextEvidenceError("source_changed_during_context_build")

        active_authorized = (
            self.config.mode == "active"
            and self.config.active_enabled
            and self.config.canary_passed
            and self.config.approved_revision == revision
            and task.task_id in self.config.task_allowlist
            and fresh_in_process_report
            and self.audit_sink is not None
        )

        if optimized_bundle is None:
            status = "fallback"
            reason = candidate_reason
            used_bundle = baseline_bundle

        elif active_authorized:
            status = "active"
            reason = "optimized_context_accepted"
            used_bundle = optimized_bundle

        elif self.config.mode == "active":
            status = "fallback"
            reason = "active_not_authorized"
            used_bundle = baseline_bundle

        else:
            status = "shadow"
            reason = "candidate_preview_valid"
            used_bundle = baseline_bundle

        def make_decision(
            current_status: Literal["baseline", "shadow", "active", "fallback"],
            current_reason: str,
            bundle: TaskContextBundle,
        ) -> ContextPolicyDecision:
            return ContextPolicyDecision(
                task_id=task.task_id,
                mode=self.config.mode,
                status=current_status,
                reason=current_reason,
                snapshot_gate_reason=gate_reason,
                proposed_drops=tuple(
                    report.get("effective_proposed_drops", [])
                    if candidate is not None
                    else ()
                ),
                context_files_before=tuple(
                    item.relative_path for item in baseline_bundle.code_context
                ),
                context_files_after=tuple(
                    item.relative_path for item in bundle.code_context
                ),
            )

        decision = make_decision(
            status,
            reason,
            used_bundle,
        )

        audit_event = {
            "policy_version": "v1.4.9-P3",
            "mode": self.config.mode,
            "task_id": task.task_id,
            "repository_revision": revision,
            "status": "complete",
            # 这里 applied 表示 Context 优化已被采用，
            # 不表示 Patch 已写入。
            "applied": status == "active",
            "context_status": status,
            "context_reason": reason,
            "snapshot_gate_reason": gate_reason,
            "context_files_before": list(decision.context_files_before),
            "context_files_after": list(decision.context_files_after),
            "effective_proposed_drops": list(decision.proposed_drops),
            "event_type": "context_policy_decision",
        }

        audit_ok = False

        if self.audit_sink is not None:
            try:
                self.audit_sink(audit_event)
                audit_ok = True
            except Exception:
                logger.exception("Context Policy Audit Failed")

        # Active 必须先成功记录审计事件，
        # 才能返回优化后的 Context。
        if status == "active" and not audit_ok:
            used_bundle = baseline_bundle
            decision = make_decision(
                "fallback",
                "active_audit_unavailable",
                baseline_bundle,
            )

        return ContextConsumptionResult(
            bundle=used_bundle,
            decision=decision,
        )
