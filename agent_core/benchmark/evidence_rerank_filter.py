from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from agent_core.benchmark.task_rerank_diagnostics import (
    TaskRerankDiagnosticResult,
)
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)


@dataclass(frozen=True)
class RerankFilterProposal:
    selection: HybridRelevantFileSelection
    status: Literal["proposed", "no_change", "abstained"]
    proposed_drops: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


class EvidenceProtectedRerankFilter:
    """
    v1.4.8-P2-B Shadow Filter。

    仅尝试移除缺乏必要证据、且相对排名较低的
    Semantic-only 文件。

    不读取 Benchmark Oracle。
    不修改原 Selection。
    不授予或撤销 Write Permission。
    """

    def __init__(
        self,
        *,
        bottom_fraction: float = 0.25,
        max_score_ratio: float = 0.80,
        min_candidates: int = 4,
        max_drops: int = 1,
    ):
        if not 0 < bottom_fraction < 1:
            raise ValueError("bottom_fraction 必须在 0 和 1 之间")
        if not 0 < max_score_ratio < 1:
            raise ValueError("max_score_ratio 必须在 0 和 1 之间")
        if min_candidates < 2 or max_drops < 1:
            raise ValueError("过滤配置非法")

        self.bottom_fraction = bottom_fraction
        self.max_score_ratio = max_score_ratio
        self.min_candidates = min_candidates
        self.max_drops = max_drops

    def propose(
        self,
        *,
        selection: HybridRelevantFileSelection,
        diagnostic: TaskRerankDiagnosticResult,
        protected_paths: set[str] | None = None,
    ) -> RerankFilterProposal:

        original = selection.model_copy(deep=True)

        def abstain(reason: str) -> RerankFilterProposal:
            return RerankFilterProposal(
                selection=original,
                status="abstained",
                reasons={"_decision": reason},
            )

        paths = [item.relative_path for item in selection.selected_files]
        n = len(paths)

        if diagnostic.status != "complete":
            return abstain("rerank_not_complete")

        if n < self.min_candidates:
            return abstain("insufficient_candidates")

        # 把 Rerank 结果绑定到完全相同的 Selection。
        if paths != diagnostic.candidate_paths or len(set(paths)) != n:
            return abstain("selection_mismatch")

        scores = diagnostic.ranked_files

        if (
            len(scores) != n
            or {s.relative_path for s in scores} != set(paths)
            or len({s.relative_path for s in scores}) != n
            or sorted(s.rank for s in scores) != list(range(1, n + 1))
        ):
            return abstain("invalid_rerank_result")

        # 当前试验仅接受 0~1 的 Provider 得分。
        if any(not math.isfinite(s.score) or not 0 <= s.score <= 1 for s in scores):
            return abstain("unsupported_score_scale")

        top_score = max(s.score for s in scores)

        if top_score <= 0:
            return abstain("invalid_top_score")

        items = {item.relative_path: item for item in selection.selected_files}

        # 如果诊断时的符号状态与当前 Selection 不一致，
        # 必须拒绝使用该诊断结果。
        for scored in scores:
            item = items[scored.relative_path]
            if scored.has_symbol_evidence != bool(item.symbol_candidates):
                return abstain("symbol_evidence_changed")

        bottom_start = math.ceil(n * (1 - self.bottom_fraction))

        eligible = []
        reasons = {}
        protected_paths = set(protected_paths or ())

        for scored in scores:
            item = items[scored.relative_path]
            path = scored.relative_path

            if path in protected_paths:
                reasons[path] = "protected_anchor_dependency"
                continue
            # 只尝试过滤一般源码上下文。
            if (
                item.category != "source"
                or item.context_role != "unclassified"
                or item.target_match_type != "none"
            ):
                reasons[path] = "protected_role_or_target"
                continue

            # 必要 Symbol Evidence 强制保留。
            if item.symbol_candidates:
                reasons[path] = "protected_symbol_evidence"
                continue

            # Lexical Evidence 或 Graph Evidence 强制保留。
            if item.lexical_hit_count > 0:
                reasons[path] = "protected_lexical_evidence"
                continue

            if any(
                reason.startswith("Graph dependency:")
                or reason.startswith("Graph dependent:")
                for reason in item.reasons
            ):
                reasons[path] = "protected_graph_evidence"
                continue

            # 仅针对 Semantic-only 文件试验。
            if item.semantic_hit_count <= 0:
                reasons[path] = "not_semantic_candidate"
                continue

            if scored.source_truncated:
                reasons[path] = "source_truncated"
                continue

            if scored.rank < bottom_start:
                reasons[path] = "rank_protected"
                continue

            score_ratio = scored.score / top_score

            if score_ratio >= self.max_score_ratio:
                reasons[path] = "score_ratio_protected"
                continue

            eligible.append(scored)

            reasons[path] = (
                f"candidate:rank={scored.rank};" f"score_ratio={score_ratio:.4f}"
            )

        # 单次最多提出一个删除建议。
        # 第一版先限制风险，不追求激进压缩。
        eligible.sort(
            key=lambda s: (
                s.score,
                s.relative_path,
            )
        )

        dropped = {s.relative_path for s in eligible[: self.max_drops]}

        if not dropped:
            return RerankFilterProposal(
                selection=original,
                status="no_change",
                reasons=reasons,
            )

        kept = [
            item
            for item in original.selected_files
            if item.relative_path not in dropped
        ]

        if len(kept) < 2:
            return abstain("minimum_context_not_preserved")

        filtered = original.model_copy(
            deep=True,
            update={
                "selected_files": kept,
                "selected_count": len(kept),
                "total_selected_bytes": sum(item.size_bytes for item in kept),
            },
        )

        return RerankFilterProposal(
            selection=filtered,
            status="proposed",
            proposed_drops=[
                item.relative_path
                for item in original.selected_files
                if item.relative_path in dropped
            ],
            reasons=reasons,
        )
