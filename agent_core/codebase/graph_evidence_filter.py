from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)


@dataclass(frozen=True)
class GraphFilterProposal:
    selection: HybridRelevantFileSelection
    dropped_paths: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


class GraphEvidenceFilter:
    """
    v1.4.8-P1: Shadow-mode Graph Evidence Filter.

    只处理 Graph-only 的低证据文件。
    不修改 Retrieval 排序，不授予写权限。

    重要：当前只是候选过滤策略，
    正式启用前必须经过独立 Benchmark。
    """

    @staticmethod
    def _is_weak_graph_only(
        item: HybridRelevantFile,
    ) -> bool:
        # 不影响强目标证据
        if item.target_match_type != "none":
            return False

        # 不影响明确的特殊职责
        if item.context_role in {
            "implementation_target",
            "test",
            "policy",
        }:
            return False

        # 当前仅处理源码依赖
        if item.category != "source":
            return False

        # 存在 Symbol Evidence，优先保留
        if item.symbol_candidates:
            return False

        # 存在直接 Lexical / Semantic Evidence
        if (
            item.lexical_hit_count > 0
            or item.semantic_hit_count > 0
            or item.query_hit_count > 0
        ):
            return False

        # 需要有明确的 Graph 来源
        is_graph = any(
            reason.startswith("Graph dependency:")
            or reason.startswith("Graph dependent:")
            for reason in item.reasons
        )

        return is_graph

    def propose(
        self,
        *,
        selection: HybridRelevantFileSelection,
    ) -> GraphFilterProposal:

        kept = []
        dropped = []
        reasons = {}

        for item in selection.selected_files:
            if self._is_weak_graph_only(item):
                dropped.append(item.relative_path)
                reasons[item.relative_path] = (
                    "仅有 Graph Evidence；" "缺少独立 Retrieval 和 Symbol Evidence"
                )
                continue

            kept.append(item)

        # 不允许把 Selection 过滤为空
        if not kept:
            return GraphFilterProposal(
                selection=selection.model_copy(deep=True),
                reasons={"selection": ("过滤后无剩余文件；" "保留原 Selection")},
            )

        filtered = selection.model_copy(
            deep=True,
            update={
                "selected_files": kept,
                "selected_count": len(kept),
                "total_selected_bytes": sum(item.size_bytes for item in kept),
            },
        )

        return GraphFilterProposal(
            selection=filtered,
            dropped_paths=dropped,
            reasons=reasons,
        )
