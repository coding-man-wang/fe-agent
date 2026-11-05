from __future__ import annotations

from dataclasses import dataclass, field

from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
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
        graph: RepositoryDependencyGraph | None = None,
        protected_paths: set[str] | None = None,
    ) -> GraphFilterProposal:
        """
        P3-D4: Dependency-aware Graph Shadow Filter.

        graph=None 时保持旧实验行为，用于对照测试。
        真实 P3 Benchmark 必须传入 graph。
        """

        selected_paths = {item.relative_path for item in selection.selected_files}

        # P3 的 Anchor Dependency Protection 结果。
        anchor_protected = set(protected_paths or ())

        # 当前 Selection 内存在实际导入关系的文件，
        # 不能仅因缺少 Symbol Evidence 就删除。
        direct_import_protected = set()

        if graph is not None:
            for edge in graph.edges:
                if (
                    edge.source_path in selected_paths
                    and edge.target_path in selected_paths
                ):
                    direct_import_protected.add(edge.target_path)

        kept = []
        dropped = []
        reasons = {}

        for item in selection.selected_files:
            path = item.relative_path

            if path in anchor_protected:
                kept.append(item)
                reasons[path] = "protected_anchor_dependency"
                continue

            if path in direct_import_protected:
                kept.append(item)
                reasons[path] = "protected_direct_import"
                continue

            if self._is_weak_graph_only(item):
                dropped.append(path)
                reasons[path] = (
                    "仅有 Graph Evidence，"
                    "无直接检索或符号证据，"
                    "且不在受保护的依赖集合中"
                )
                continue

            kept.append(item)

        if not kept:
            return GraphFilterProposal(
                selection=selection.model_copy(deep=True),
                reasons={"_decision": "empty_selection_guard"},
            )

        filtered = selection.model_copy(
            deep=True,
            update={
                "selected_files": [item.model_copy(deep=True) for item in kept],
                "selected_count": len(kept),
                "total_selected_bytes": sum(item.size_bytes for item in kept),
            },
        )

        return GraphFilterProposal(
            selection=filtered,
            dropped_paths=dropped,
            reasons=reasons,
        )
