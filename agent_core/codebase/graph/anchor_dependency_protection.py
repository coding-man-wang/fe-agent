from __future__ import annotations

from collections import defaultdict

from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
)
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)


class AnchorDependencyProtection:
    """
    从高优先级 Retrieval Anchor 出发，
    保护实际导入依赖。

    保护的是阅读上下文证据，
    不是文件修改权限。
    """

    RUNTIME_IMPORT_KINDS = frozenset(
        {
            "value",
            "side_effect",
            "dynamic",
            "reexport",
        }
    )

    def __init__(
        self,
        *,
        anchor_count: int = 5,
        max_depth: int = 2,
    ):
        if anchor_count < 1 or max_depth < 1:
            raise ValueError("保护参数必须 >= 1")

        self.anchor_count = anchor_count
        self.max_depth = max_depth

    def protected_paths(
        self,
        *,
        selection: HybridRelevantFileSelection,
        graph: RepositoryDependencyGraph,
    ) -> set[str]:

        ranked = sorted(
            selection.selected_files,
            key=lambda item: (
                item.rerank_rank or item.retrieval_rank,
                item.relative_path,
            ),
        )

        # 当前任务最强的直接检索证据
        anchors = {item.relative_path for item in ranked[: self.anchor_count]}

        adjacency = defaultdict(set)

        for edge in graph.edges:
            # 避免共享类型文件把所有领域连接起来。
            # type-only 依赖留给其他 Context Policy 处理。
            if edge.import_kind in self.RUNTIME_IMPORT_KINDS:
                adjacency[edge.source_path].add(edge.target_path)

        protected = set(anchors)
        frontier = set(anchors)

        for _ in range(self.max_depth):
            next_frontier = set()

            for path in frontier:
                next_frontier.update(adjacency.get(path, set()))

            next_frontier -= protected

            if not next_frontier:
                break

            protected.update(next_frontier)
            frontier = next_frontier

        return protected
