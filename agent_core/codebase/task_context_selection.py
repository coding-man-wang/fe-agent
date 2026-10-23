from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from agent_core.codebase.chunker import (
    CodeChunker,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.graph import (
    RepositoryDependencyGraphBuilder,
    RepositoryGraphContextExpander,
)
from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
)
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)

ContextRetrievalMode = Literal[
    "hybrid",
    "lexical_only",
    "lexical_fallback",
]


@dataclass(frozen=True)
class TaskContextSelectionResult:
    """
    某个 Task 基于当前 Repository 版本，
    重新计算出的代码阅读范围。

    index:
        当前代码目录的文件索引。

    retrieval_selection:
        只经过关键词 / 语义检索得到的结果。

    graph:
        当前代码版本重新生成的文件依赖关系。

    selection:
        Retrieval + Graph 合并后的最终结果。
        这才是真正给后续 Context 使用的文件集合。
    """

    repository_revision: str

    index: RepositoryFileIndex

    retrieval_selection: HybridRelevantFileSelection

    graph: RepositoryDependencyGraph

    selection: HybridRelevantFileSelection

    retrieval_mode: ContextRetrievalMode

    semantic_error: str | None


class TaskContextSelectionService:
    """
    根据“当前最新代码”，重新决定一个 Task
    应该阅读哪些文件。

    每次 Repository Revision 改变后：

        重新扫描文件
            ↓
        必要时重建语义搜索数据
            ↓
        重新 Retrieval
            ↓
        重新建立 Dependency Graph
            ↓
        Graph Context Expansion
            ↓
        得到最终 Selection
    """

    def __init__(
        self,
        *,
        file_indexer: RepositoryFileIndexer,
        retrieval_service: TaskRetrievalService,
        graph_builder: RepositoryDependencyGraphBuilder | None = None,
        graph_expander: RepositoryGraphContextExpander | None = None,
        code_chunker: CodeChunker | None = None,
        semantic_index: InMemorySemanticIndex | None = None,
    ) -> None:

        # 语义搜索需要：
        #
        # 1. CodeChunker：把代码切成小块
        # 2. SemanticIndex：保存这些代码块的向量
        #
        # 两个必须同时配置。
        if (code_chunker is None) != (semantic_index is None):
            raise ValueError("code_chunker 和 semantic_index " "必须同时配置或同时省略")

        self.file_indexer = file_indexer

        self.retrieval_service = retrieval_service

        self.graph_builder = graph_builder or RepositoryDependencyGraphBuilder()

        self.graph_expander = graph_expander or RepositoryGraphContextExpander(
            max_files=8,
            max_graph_files=3,
            min_retrieval_files=5,
            dependency_depth=2,
            dependent_depth=1,
        )

        self.code_chunker = code_chunker

        self.semantic_index = semantic_index

        # 当前语义索引对应哪个 Commit。
        #
        # 如果 Commit 没变，不需要重新调用
        # Embedding API。
        self._semantic_revision: str | None = None

        self._semantic_build_lock = asyncio.Lock()

    async def select(
        self,
        *,
        repository_root: str | Path,
        repository_revision: str,
        task: TaskDefinition,
        raw_requirement: str,
        requirement_spec: RequirementSpec,
    ) -> TaskContextSelectionResult:
        """
        根据当前 Repository Revision，
        完整重新计算当前 Task 的 Context Selection。
        """

        root = Path(repository_root).resolve(strict=True)

        if not root.is_dir():
            raise ValueError("repository_root 不是目录")

        revision = repository_revision.strip()

        if not revision:
            raise ValueError("repository_revision 不能为空")

        # ==================================
        # 1. 重新扫描当前代码
        # ==================================

        index = self.file_indexer.build(root)

        # ==================================
        # 2. 如果启用了语义搜索，
        #    保证语义索引对应当前 Commit
        # ==================================

        (
            semantic_enabled,
            semantic_error,
        ) = await self._ensure_semantic_index(
            index=index,
            repository_revision=revision,
        )

        if self.semantic_index is None:
            retrieval_mode: ContextRetrievalMode = "lexical_only"

        elif semantic_enabled:
            retrieval_mode = "hybrid"

        else:
            retrieval_mode = "lexical_fallback"

        # ==================================
        # 3. 重新生成检索 Query
        #
        # Query：
        # 可以简单理解成：
        # “这次拿什么问题去搜索代码”
        # ==================================

        query_specs = self.retrieval_service.build_query_specs(
            task=task,
            raw_requirement=(raw_requirement),
            requirement_spec=(requirement_spec),
        )

        # ==================================
        # 4. 基于当前代码重新找相关文件
        # ==================================

        retrieval_selection = await self.retrieval_service.retrieve(
            task=task,
            index=index,
            query_specs=query_specs,
            semantic_enabled=(semantic_enabled),
        )

        # ==================================
        # 5. 基于当前代码重新建立依赖关系
        # ==================================

        graph = self.graph_builder.build(
            file_index=index,
            repository_revision=revision,
        )

        # ==================================
        # 6. 根据文件依赖关系补充 Context
        # ==================================

        selection = self.graph_expander.expand(
            selection=(retrieval_selection),
            index=index,
            graph=graph,
        )

        return TaskContextSelectionResult(
            repository_revision=revision,
            index=index,
            retrieval_selection=(retrieval_selection),
            graph=graph,
            selection=selection,
            retrieval_mode=retrieval_mode,
            semantic_error=semantic_error,
        )

    async def _ensure_semantic_index(
        self,
        *,
        index: RepositoryFileIndex,
        repository_revision: str,
    ) -> tuple[
        bool,
        str | None,
    ]:
        """
        保证语义索引对应当前代码版本。

        返回：

            (True, None)
                当前版本语义索引可用。

            (False, None)
                本来就没有配置语义搜索。

            (False, error)
                配置了语义搜索，
                但是本次外部请求失败，
                当前 Task 应降级到关键词搜索。

        特别注意：

        失败以后不能继续使用旧 Commit
        的 semantic_index。
        """

        if self.semantic_index is None:
            return (
                False,
                None,
            )

        if self.code_chunker is None:
            raise RuntimeError("semantic_index 已配置，" "但 code_chunker 缺失")

        if self._semantic_revision == repository_revision:
            return (
                True,
                None,
            )

        async with self._semantic_build_lock:

            if self._semantic_revision == repository_revision:
                return (
                    True,
                    None,
                )

            chunks = self.code_chunker.build_chunks(index)

            try:
                await self.semantic_index.build(chunks)

            except RuntimeError as exc:
                # Embedding 外部请求失败。
                #
                # 不更新 _semantic_revision，
                # 这样绝不会把旧索引误认为
                # 当前 Commit 的索引。
                return (
                    False,
                    (f"{type(exc).__name__}: " f"{exc}"),
                )

            # 只有真正成功以后
            # 才更新 revision。
            self._semantic_revision = repository_revision

            return (
                True,
                None,
            )
