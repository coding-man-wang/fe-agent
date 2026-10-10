from typing import Protocol

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)
from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
    RepositoryFileIndex,
    SemanticSearchResponse,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class SemanticCodeSearcher(Protocol):
    async def search(
            self,
            query: str,
            *,
            top_k: int = 5,
    ) -> SemanticSearchResponse: ...


class TaskRetrievalService:
    """
    根据 TaskDefinition 执行完整的
    Hybrid Retrieval Pipeline。

    职责：

    Task
      -> Retrieval Queries
      -> Lexical Retrieval
      -> Semantic Retrieval
      -> RRF Fusion
      -> Relevant File Selection

    本类不负责：

    - Repository Scan
    - File Index 构建
    - Semantic Index 构建
    - 文件内容读取
    - TaskContextBundle 构建
    """

    def __init__(
            self,
            *,
            exact_searcher: ExactCodeSearcher,
            semantic_searcher: SemanticCodeSearcher,
            hybrid_retriever: HybridRetriever,
            selector: HybridRelevantFileSelector,
            max_queries: int = 6,
            lexical_max_results: int = 20,
            semantic_top_k: int = 8,
            max_files: int = 8,
            max_total_bytes: int = 200_000,
    ) -> None:
        if max_queries < 1:
            raise ValueError("max_queries 必须 >= 1")

        if lexical_max_results < 1:
            raise ValueError("lexical_max_results 必须 >= 1")

        if semantic_top_k < 1:
            raise ValueError("semantic_top_k 必须 >= 1")

        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_total_bytes < 1:
            raise ValueError("max_total_bytes 必须 >= 1")

        self.exact_searcher = exact_searcher

        self.semantic_searcher = semantic_searcher

        self.hybrid_retriever = hybrid_retriever

        self.selector = selector

        self.max_queries = max_queries

        self.lexical_max_results = lexical_max_results

        self.semantic_top_k = semantic_top_k

        self.max_files = max_files

        self.max_total_bytes = max_total_bytes

    async def retrieve(
            self,
            *,
            task: TaskDefinition,
            index: RepositoryFileIndex,
    ) -> HybridRelevantFileSelection:

        queries = self.build_queries(task)

        hybrid_responses = []

        for query in queries:
            lexical = self.exact_searcher.search(
                index,
                query,
                max_result=(self.lexical_max_results),
            )

            semantic = await self.semantic_searcher.search(
                query,
                top_k=(self.semantic_top_k),
            )

            hybrid = self.hybrid_retriever.fuse(
                query=query,
                lexical=lexical,
                semantic=semantic,
            )

            hybrid_responses.append(hybrid)

        return self.selector.select(
            index=index,
            hybrid_responses=(hybrid_responses),
            task=task,
            max_files=self.max_files,
            max_total_bytes=(self.max_total_bytes),
        )

    def build_queries(
            self,
            task: TaskDefinition,
    ) -> list[str]:
        """
        v0.1 使用确定性 Query Builder。

        暂时不调用 LLM，
        保证 Retrieval 行为：
        - 可测试
        - 可复现
        - 可解释
        """

        candidates = [
            task.title,
            task.objective,
            *task.target_modules,
            *task.inputs,
        ]

        queries: list[str] = []

        seen: set[str] = set()

        for candidate in candidates:
            query = candidate.strip()

            if not query:
                continue

            normalized = query.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            queries.append(query)

            if len(queries) >= self.max_queries:
                break

        if not queries:
            raise ValueError("Task 无法生成 Retrieval Query")

        return queries
