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
from agent_core.codebase.retrieval_query import (
    RetrievalQuery,
    RetrievalQueryBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)
from agent_core.codebase.retrieval_rerank import RetrievalRerankService
from agent_core.codebase.context_role import (
    ContextRoleClassifier,
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
    根据任务和需求上下文执行 Retrieval。

    v0.2：

    Retrieval Query 不再只来自 Task。

    同时可以来自：

    - Raw Requirement
    - RequirementSpec
    - TaskDefinition
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
        query_builder: RetrievalQueryBuilder | None = None,
        rerank_service: RetrievalRerankService | None = None,
        rerank_candidate_limit: int = 16,
        context_role_classifier: ContextRoleClassifier | None = None,
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

        self.query_builder = query_builder or RetrievalQueryBuilder(
            max_queries=max_queries
        )
        self.rerank_service = rerank_service

        self.rerank_candidate_limit = rerank_candidate_limit

        self.context_role_classifier = context_role_classifier

    async def retrieve(
        self,
        *,
        task: TaskDefinition,
        index: RepositoryFileIndex,
        raw_requirement: str | None = None,
        requirement_spec: RequirementSpec | None = None,
        query_specs: list[RetrievalQuery] | None = None,
        semantic_enabled: bool = True,
    ) -> HybridRelevantFileSelection:

        # Capture 可以提前生成 Query，
        # 再把同一组 Query 传进来。
        #
        # 这样：
        #
        # Benchmark 记录的 Query
        #
        # 和
        #
        # 实际执行 Retrieval 的 Query
        #
        # 一定完全一致。

        if query_specs is None:

            query_specs = self.build_query_specs(
                task=task,
                raw_requirement=(raw_requirement),
                requirement_spec=(requirement_spec),
            )

        hybrid_responses = []

        for query_spec in query_specs:

            query = query_spec.text

            lexical = self.exact_searcher.search(
                index,
                query,
                max_result=(self.lexical_max_results),
            )

            if semantic_enabled:
                semantic = await self.semantic_searcher.search(
                    query,
                    top_k=self.semantic_top_k,
                )
            else:
                semantic = SemanticSearchResponse(
                    query=query,
                    results=[],
                )

            hybrid = self.hybrid_retriever.fuse(
                query=query,
                lexical=lexical,
                semantic=semantic,
            )

            hybrid_responses.append(hybrid)

        candidate_limit = (
            self.rerank_candidate_limit
            if self.rerank_service is not None
            else self.max_files
        )

        selection = self.selector.select(
            index=index,
            hybrid_responses=(hybrid_responses),
            task=task,
            query_specs=query_specs,
            max_files=candidate_limit,
            max_total_bytes=(self.max_total_bytes * 2),
        )

        if self.rerank_service is None:
            return self._classify_roles(
                selection=selection,
                task=task,
            )

        rerank_query = self._select_rerank_query(
            query_specs=query_specs,
            task=task,
        )

        reranked = await self.rerank_service.rerank(
            query=rerank_query,
            repository_root=(index.root_path),
            selection=selection,
            top_n=self.max_files,
        )

        return self._classify_roles(
            selection=reranked,
            task=task,
        )

    @staticmethod
    def _select_rerank_query(
        *,
        query_specs: list[RetrievalQuery],
        task: TaskDefinition,
    ) -> str:

        # 优先用户原始需求。
        for query in query_specs:

            if "raw_requirement" in query.sources:
                return query.text

        # 其次 Task objective。
        if task.objective.strip():
            return task.objective

        return task.title

    def build_query_specs(
        self,
        *,
        task: TaskDefinition,
        raw_requirement: str | None = None,
        requirement_spec: RequirementSpec | None = None,
    ) -> list[RetrievalQuery]:

        return self.query_builder.build(
            task=task,
            raw_requirement=(raw_requirement),
            requirement_spec=(requirement_spec),
        )

    def build_queries(
        self,
        task: TaskDefinition,
        *,
        raw_requirement: str | None = None,
        requirement_spec: RequirementSpec | None = None,
    ) -> list[str]:
        """
        保留旧接口。

        其他旧代码如果只需要：
            list[str]

        仍然可以继续使用。
        """

        query_specs = self.build_query_specs(
            task=task,
            raw_requirement=(raw_requirement),
            requirement_spec=(requirement_spec),
        )

        return [query.text for query in query_specs]

    def _classify_roles(
        self,
        *,
        selection: HybridRelevantFileSelection,
        task: TaskDefinition,
    ) -> HybridRelevantFileSelection:

        if self.context_role_classifier is None:
            return selection

        return self.context_role_classifier.classify_selection(
            selection=selection,
            task=task,
        )
