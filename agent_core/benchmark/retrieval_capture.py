import asyncio
from pathlib import Path

from agent_core.benchmark.retrieval_model import (
    BenchmarkRetrievalCapture,
    BenchmarkRetrievalQuery,
    BenchmarkRetrievedFile,
    BenchmarkTaskRetrieval,
)
from agent_core.codebase.chunker import (
    CodeChunker,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskPlan,
)


class BenchmarkRetrievalCaptureError(ValueError):
    pass


class BenchmarkRetrievalCaptureService:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        file_indexer: RepositoryFileIndexer,
        retrieval_service: TaskRetrievalService,
        code_chunker: CodeChunker | None = None,
        semantic_index: InMemorySemanticIndex | None = None,
    ) -> None:

        self.repository_root = Path(repository_root).resolve(strict=True)

        if not (self.repository_root.is_dir()):
            raise (BenchmarkRetrievalCaptureError("repository_root 不是目录"))

        # code_chunker 和 semantic_index
        # 必须同时存在或者同时不存在。

        if (code_chunker is None) != (semantic_index is None):
            raise ValueError("code_chunker 和 " "semantic_index " "必须同时配置")

        self.file_indexer = file_indexer

        self.retrieval_service = retrieval_service

        self.code_chunker = code_chunker

        self.semantic_index = semantic_index

        # 当前 Semantic Index
        # 对应哪个 Git Revision。
        self._semantic_revision: str | None = None

        # 避免未来并发 Case
        # 同时构建两次 Index。
        self._semantic_build_lock = asyncio.Lock()

    async def capture(
        self,
        *,
        task_plan: TaskPlan,
        raw_requirement: str,
        requirement_spec: RequirementSpec,
        repository_revision: str,
    ) -> BenchmarkRetrievalCapture:

        # -------------------------------
        # 1. 建 File Index
        # -------------------------------

        index = self.file_indexer.build(self.repository_root)

        # -------------------------------
        # 2. 确保 Semantic Index
        #    已经为当前 Commit 构建。
        # -------------------------------

        await self._ensure_semantic_index(
            index=index,
            repository_revision=(repository_revision),
        )

        captures: list[BenchmarkTaskRetrieval] = []

        # -------------------------------
        # 3. 每个 Task Retrieval
        # -------------------------------

        for task in task_plan.tasks:

            query_specs = self.retrieval_service.build_query_specs(
                task=task,
                raw_requirement=(raw_requirement),
                requirement_spec=(requirement_spec),
            )

            selection = await self.retrieval_service.retrieve(
                task=task,
                index=index,
                query_specs=(query_specs),
            )

            selected_files = [
                BenchmarkRetrievedFile(
                    relative_path=(item.relative_path),
                    category=(item.category),
                    selection_score=(item.selection_score),
                    matched_queries=list(item.matched_queries),
                    retrieval_score=(item.retrieval_score),
                    best_semantic_score=(item.best_semantic_score),
                    query_hit_count=(item.query_hit_count),
                    lexical_hit_count=(item.lexical_hit_count),
                    semantic_hit_count=(item.semantic_hit_count),
                    hybrid_hit_count=(item.hybrid_hit_count),
                    reasons=list(item.reasons),
                    matched_intents=list(item.matched_intents),
                    context_role=(item.context_role),
                    role_reason=(item.role_reason),
                    rerank_score=(item.rerank_score),
                    rerank_rank=(item.rerank_rank),
                )
                for item in selection.selected_files
            ]

            query_details = [
                BenchmarkRetrievalQuery(
                    text=query.text,
                    sources=list(query.sources),
                    kinds=list(query.kinds),
                    intents=list(query.intents),
                )
                for query in query_specs
            ]

            captures.append(
                BenchmarkTaskRetrieval(
                    task_id=(task.task_id),
                    task_type=(task.task_type),
                    title=(task.title),
                    queries=[query.text for query in query_specs],
                    query_details=(query_details),
                    selected_files=(selected_files),
                    candidate_count=(selection.candidate_count),
                    selected_count=(selection.selected_count),
                )
            )

        mode = (
            "hybrid_semantic_intent_v0.4"
            if self.semantic_index is not None
            else "lexical_intent_v0.4"
        )

        return BenchmarkRetrievalCapture(
            mode=mode,
            indexed_files=(index.indexable_files),
            task_retrievals=(captures),
        )

    async def _ensure_semantic_index(
        self,
        *,
        index,
        repository_revision: str,
    ) -> None:

        # 没配置 Semantic Search，
        # 继续走 Lexical 模式。
        if self.semantic_index is None:
            return

        if self.code_chunker is None:
            raise RuntimeError("Semantic Index 已配置，" "但 CodeChunker 缺失")

        # 当前 Commit 已经构建过。
        if self._semantic_revision == repository_revision:
            return

        async with self._semantic_build_lock:

            # 获取锁之后再检查一次。
            if self._semantic_revision == repository_revision:
                return

            chunks = self.code_chunker.build_chunks(index)

            await self.semantic_index.build(chunks)

            self._semantic_revision = repository_revision
