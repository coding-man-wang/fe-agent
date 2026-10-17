import os
from pathlib import Path

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.project_context import (
    BenchmarkProjectContextProvider,
)
from agent_core.benchmark.repository_guard import (
    BenchmarkRepositoryGuard,
)
from agent_core.benchmark.retrieval_adapter import (
    EmptySemanticCodeSearcher,
)
from agent_core.benchmark.retrieval_capture import (
    BenchmarkRetrievalCaptureService,
)
from agent_core.benchmark.retrieval_evaluator import (
    BenchmarkRetrievalEvaluator,
)
from agent_core.benchmark.runner import (
    BenchmarkRunner,
)
from agent_core.benchmark.run_store import (
    BenchmarkRunStore,
)
from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)
from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.llm.openai import (
    OpenAILLM,
)
from agent_core.sdd.plan import (
    ImplementationPlanGenerator,
)
from agent_core.sdd.requirement import (
    RequirementAnalyzer,
)
from agent_core.sdd.task import (
    TaskPlanner,
)
from agent_core.sdd.technical import (
    TechnicalSpecGenerator,
)
from agent_core.agent.retry import (
    RetryPolicy,
)

from agent_core.llm.retrying import (
    RetryingLLM,
)
from agent_core.codebase.chunker import (
    CodeChunker,
)
from agent_core.codebase.openai_embedding import (
    OpenAIEmbeddingProvider,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)
from agent_core.codebase.openai_reranker import (
    OpenAICompatibleReranker,
)

from agent_core.codebase.retrieval_rerank import (
    RetrievalRerankService,
)
from agent_core.codebase.context_role import (
    ContextRoleClassifier,
)


def create_benchmark_runner(
    *,
    fe_agent_root: Path,
    benchmark_repository: Path,
) -> BenchmarkRunner:
    provider_llm = OpenAILLM(
        apikey=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("LLM_BASE_URL"),
        # 单个 HTTP 请求最长等待 60 秒
        request_timeout_seconds=60.0,
    )

    llm = RetryingLLM(
        llm=provider_llm,
        retry_policy=RetryPolicy(
            # 第一次失败后再试 2 次
            max_retries=2,
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=5.0,
            jitter=True,
        ),
    )

    embedding_model = os.getenv("EMBEDDING_MODEL_ID")

    semantic_index = None

    code_chunker = None

    if embedding_model:

        embedding_api_key = os.getenv("EMBEDDING_API_KEY")

        if not embedding_api_key:
            raise RuntimeError("配置了 EMBEDDING_MODEL_ID，" "但缺少 EMBEDDING_API_KEY")

        embedding_provider = OpenAIEmbeddingProvider(
            model=embedding_model,
            api_key=(embedding_api_key),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            batch_size=20,
            request_timeout_seconds=60,
        )

        semantic_index = InMemorySemanticIndex(embedding_provider)

        semantic_searcher = semantic_index

        code_chunker = CodeChunker(
            max_lines=40,
            overlap_lines=5,
        )

    else:

        semantic_searcher = EmptySemanticCodeSearcher()
    rerank_model = os.getenv("RERANK_MODEL_ID")

    rerank_service = None

    if rerank_model:
        reranker = OpenAICompatibleReranker(
            model=rerank_model,
            api_key=os.getenv("RERANK_API_KEY"),
            base_url=os.getenv("RERANK_BASE_URL"),
            request_timeout_seconds=60,
        )

        rerank_service = RetrievalRerankService(reranker=reranker)

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=semantic_searcher,
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=6,
        lexical_max_results=20,
        semantic_top_k=8,
        max_files=8,
        max_total_bytes=200_000,
        rerank_service=(rerank_service),
        rerank_candidate_limit=16,
        context_role_classifier=(ContextRoleClassifier()),
    )

    retrieval_capture_service = BenchmarkRetrievalCaptureService(
        repository_root=(benchmark_repository),
        file_indexer=(RepositoryFileIndexer()),
        retrieval_service=(retrieval_service),
        code_chunker=(code_chunker),
        semantic_index=(semantic_index),
    )

    return BenchmarkRunner(
        case_loader=(BenchmarkCaseLoader(fe_agent_root / "benchmark")),
        repository_guard=(
            BenchmarkRepositoryGuard(repository_root=(benchmark_repository))
        ),
        project_context_provider=(
            BenchmarkProjectContextProvider(repository_root=(benchmark_repository))
        ),
        requirement_analyzer=(RequirementAnalyzer(llm=llm)),
        technical_spec_generator=(TechnicalSpecGenerator(llm=llm)),
        implementation_plan_generator=(ImplementationPlanGenerator(llm=llm)),
        task_planner=(TaskPlanner(llm=llm)),
        retrieval_capture_service=(retrieval_capture_service),
        retrieval_evaluator=(BenchmarkRetrievalEvaluator()),
        run_store=(BenchmarkRunStore(fe_agent_root)),
    )
