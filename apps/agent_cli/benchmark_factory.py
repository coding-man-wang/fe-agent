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


def create_benchmark_runner(
    *,
    fe_agent_root: Path,
    benchmark_repository: Path,
) -> BenchmarkRunner:

    llm = OpenAILLM(
        apikey=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(EmptySemanticCodeSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=6,
        lexical_max_results=20,
        semantic_top_k=8,
        max_files=8,
        max_total_bytes=200_000,
    )

    retrieval_capture_service = BenchmarkRetrievalCaptureService(
        repository_root=(benchmark_repository),
        file_indexer=(RepositoryFileIndexer()),
        retrieval_service=(retrieval_service),
    )

    return BenchmarkRunner(
        case_loader=(BenchmarkCaseLoader(fe_agent_root / "benchmark" / "cases")),
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
