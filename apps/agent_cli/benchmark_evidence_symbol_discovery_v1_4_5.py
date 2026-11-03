from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_core.artifacts.query_builder import (
    ArtifactAwareRetrievalQueryBuilder,
)
from agent_core.codebase.chunker import (
    CodeChunker,
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
from agent_core.codebase.openai_embedding import (
    OpenAIEmbeddingProvider,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)
from agent_core.codebase.symbol_discovery import (
    RetrievalEvidenceSymbolMapper,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolRanker,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)

LIST_ORDERS_ID = "src/api/orders.ts" "::function" "::listOrders"


RAW_REQUIREMENT = "给订单列表增加状态筛选，" "不要修改用户列表"


def resolve_repository_root() -> Path:

    configured = os.getenv("FE_AGENT_BENCHMARK_WEB_ROOT")

    if configured:

        root = Path(configured).resolve()

    else:

        agent_root = Path(__file__).resolve().parents[2]

        root = (agent_root.parent / "fe-agent-benchmark-web").resolve()

    if not root.is_dir():

        raise RuntimeError("找不到 " "fe-agent-benchmark-web")

    return root


def build_requirement() -> RequirementSpec:

    return RequirementSpec(
        title="订单列表状态筛选",
        summary=("为订单列表增加按状态" "筛选能力，" "保持现有订单加载行为"),
        goals=[
            "支持订单列表按状态过滤",
            "保持订单数据加载能力",
        ],
        non_goals=[
            "不修改用户列表功能",
        ],
        acceptance_criteria=[
            ("选择订单状态后，" "只展示匹配状态的订单"),
            ("订单列表仍能正常加载"),
        ],
        constraints=[
            "不要修改用户列表功能",
        ],
    )


def build_task() -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-ORDER-FILTER",
        task_type="code_change",
        title="实现订单列表状态筛选",
        objective=("让订单列表支持" "按状态过滤，" "保持现有加载行为"),
        source_step_order=1,
        responsibilities=[
            ("定位订单列表的数据" "加载责任"),
            ("实现订单状态过滤行为"),
        ],
        investigation_questions=[
            ("订单列表的数据" "从哪里加载？"),
            ("订单状态在哪里" "定义和展示？"),
        ],
        repository_hints=[],
        target_modules=[],
        expected_outputs=[
            "订单状态筛选可用",
        ],
        verification=[
            "验证订单筛选行为",
        ],
    )


async def main() -> None:

    load_dotenv()

    repository_root = resolve_repository_root()

    embedding_model = os.getenv("EMBEDDING_MODEL_ID")

    embedding_api_key = os.getenv("EMBEDDING_API_KEY")

    if not embedding_model:

        raise RuntimeError("当前 Benchmark 必须配置 " "EMBEDDING_MODEL_ID")

    if not embedding_api_key:

        raise RuntimeError("当前 Benchmark 必须配置 " "EMBEDDING_API_KEY")

    print(
        "Repository:",
        repository_root,
    )

    # ==================================
    # 1. Repository File Index
    # ==================================

    index = RepositoryFileIndexer().build(repository_root)

    # ==================================
    # 2. Semantic Index
    #
    # 与 benchmark_factory
    # 使用同样的基础组件。
    # ==================================

    embedding_provider = OpenAIEmbeddingProvider(
        model=(embedding_model),
        api_key=(embedding_api_key),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        batch_size=20,
        request_timeout_seconds=60,
    )

    semantic_index = InMemorySemanticIndex(embedding_provider)

    code_chunker = CodeChunker(
        max_lines=40,
        overlap_lines=5,
    )

    chunks = code_chunker.build_chunks(index)

    await semantic_index.build(chunks)

    # ==================================
    # 3. Production-shaped
    # File Retrieval
    # ==================================

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(semantic_index),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=6,
        lexical_max_results=20,
        semantic_top_k=8,
        max_files=8,
        max_total_bytes=200_000,
        query_builder=(ArtifactAwareRetrievalQueryBuilder(max_queries=6)),
    )

    requirement = build_requirement()

    task = build_task()

    query_specs = retrieval_service.build_query_specs(
        task=task,
        raw_requirement=(RAW_REQUIREMENT),
        requirement_spec=(requirement),
    )

    selection = await retrieval_service.retrieve(
        task=task,
        index=index,
        query_specs=(query_specs),
        semantic_enabled=True,
    )

    # ==================================
    # 4. Symbol Snapshot
    # ==================================

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("real-evidence-" "symbol-discovery"),
    )

    # ==================================
    # 5. Name-only Baseline
    # ==================================

    name_candidates = SymbolRanker().rank(
        queries=[query.text for query in query_specs],
        symbol_index=(snapshot.symbol_index),
    )

    # ==================================
    # 6. Evidence-driven Discovery
    # ==================================

    evidence_candidates = RetrievalEvidenceSymbolMapper().discover(
        selection=selection,
        symbol_index=(snapshot.symbol_index),
    )

    name_ids = [candidate.symbol_id for candidate in name_candidates]

    evidence_ids = [candidate.symbol_id for candidate in evidence_candidates]

    retrieval_files = []

    for item in selection.selected_files:

        chunk = item.best_semantic_chunk

        retrieval_files.append(
            {
                "relative_path": (item.relative_path),
                "retrieval_score": (item.retrieval_score),
                "best_semantic_score": (item.best_semantic_score),
                "best_semantic_chunk": (
                    None
                    if chunk is None
                    else {
                        "chunk_id": (chunk.chunk_id),
                        "start_line": (chunk.start_line),
                        "end_line": (chunk.end_line),
                    }
                ),
                "matched_queries": (item.matched_queries),
            }
        )

    output = {
        "raw_requirement": (RAW_REQUIREMENT),
        "generated_queries": [query.text for query in query_specs],
        "query_contains_list_orders": (
            any("listorders" in query.text.casefold() for query in query_specs)
        ),
        "retrieval_files": (retrieval_files),
        "name_candidates": (name_ids),
        "evidence_candidates": (evidence_ids),
        "name_direct_hit": (LIST_ORDERS_ID in name_ids),
        "evidence_hit": (LIST_ORDERS_ID in evidence_ids),
        "discovery_gap_closed": (
            LIST_ORDERS_ID not in name_ids and LIST_ORDERS_ID in evidence_ids
        ),
    }

    print()

    print("=" * 80)

    print("Real Evidence-driven " "Symbol Discovery Benchmark")

    print("=" * 80)

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":

    asyncio.run(main())
