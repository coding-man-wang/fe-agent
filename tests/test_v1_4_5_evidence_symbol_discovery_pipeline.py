from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.artifacts.query_builder import (
    ArtifactAwareRetrievalQueryBuilder,
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
from agent_core.codebase.model import (
    CodeChunk,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from agent_core.codebase.symbol_discovery import (
    RetrievalEvidenceSymbolMapper,
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


class FakeSemanticSearcher:
    """
    Deterministic Semantic Searcher。

    这里不测试 Embedding Quality。

    只测试：

        Natural Language Query
            ↓
        Semantic Evidence
            ↓
        File Retrieval
            ↓
        Symbol Mapping
    """

    def __init__(
        self,
        *,
        orders_chunk: CodeChunk,
    ) -> None:

        self.orders_chunk = orders_chunk

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> SemanticSearchResponse:

        return SemanticSearchResponse(
            query=query,
            results=[
                SemanticSearchResult(
                    chunk=(self.orders_chunk),
                    score=0.95,
                )
            ],
        )


def write_file(
    root: Path,
    relative_path: str,
    content: str,
) -> None:

    path = root / relative_path

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )


def create_repository(
    root: Path,
) -> None:

    write_file(
        root,
        "src/api/orders.ts",
        """
import type { Order } from "../types/domain"

const useMock = true

export function listOrders(): Promise<Order[]> {
  if (useMock) {
    return []
  }

  return []
}
""",
    )

    write_file(
        root,
        "src/api/users.ts",
        """
export function listUsers() {
  return []
}
""",
    )

    write_file(
        root,
        "src/types/domain.ts",
        """
export interface Order {
  id: string
  status: string
}
""",
    )


def build_requirement() -> RequirementSpec:

    return RequirementSpec(
        title="订单列表状态筛选",
        summary=("为订单列表增加按状态筛选能力，" "保持现有订单加载行为"),
        goals=[
            "支持订单状态过滤",
        ],
        non_goals=[
            "不修改用户列表",
        ],
        acceptance_criteria=["订单列表只显示" "符合状态条件的数据"],
        constraints=[
            "不修改用户列表",
        ],
    )


def build_task() -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="实现订单状态筛选",
        objective=("让订单列表支持" "按状态过滤"),
        source_step_order=1,
        responsibilities=[
            ("定位订单列表的" "数据加载责任"),
            ("实现订单状态" "过滤行为"),
        ],
        investigation_questions=["订单列表的数据" "从哪里加载？"],
        repository_hints=[],
        target_modules=[],
        expected_outputs=["订单状态筛选可用"],
        verification=["验证订单筛选行为"],
    )


@pytest.mark.asyncio
async def test_natural_language_retrieval_evidence_discovers_list_orders(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    # ==================================
    # 真实源码位置。
    #
    # 注意：
    # 这里 Fake 的只是 Semantic Ranking，
    # Chunk 自身仍来自真实 Repository
    # 的位置语义。
    # ==================================

    orders_chunk = CodeChunk(
        chunk_id=("src/api/orders.ts:5-11"),
        relative_path=("src/api/orders.ts"),
        start_line=5,
        end_line=11,
        content=(
            "export function "
            "listOrders(): "
            "Promise<Order[]> {\n"
            "  if (useMock) {\n"
            "    return []\n"
            "  }\n"
            "\n"
            "  return []\n"
            "}"
        ),
        language="TypeScript",
        category="source",
    )

    semantic_searcher = FakeSemanticSearcher(orders_chunk=(orders_chunk))

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(semantic_searcher),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        query_builder=(ArtifactAwareRetrievalQueryBuilder(max_queries=6)),
        max_queries=6,
        lexical_max_results=20,
        semantic_top_k=8,
        max_files=8,
        max_total_bytes=200_000,
    )

    requirement = build_requirement()

    task = build_task()

    raw_requirement = "给订单列表增加状态筛选，" "不要修改用户列表"

    query_specs = retrieval_service.build_query_specs(
        task=task,
        raw_requirement=(raw_requirement),
        requirement_spec=(requirement),
    )

    # ==================================
    # 确保 QueryBuilder 没有作弊。
    # ==================================

    assert not any("listorders" in query.text.casefold() for query in query_specs)

    selection = await retrieval_service.retrieve(
        task=task,
        index=index,
        query_specs=(query_specs),
        semantic_enabled=True,
    )

    selected_paths = [item.relative_path for item in selection.selected_files]

    assert "src/api/orders.ts" in selected_paths

    orders_file = next(
        item
        for item in selection.selected_files
        if item.relative_path == "src/api/orders.ts"
    )

    # ==================================
    # File Retrieval 必须保留
    # Chunk-level Evidence。
    # ==================================

    assert orders_file.best_semantic_chunk is not None

    assert orders_file.best_semantic_chunk.relative_path == "src/api/orders.ts"

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("evidence-pipeline-test"),
    )

    evidence_candidates = RetrievalEvidenceSymbolMapper().discover(
        selection=selection,
        symbol_index=(snapshot.symbol_index),
    )

    ids = {candidate.symbol_id for candidate in evidence_candidates}

    # ==================================
    # 最终：
    #
    # Natural Language
    #     ↓
    # Semantic Evidence
    #     ↓
    # Source Range
    #     ↓
    # listOrders
    # ==================================

    assert LIST_ORDERS_ID in ids

    list_orders = next(
        candidate
        for candidate in evidence_candidates
        if candidate.symbol_id == LIST_ORDERS_ID
    )

    assert list_orders.source == "evidence"

    # 仍然没有让 QueryBuilder
    # 猜函数名。
    assert not any(
        "listorders" in query.casefold() for query in list_orders.matched_queries
    )
