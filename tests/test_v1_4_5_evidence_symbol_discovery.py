from __future__ import annotations

from pathlib import Path

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_discovery import (
    RetrievalEvidenceSymbolMapper,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)

LIST_ORDERS_ID = "src/api/orders.ts" "::function" "::listOrders"

MOCK_LIST_ORDERS_ID = "src/mocks/service.ts" "::function" "::mockListOrders"


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
import {
  mockListOrders
} from "../mocks/service"

const useMock = true

export function listOrders() {
  if (useMock) {
    return mockListOrders()
  }

  return []
}
""",
    )

    write_file(
        root,
        "src/mocks/service.ts",
        """
export function mockListOrders() {
  return []
}

export function mockListUsers() {
  return []
}
""",
    )


def build_snapshot(
    root: Path,
):

    index = RepositoryFileIndexer().build(root)

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("evidence-test"),
    )

    return (
        index,
        snapshot,
    )


def build_selection(
    *,
    index,
    with_chunk: bool = True,
):

    record = next(
        item for item in index.files if item.relative_path == "src/api/orders.ts"
    )

    chunk = None

    if with_chunk:

        # listOrders 位于后半段。
        chunk = CodeChunk(
            chunk_id=("src/api/orders.ts:6-12"),
            relative_path=("src/api/orders.ts"),
            start_line=6,
            end_line=12,
            content=(
                "export function "
                "listOrders() {\n"
                "  if (useMock) {\n"
                "    return "
                "mockListOrders()\n"
                "  }\n"
                "  return []\n"
                "}"
            ),
            language="TypeScript",
            category="source",
        )

    selected = HybridRelevantFile(
        relative_path=(record.relative_path),
        category=(record.category),
        language=(record.language),
        size_bytes=(record.size_bytes),
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
        best_semantic_chunk=(chunk),
        best_semantic_score=(0.9 if with_chunk else None),
        matched_queries=["订单列表状态筛选"],
        target_match_type="none",
    )

    return HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=(record.size_bytes),
        selected_files=[selected],
        skipped_by_budget=[],
        skipped_non_indexable=[],
    )


def test_semantic_chunk_maps_to_real_symbol(
    tmp_path: Path,
):

    create_repository(tmp_path)

    (
        index,
        snapshot,
    ) = build_snapshot(tmp_path)

    selection = build_selection(index=index)

    candidates = RetrievalEvidenceSymbolMapper().discover(
        selection=selection,
        symbol_index=(snapshot.symbol_index),
    )

    ids = {candidate.symbol_id for candidate in candidates}

    assert LIST_ORDERS_ID in ids

    candidate = next(
        candidate for candidate in candidates if candidate.symbol_id == LIST_ORDERS_ID
    )

    assert candidate.source == "evidence"

    assert "订单列表状态筛选" in candidate.matched_queries


def test_file_selection_without_source_range_does_not_invent_symbols(
    tmp_path: Path,
):

    create_repository(tmp_path)

    (
        index,
        snapshot,
    ) = build_snapshot(tmp_path)

    selection = build_selection(
        index=index,
        with_chunk=False,
    )

    candidates = RetrievalEvidenceSymbolMapper().discover(
        selection=selection,
        symbol_index=(snapshot.symbol_index),
    )

    assert candidates == []


def test_evidence_candidate_does_not_automatically_expand_graph(
    tmp_path: Path,
):

    create_repository(tmp_path)

    (
        index,
        snapshot,
    ) = build_snapshot(tmp_path)

    selection = build_selection(index=index)

    evidence = RetrievalEvidenceSymbolMapper().discover(
        selection=selection,
        symbol_index=(snapshot.symbol_index),
    )

    expanded = SymbolGraphExpander().expand(
        candidates=evidence,
        symbol_index=(snapshot.symbol_index),
        symbol_graph=(snapshot.symbol_graph),
    )

    ids = {candidate.symbol_id for candidate in expanded}

    # Discovery 成功。
    assert LIST_ORDERS_ID in ids

    # 但是当前阶段 evidence
    # 不能自动获得 Graph Expansion Permission。
    assert MOCK_LIST_ORDERS_ID not in ids

    assert not any(candidate.source == "graph" for candidate in expanded)
