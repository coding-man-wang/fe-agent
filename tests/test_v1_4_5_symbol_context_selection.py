"""v1.4.5 集成契约：补充阅读证据，保持文件基线与权限边界。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_core.codebase.chunker import CodeChunker
from agent_core.codebase.exact_search import ExactCodeSearcher
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph import RepositoryGraphContextExpander
from agent_core.codebase.graph.model import RepositoryDependencyGraph
from agent_core.codebase.hybrid_relevance import HybridRelevantFileSelector
from agent_core.codebase.hybrid_retrieval import HybridRetriever
from agent_core.codebase.model import HybridRelevantFile, HybridRelevantFileSelection
from agent_core.codebase.retrieval_query import RetrievalQuery
from agent_core.codebase.task_context_selection import TaskContextSelectionService
from agent_core.codebase.task_retrieval import TaskRetrievalService
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.code_intelligence.symbol_graph import SymbolGraph
from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.sdd.model import RequirementSpec, TaskDefinition


def task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-SYMBOL",
        task_type="code_change",
        title="查询订单",
        objective="理解 fetchOrders",
        source_step_order=1,
    )


def query(text: str = "fetchOrders") -> RetrievalQuery:
    return RetrievalQuery(
        text=text,
        sources=["raw_requirement"],
        kinds=["literal"],
        intents=["dependency_context"],
    )


def select(service, root, revision="revision-1", current_task=None):
    return asyncio.run(
        service.select(
            repository_root=root,
            repository_revision=revision,
            task=current_task or task(),
            raw_requirement="fetchOrders",
            requirement_spec=RequirementSpec(title="查询订单", summary="fetchOrders"),
        )
    )


def baseline_file() -> HybridRelevantFile:
    return HybridRelevantFile(
        relative_path="src/api.ts",
        category="source",
        language="TypeScript",
        size_bytes=10,
        retrieval_score=0.4,
        retrieval_rank=1,
        policy_score=25,
        selection_score=0.3,
        context_role="implementation_target",
        target_match_type="exact_file",
        target_match_value="src/api.ts",
        reasons=["原有文件证据"],
    )


def mock_pipeline(root: Path, **options):
    index = RepositoryFileIndexer().build(root)
    selected = baseline_file()
    baseline = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=selected.size_bytes,
        selected_files=[selected],
    )
    indexer = Mock()
    indexer.build.return_value = index
    retrieval = Mock()
    retrieval.max_total_bytes = 200_000
    retrieval.build_query_specs.return_value = [query()]
    retrieval.retrieve = AsyncMock(return_value=baseline)
    graph = RepositoryDependencyGraph(
        root_path=str(root), repository_revision="revision-1"
    )
    graph_builder = Mock()
    graph_builder.build.return_value = graph
    graph_expander = Mock()
    graph_expander.max_files = 8
    graph_expander.expand.return_value = baseline
    builder = options.pop("symbol_snapshot_builder", Mock())
    if isinstance(builder, Mock) and builder.build.side_effect is None:
        builder.build.return_value = SimpleNamespace(
            repository_revision="revision-1",
            symbol_index=SymbolIndex(),
            symbol_graph=SymbolGraph(),
            issues=[],
        )
    service = TaskContextSelectionService(
        file_indexer=indexer,
        retrieval_service=retrieval,
        graph_builder=graph_builder,
        graph_expander=graph_expander,
        symbol_snapshot_builder=builder,
        **options,
    )
    return service, builder, baseline


def test_disabled_by_default_preserves_original_pipeline(tmp_path):
    service, builder, baseline = mock_pipeline(tmp_path)
    result = select(service, tmp_path)
    builder.build.assert_not_called()
    assert result.retrieval_selection is baseline
    assert result.selection is baseline
    assert result.symbol_candidates == []
    assert result.symbol_issues == []
    assert result.symbol_added_files == []
    assert result.symbol_error is None
    service.retrieval_service.retrieve.assert_awaited_once_with(
        task=task(),
        index=result.index,
        query_specs=[query()],
        semantic_enabled=False,
    )


@pytest.mark.parametrize(
    "stage,method",
    [
        ("symbol_snapshot_builder", "build"),
        ("symbol_ranker", "rank"),
        ("symbol_expander", "expand"),
        ("symbol_fusion", "fuse"),
    ],
)
def test_optional_stage_failure_preserves_file_selection(tmp_path, stage, method):
    component = Mock()
    getattr(component, method).side_effect = RuntimeError("symbol stage unavailable")
    service, _, baseline = mock_pipeline(
        tmp_path,
        enable_symbol_retrieval=True,
        **{stage: component},
    )
    before = baseline.model_dump()
    current_task = task()
    before_task = current_task.model_dump()
    result = select(service, tmp_path, current_task=current_task)
    assert result.symbol_error == "RuntimeError: symbol stage unavailable"
    assert result.selection is baseline
    assert result.selection.model_dump() == before
    assert result.symbol_added_files == []
    assert result.retrieval_selection is baseline
    assert current_task.model_dump() == before_task


def test_stale_snapshot_is_rejected_before_ranking(tmp_path):
    ranker = Mock()
    service, builder, baseline = mock_pipeline(
        tmp_path,
        enable_symbol_retrieval=True,
        symbol_ranker=ranker,
    )
    builder.build.return_value.repository_revision = "older-revision"
    result = select(service, tmp_path)
    assert result.symbol_error is not None
    assert "版本" in result.symbol_error
    ranker.rank.assert_not_called()
    assert result.selection is baseline


class FixedQueryBuilder:
    """固定查询，避免查询生成策略掩盖符号链路的行为。"""

    def build(self, **kwargs):
        return [query()]


def real_service(*, enabled=True, max_files=8, builder=None, semantic_index=None):
    semantic_searcher = semantic_index or Mock()
    if semantic_index is None:
        semantic_searcher.search = AsyncMock(
            side_effect=AssertionError("不应调用语义搜索")
        )
    retrieval = TaskRetrievalService(
        exact_searcher=ExactCodeSearcher(),
        semantic_searcher=semantic_searcher,
        hybrid_retriever=HybridRetriever(),
        selector=HybridRelevantFileSelector(),
        max_files=1,
        max_total_bytes=200_000,
        query_builder=FixedQueryBuilder(),
    )
    return TaskContextSelectionService(
        file_indexer=RepositoryFileIndexer(),
        retrieval_service=retrieval,
        graph_expander=RepositoryGraphContextExpander(
            max_files=max_files,
            max_graph_files=0,
            min_retrieval_files=1,
        ),
        code_chunker=CodeChunker() if semantic_index is not None else None,
        semantic_index=semantic_index,
        enable_symbol_retrieval=enabled,
        symbol_snapshot_builder=builder,
    )


def create_repository(root: Path):
    source = root / "src"
    source.mkdir()
    (source / "api.ts").write_text(
        "function normalize(value: number) { return value; }\n"
        "export function fetchOrders() { return normalize(1); }\n",
        encoding="utf-8",
    )
    (source / "Orders.vue").write_text(
        '<script setup lang="ts">\n'
        "import { fetchOrders } from './api';\n"
        "function reload() { return fetchOrders(); }\n"
        "</script>\n<template><button>Load</button></template>\n",
        encoding="utf-8",
    )


def paths(selection):
    return [item.relative_path for item in selection.selected_files]


def test_real_typescript_vue_pipeline_adds_read_only_symbol_evidence(tmp_path):
    create_repository(tmp_path)
    current_task = task()
    task_before = current_task.model_dump()
    original = select(real_service(enabled=False), tmp_path, current_task=current_task)
    result = select(real_service(), tmp_path, current_task=current_task)
    assert result.symbol_error is None
    assert (
        result.retrieval_selection.model_dump()
        == original.retrieval_selection.model_dump()
    )
    assert paths(result.selection)[: len(paths(original.selection))] == paths(
        original.selection
    )
    assert set(paths(result.selection)) == {"src/api.ts", "src/Orders.vue"}
    by_name = {item.name: item for item in result.symbol_candidates}
    assert by_name["fetchOrders"].source == "name"
    assert by_name["normalize"].source == "graph"
    assert by_name["reload"].source == "graph"
    assert by_name["normalize"].relation == "calls"
    assert by_name["reload"].relation == "calls"
    assert by_name["normalize"].depth == by_name["reload"].depth == 1
    assert all(item.symbol_candidates for item in result.selection.selected_files)
    additions = [
        item
        for item in result.selection.selected_files
        if item.relative_path not in paths(original.selection)
    ]
    assert additions
    assert result.symbol_added_files == [item.relative_path for item in additions]
    assert all(item.context_role == "reference" for item in additions)
    assert all(
        item.target_match_type == "none" and item.policy_score == 0
        for item in additions
    )
    assert current_task.model_dump() == task_before


def test_full_file_budget_retains_every_original_file(tmp_path):
    create_repository(tmp_path)
    original = select(real_service(enabled=False, max_files=1), tmp_path)
    result = select(real_service(max_files=1), tmp_path)
    assert result.symbol_error is None
    assert paths(result.selection) == paths(original.selection)
    assert result.symbol_added_files == []
    assert (
        result.selection.total_selected_bytes == original.selection.total_selected_bytes
    )
    assert result.symbol_candidates
    assert result.selection.selected_files[0].symbol_candidates
    missing = {"src/api.ts", "src/Orders.vue"} - set(paths(original.selection))
    assert missing <= set(result.selection.skipped_by_budget)


def test_each_selection_rebuilds_current_symbols_even_at_same_revision(tmp_path):
    source = tmp_path / "api.ts"
    source.write_text("export function fetchOrders() {}\n", encoding="utf-8")
    builder = Mock(wraps=RepositorySymbolSnapshotBuilder())
    service = real_service(builder=builder)
    first = select(service, tmp_path)
    assert any(item.name == "fetchOrders" for item in first.symbol_candidates)
    source.write_text("export function completelyDifferent() {}\n", encoding="utf-8")
    second = select(service, tmp_path)
    assert second.symbol_error is None
    assert second.symbol_candidates == []
    source.unlink()
    (tmp_path / "replacement.ts").write_text(
        "export function fetchOrders() {}\n",
        encoding="utf-8",
    )
    third = select(service, tmp_path, revision="revision-2")
    assert third.symbol_error is None
    assert {item.relative_path for item in third.symbol_candidates} == {
        "replacement.ts"
    }
    assert builder.build.call_count == 3
    assert [
        call.kwargs["repository_revision"] for call in builder.build.call_args_list
    ] == [
        "revision-1",
        "revision-1",
        "revision-2",
    ]


def test_semantic_failure_still_allows_current_symbol_retrieval(tmp_path):
    create_repository(tmp_path)
    semantic = Mock()
    semantic.build = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
    semantic.search = AsyncMock(side_effect=AssertionError("不可使用旧语义索引"))
    result = select(real_service(semantic_index=semantic), tmp_path)
    assert result.retrieval_mode == "lexical_fallback"
    assert result.semantic_error == "RuntimeError: embedding unavailable"
    assert result.symbol_error is None
    assert any(item.name == "fetchOrders" for item in result.symbol_candidates)
    assert result.selection.selected_files
    semantic.search.assert_not_awaited()
