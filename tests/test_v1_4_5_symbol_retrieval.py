"""v1.4.5 符号检索完整回归：排序、图扩展、召回下限、阅读权限边界。

所有输入都在本文件构造，不依赖远端服务或预置测试仓库。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.code_intelligence.model import CodeSymbol, SymbolGraphEdge
from agent_core.code_intelligence.symbol_graph import SymbolGraph
from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.codebase.model import (
    FileRecord,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
    SymbolRanker,
    SymbolResultFusion,
)
from agent_core.codebase.symbol_retrieval_model import SymbolCandidate


def symbol(name: str, path: str, *, line: int = 1) -> CodeSymbol:
    return CodeSymbol(
        symbol_id=f"{path}:{line}:{name}",
        name=name,
        kind="function",
        relative_path=path,
        start_line=line,
        end_line=line + 2,
    )


def symbol_index(*symbols: CodeSymbol) -> SymbolIndex:
    result = SymbolIndex()
    for item in symbols:
        result.add_symbol(item)
    return result


def candidate(item: CodeSymbol, *, score: float = 1.0) -> SymbolCandidate:
    return SymbolCandidate(
        **item.model_dump(
            include={
                "symbol_id",
                "name",
                "kind",
                "relative_path",
                "start_line",
                "end_line",
            }
        ),
        score=score,
        source="name",
        matched_queries=[item.name],
        reasons=["测试中的名称命中"],
    )


def edge(
    source: CodeSymbol, target: CodeSymbol, *, relation: str = "calls", line: int = 2
) -> SymbolGraphEdge:
    return SymbolGraphEdge(
        source_symbol_id=source.symbol_id,
        target_symbol_id=target.symbol_id,
        relation=relation,
        source_path=source.relative_path,
        line_number=line,
    )


def file_index(root: Path, files: dict[str, str]) -> RepositoryFileIndex:
    records = []
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        records.append(
            FileRecord(
                relative_path=path,
                name=target.name,
                suffix=target.suffix,
                size_bytes=len(content.encode("utf-8")),
                category="source",
                language="TypeScript",
                is_indexable=True,
            )
        )
    return RepositoryFileIndex(
        root_path=str(root),
        total_files=len(records),
        indexable_files=len(records),
        files=records,
    )


def selection(
    index: RepositoryFileIndex, paths: list[str]
) -> HybridRelevantFileSelection:
    records = {item.relative_path: item for item in index.files}
    selected = [
        HybridRelevantFile(
            relative_path=path,
            category=records[path].category,
            language=records[path].language,
            size_bytes=records[path].size_bytes,
            retrieval_score=1 / rank,
            retrieval_rank=rank,
            policy_score=25 if rank == 1 else 0,
            policy_rank=1 if rank == 1 else None,
            selection_score=1 / rank,
            context_role="implementation_target" if rank == 1 else "reference",
            target_match_type="exact_file" if rank == 1 else "none",
            target_match_value=path if rank == 1 else None,
            matched_queries=["原文件查询"],
            reasons=["原文件选择证据"],
        )
        for rank, path in enumerate(paths, start=1)
    ]
    return HybridRelevantFileSelection(
        candidate_count=len(paths),
        eligible_count=len(paths),
        selected_count=len(selected),
        total_selected_bytes=sum(item.size_bytes for item in selected),
        selected_files=selected,
    )


def test_exact_identifier_outranks_camel_case_token_match():
    exact = symbol("listOrders", "src/api.ts")
    token_match = symbol("listArchivedOrders", "src/archive.ts")
    unrelated = symbol("renderCart", "src/listOrders.ts")
    result = SymbolRanker().rank(
        queries=["修复 listOrders()"],
        symbol_index=symbol_index(token_match, unrelated, exact),
    )
    assert result[0].symbol_id == exact.symbol_id
    assert result[0].score > result[1].score > 0
    assert {item.symbol_id for item in result} == {
        exact.symbol_id,
        token_match.symbol_id,
    }
    assert result[0].matched_queries == ["修复 listOrders()"]
    assert result[0].reasons


def test_tokenization_handles_acronyms_and_snake_case():
    items = [symbol("HTTPClient", "http.ts"), symbol("http_client", "snake.ts")]
    result = SymbolRanker().rank(
        queries=["http client"], symbol_index=symbol_index(*items)
    )
    assert {item.symbol_id for item in result} == {item.symbol_id for item in items}
    assert all(item.score > 0 for item in result)


def test_names_are_not_merged_across_files_and_ranking_is_deterministic():
    a = symbol("load", "a.ts")
    b = symbol("load", "b.ts")
    ranker = SymbolRanker()
    first = ranker.rank(queries=["load"], symbol_index=symbol_index(b, a))
    second = ranker.rank(queries=["load", "load"], symbol_index=symbol_index(a, b))
    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]
    assert [item.relative_path for item in first] == ["a.ts", "b.ts"]
    assert (
        ranker.rank(queries=["load"], symbol_index=symbol_index(a, b), limit=1)
        == first[:1]
    )


@pytest.mark.parametrize("queries", [[], [""], ["   "]])
def test_empty_query_has_no_candidates(queries):
    assert (
        SymbolRanker().rank(
            queries=queries, symbol_index=symbol_index(symbol("load", "a.ts"))
        )
        == []
    )


def test_filename_alone_does_not_create_a_symbol_candidate():
    assert (
        SymbolRanker().rank(
            queries=["orders"],
            symbol_index=symbol_index(symbol("renderCart", "src/orders.ts")),
        )
        == []
    )


def test_graph_expansion_is_one_hop_bidirectional_and_explainable():
    seed = symbol("loadOrders", "seed.ts")
    called = symbol("request", "called.ts")
    caller = symbol("refresh", "caller.ts")
    typed = symbol("Order", "type.ts")
    distant = symbol("decode", "distant.ts")
    missing = symbol("deleted", "deleted.ts")
    index = symbol_index(seed, called, caller, typed, distant)
    graph = SymbolGraph()
    for item in [
        edge(seed, called),
        edge(caller, seed),
        edge(seed, typed, relation="references"),
        edge(called, distant),
        edge(called, seed),
        edge(seed, seed),
        edge(seed, missing),
        edge(missing, seed),
    ]:
        graph.add_edge(item)
    result = SymbolGraphExpander().expand(
        candidates=[candidate(seed)], symbol_index=index, symbol_graph=graph
    )
    by_id = {item.symbol_id: item for item in result}
    assert set(by_id) == {item.symbol_id for item in [seed, called, caller, typed]}
    assert len(result) == len(by_id)
    assert by_id[seed.symbol_id].source == "name"
    assert by_id[caller.symbol_id].direction == "incoming"
    assert by_id[typed.symbol_id].relation == "references"
    for item in result:
        if item.symbol_id != seed.symbol_id:
            assert item.source == "graph"
            assert item.depth == 1
            assert 0 < item.score < by_id[seed.symbol_id].score
            assert item.seed_symbol_id == seed.symbol_id
            assert item.matched_queries == [seed.name]
            assert item.reasons


def test_graph_budget_caps_neighbors_without_removing_direct_matches():
    seeds = [symbol("first", "first.ts"), symbol("second", "second.ts")]
    neighbors = [symbol(f"neighbor{i}", f"n{i}.ts") for i in range(4)]
    index = symbol_index(*seeds, *neighbors)
    graph = SymbolGraph()
    edges = [edge(seed, other) for seed in seeds for other in neighbors]
    for item in reversed(edges):
        graph.add_edge(item)
    result = SymbolGraphExpander().expand(
        candidates=[candidate(item) for item in reversed(seeds)],
        symbol_index=index,
        symbol_graph=graph,
        max_neighbors=1,
    )
    assert len(result) == 3
    assert {item.symbol_id for item in result if item.source == "name"} == {
        item.symbol_id for item in seeds
    }
    again = SymbolGraphExpander().expand(
        candidates=[candidate(item) for item in seeds],
        symbol_index=index,
        symbol_graph=graph,
        max_neighbors=1,
    )
    assert result == again
    assert (
        len(
            SymbolGraphExpander().expand(
                candidates=[candidate(item) for item in seeds],
                symbol_index=index,
                symbol_graph=graph,
                max_neighbors=0,
            )
        )
        == 2
    )


def test_graph_only_result_cannot_be_used_to_recursively_expand():
    seed, first, second = [
        symbol(name, f"{name}.ts") for name in ["seed", "first", "second"]
    ]
    index = symbol_index(seed, first, second)
    graph = SymbolGraph()
    graph.add_edge(edge(seed, first))
    graph.add_edge(edge(first, second))
    expander = SymbolGraphExpander()
    one = expander.expand(
        candidates=[candidate(seed)], symbol_index=index, symbol_graph=graph
    )
    two = expander.expand(candidates=one, symbol_index=index, symbol_graph=graph)
    assert {item.symbol_id for item in two} == {seed.symbol_id, first.symbol_id}


def test_full_file_budget_preserves_baseline_and_attaches_symbol_evidence(tmp_path):
    index = file_index(tmp_path, {"z.ts": "zz", "a.ts": "aa", "new.ts": "nn"})
    baseline = selection(index, ["z.ts", "a.ts"])
    baseline_before = baseline.model_dump()
    ranked = [
        candidate(symbol("new", "new.ts")),
        candidate(symbol("load", "a.ts"), score=0.4),
    ]
    ranked_before = [item.model_dump() for item in ranked]
    result = SymbolResultFusion().fuse(
        selection=baseline,
        index=index,
        candidates=ranked,
        max_files=2,
        max_total_bytes=100,
    )
    assert [item.relative_path for item in result.selected_files] == ["z.ts", "a.ts"]
    assert result.selected_files[1].symbol_candidates[0].name == "load"
    for before, after in zip(baseline.selected_files, result.selected_files):
        assert before.model_dump(exclude={"symbol_candidates"}) == after.model_dump(
            exclude={"symbol_candidates"}
        )
    assert result.skipped_by_budget == ["new.ts"]
    assert result.selected_count == 2
    assert result.total_selected_bytes == 4
    assert baseline.model_dump() == baseline_before
    assert [item.model_dump() for item in ranked] == ranked_before


def test_spare_budget_adds_one_file_per_path_without_write_target_promotion(tmp_path):
    index = file_index(tmp_path, {"base.ts": "bb", "extra.ts": "eee"})
    baseline = selection(index, ["base.ts"])
    first = candidate(symbol("find", "extra.ts", line=1))
    second = candidate(symbol("load", "extra.ts", line=5), score=0.4)
    fusion = SymbolResultFusion()
    result = fusion.fuse(
        selection=baseline,
        index=index,
        candidates=[first, second, first],
        max_files=2,
        max_total_bytes=5,
    )
    assert [item.relative_path for item in result.selected_files] == [
        "base.ts",
        "extra.ts",
    ]
    added = result.selected_files[-1]
    assert len(added.symbol_candidates) == 2
    assert added.context_role == "reference"
    assert added.target_match_type == "none"
    assert added.target_match_value is None
    assert added.policy_score == 0
    assert added.policy_rank is None
    assert added.retrieval_score == 0
    assert added.query_hit_count == 0
    assert added.lexical_hit_count == 0
    assert added.semantic_hit_count == 0
    assert added.hybrid_hit_count == 0
    assert result.selected_count == 2
    assert result.total_selected_bytes == 5
    assert result.candidate_count == 2
    assert result.eligible_count == 2
    single = fusion.fuse(
        selection=baseline,
        index=index,
        candidates=[first],
        max_files=2,
        max_total_bytes=5,
    )
    assert added.selection_score == single.selected_files[-1].selection_score


def test_byte_budget_does_not_evict_baseline_and_can_skip_large_candidate(tmp_path):
    index = file_index(
        tmp_path, {"base.ts": "bb", "large.ts": "0123456789", "small.ts": "s"}
    )
    baseline = selection(index, ["base.ts"])
    result = SymbolResultFusion().fuse(
        selection=baseline,
        index=index,
        candidates=[
            candidate(symbol("large", "large.ts")),
            candidate(symbol("small", "small.ts"), score=0.5),
        ],
        max_files=3,
        max_total_bytes=3,
    )
    assert [item.relative_path for item in result.selected_files] == [
        "base.ts",
        "small.ts",
    ]
    assert result.skipped_by_budget == ["large.ts"]
    assert result.total_selected_bytes == 3


def test_baseline_is_retained_even_if_it_already_exceeds_supplement_budget(tmp_path):
    index = file_index(tmp_path, {"base.ts": "existing file", "extra.ts": "e"})
    baseline = selection(index, ["base.ts"])
    result = SymbolResultFusion().fuse(
        selection=baseline,
        index=index,
        candidates=[candidate(symbol("extra", "extra.ts"))],
        max_files=0,
        max_total_bytes=0,
    )
    assert result.selected_files == baseline.selected_files
    assert result.total_selected_bytes == baseline.total_selected_bytes
    assert "extra.ts" in result.skipped_by_budget


@pytest.mark.parametrize("unsafe", ["../outside.ts", "C:/outside.ts", "/outside.ts"])
def test_outside_paths_are_rejected_even_when_present_in_file_index(tmp_path, unsafe):
    index = file_index(tmp_path, {"base.ts": "b"})
    index.files.append(
        FileRecord(
            relative_path=unsafe,
            name="outside.ts",
            suffix=".ts",
            size_bytes=1,
            category="source",
            is_indexable=True,
        )
    )
    result = SymbolResultFusion().fuse(
        selection=selection(index, ["base.ts"]),
        index=index,
        candidates=[candidate(symbol("outside", unsafe))],
        max_files=10,
        max_total_bytes=100,
    )
    assert [item.relative_path for item in result.selected_files] == ["base.ts"]
    assert unsafe in result.skipped_non_indexable


def test_non_indexable_missing_and_deleted_files_are_not_added(tmp_path):
    index = file_index(tmp_path, {"base.ts": "b", "denied.ts": "d", "deleted.ts": "x"})
    index.files[1].is_indexable = False
    (tmp_path / "deleted.ts").unlink()
    blocked = ["denied.ts", "deleted.ts", "unknown.ts"]
    result = SymbolResultFusion().fuse(
        selection=selection(index, ["base.ts"]),
        index=index,
        candidates=[candidate(symbol("candidate", path)) for path in blocked],
        max_files=10,
        max_total_bytes=100,
    )
    assert [item.relative_path for item in result.selected_files] == ["base.ts"]
    assert set(result.skipped_non_indexable) == set(blocked)


def test_empty_symbol_result_is_exactly_the_file_baseline(tmp_path):
    index = file_index(tmp_path, {"base.ts": "b"})
    baseline = selection(index, ["base.ts"])
    result = SymbolResultFusion().fuse(
        selection=baseline, index=index, candidates=[], max_files=1, max_total_bytes=1
    )
    assert result == baseline
    assert result is not baseline


def test_new_symbols_are_ordered_by_signal_without_moving_baseline(tmp_path):
    index = file_index(tmp_path, {"base.ts": "b", "weak.ts": "w", "strong.ts": "s"})
    result = SymbolResultFusion().fuse(
        selection=selection(index, ["base.ts"]),
        index=index,
        candidates=[
            candidate(symbol("weak", "weak.ts"), score=0.1),
            candidate(symbol("strong", "strong.ts")),
        ],
        max_files=2,
        max_total_bytes=100,
    )
    assert [item.relative_path for item in result.selected_files] == [
        "base.ts",
        "strong.ts",
    ]
    assert result.skipped_by_budget == ["weak.ts"]


def test_invalid_negative_budgets_are_rejected(tmp_path):
    index = file_index(tmp_path, {})
    with pytest.raises(ValueError):
        SymbolRanker().rank(queries=["load"], symbol_index=SymbolIndex(), limit=-1)
    with pytest.raises(ValueError):
        SymbolGraphExpander().expand(
            candidates=[],
            symbol_index=SymbolIndex(),
            symbol_graph=SymbolGraph(),
            max_neighbors=-1,
        )
    for max_files, max_bytes in [(-1, 100), (1, -1)]:
        with pytest.raises(ValueError):
            SymbolResultFusion().fuse(
                selection=selection(index, []),
                index=index,
                candidates=[],
                max_files=max_files,
                max_total_bytes=max_bytes,
            )
