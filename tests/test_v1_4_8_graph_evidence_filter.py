from agent_core.codebase.graph_evidence_filter import (
    GraphEvidenceFilter,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)


def make_file(
    path,
    *,
    graph=False,
    semantic_hits=0,
    symbols=None,
):
    return HybridRelevantFile(
        relative_path=path,
        category="source",
        language="TypeScript",
        size_bytes=100,
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
        semantic_hit_count=semantic_hits,
        context_role=("dependency" if graph else "unclassified"),
        symbol_candidates=symbols or [],
        reasons=(["Graph dependency: test"] if graph else ["Hybrid Retrieval"]),
    )


def make_selection(files):
    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(file.size_bytes for file in files),
        selected_files=files,
    )


def test_filter_removes_weak_graph_only_files():
    selection = make_selection(
        [
            make_file(
                "src/api/orders.ts",
                semantic_hits=6,
            ),
            make_file(
                "src/components/UserTable.vue",
                graph=True,
            ),
        ]
    )

    proposal = GraphEvidenceFilter().propose(selection=selection)

    assert proposal.dropped_paths == ["src/components/UserTable.vue"]

    assert [item.relative_path for item in proposal.selection.selected_files] == [
        "src/api/orders.ts"
    ]

    # 原始结果不可被修改
    assert selection.selected_count == 2
    assert len(selection.selected_files) == 2


def test_filter_keeps_graph_symbol_evidence():
    symbol = SymbolCandidate(
        symbol_id="src/api/request.ts::function::request",
        name="request",
        kind="function",
        relative_path="src/api/request.ts",
        start_line=1,
        end_line=3,
        score=0.9,
        source="graph",
    )

    selection = make_selection(
        [
            make_file(
                "src/api/request.ts",
                graph=True,
                symbols=[symbol],
            ),
        ]
    )

    proposal = GraphEvidenceFilter().propose(selection=selection)

    assert proposal.dropped_paths == []
    assert proposal.selection.selected_count == 1


def test_filter_keeps_semantic_only_file():
    selection = make_selection(
        [
            make_file(
                "src/components/UserFilters.vue",
                semantic_hits=4,
            ),
        ]
    )

    proposal = GraphEvidenceFilter().propose(selection=selection)

    assert proposal.dropped_paths == []
    assert proposal.selection.selected_count == 1
