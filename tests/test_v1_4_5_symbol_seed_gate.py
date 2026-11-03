from __future__ import annotations

from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.codebase.symbol_seed import (
    SymbolSeedEligibilityGate,
)
from agent_core.code_intelligence.model import (
    CodeSymbol,
    SymbolGraphEdge,
)
from agent_core.code_intelligence.symbol_graph import (
    SymbolGraph,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)

LIST_ORDERS_ID = "src/api/orders.ts" "::function::listOrders"

LIST_USERS_ID = "src/api/users.ts" "::function::listUsers"

MOCK_LIST_ORDERS_ID = "src/mocks/service.ts" "::function::mockListOrders"

MOCK_LIST_USERS_ID = "src/mocks/service.ts" "::function::mockListUsers"


def make_candidate(
    *,
    symbol_id: str,
    name: str,
    kind: str,
    path: str,
    start: int,
    end: int,
    score: float,
    source: str,
    reasons: list[str] | None = None,
) -> SymbolCandidate:

    return SymbolCandidate(
        symbol_id=symbol_id,
        name=name,
        kind=kind,
        relative_path=path,
        start_line=start,
        end_line=end,
        score=score,
        source=source,
        matched_queries=[],
        reasons=(reasons or []),
    )


def make_file(
    *,
    path: str,
    rank: int,
    start: int,
    end: int,
) -> HybridRelevantFile:

    return HybridRelevantFile(
        relative_path=path,
        category="source",
        language="TypeScript",
        size_bytes=100,
        retrieval_score=1.0,
        retrieval_rank=rank,
        policy_score=0,
        selection_score=1.0,
        best_semantic_score=0.5,
        best_semantic_chunk=(
            CodeChunk(
                chunk_id=(f"{path}:" f"{start}-{end}"),
                relative_path=path,
                start_line=start,
                end_line=end,
                content="test",
                language="TypeScript",
                category="source",
            )
        ),
    )


def make_selection(
    files: list[HybridRelevantFile],
) -> HybridRelevantFileSelection:

    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(file.size_bytes for file in files),
        selected_files=files,
        skipped_by_budget=[],
        skipped_non_indexable=[],
    )


def test_exact_name_match_is_seed():

    index = SymbolIndex()

    index.add_symbol(
        CodeSymbol(
            symbol_id=LIST_ORDERS_ID,
            name="listOrders",
            kind="function",
            relative_path=("src/api/orders.ts"),
            start_line=5,
            end_line=10,
        )
    )

    candidate = make_candidate(
        symbol_id=LIST_ORDERS_ID,
        name="listOrders",
        kind="function",
        path="src/api/orders.ts",
        start=5,
        end=10,
        score=1.0,
        source="name",
        reasons=["符号名称精确匹配: " "listOrders"],
    )

    result = SymbolSeedEligibilityGate().decide(
        candidates=[candidate],
        selection=(make_selection([])),
        symbol_index=index,
    )

    assert result.seed_symbol_ids == [LIST_ORDERS_ID]

    assert result.decisions[0].confidence == "high"


def test_weak_token_match_is_candidate_but_not_seed():

    index = SymbolIndex()

    index.add_symbol(
        CodeSymbol(
            symbol_id=LIST_USERS_ID,
            name="listUsers",
            kind="function",
            relative_path=("src/api/users.ts"),
            start_line=1,
            end_line=5,
        )
    )

    candidate = make_candidate(
        symbol_id=LIST_USERS_ID,
        name="listUsers",
        kind="function",
        path="src/api/users.ts",
        start=1,
        end=5,
        score=0.3,
        source="name",
        reasons=["符号名称词元匹配: " "list"],
    )

    result = SymbolSeedEligibilityGate().decide(
        candidates=[candidate],
        selection=(make_selection([])),
        symbol_index=index,
    )

    assert result.seed_symbol_ids == []

    assert result.decisions[0].eligible is False


def test_unique_structural_symbol_in_top3_evidence_becomes_seed():

    index = SymbolIndex()

    index.add_symbol(
        CodeSymbol(
            symbol_id=("src/api/orders.ts" "::variable::useMock"),
            name="useMock",
            kind="variable",
            relative_path=("src/api/orders.ts"),
            start_line=3,
            end_line=3,
        )
    )

    index.add_symbol(
        CodeSymbol(
            symbol_id=LIST_ORDERS_ID,
            name="listOrders",
            kind="function",
            relative_path=("src/api/orders.ts"),
            start_line=5,
            end_line=10,
        )
    )

    evidence = make_candidate(
        symbol_id=LIST_ORDERS_ID,
        name="listOrders",
        kind="function",
        path="src/api/orders.ts",
        start=5,
        end=10,
        score=0.47,
        source="evidence",
    )

    result = SymbolSeedEligibilityGate().decide(
        candidates=[evidence],
        selection=(
            make_selection(
                [
                    make_file(
                        path=("src/api/" "orders.ts"),
                        rank=3,
                        start=1,
                        end=10,
                    )
                ]
            )
        ),
        symbol_index=index,
    )

    assert result.seed_symbol_ids == [LIST_ORDERS_ID]


def test_ambiguity_uses_full_symbol_index():

    index = SymbolIndex()

    definitions = [
        (
            "delay",
            2,
            4,
        ),
        (
            "mockListUsers",
            6,
            12,
        ),
        (
            "mockGetUser",
            14,
            19,
        ),
        (
            "mockListOrders",
            21,
            25,
        ),
    ]

    for (
        name,
        start,
        end,
    ) in definitions:

        index.add_symbol(
            CodeSymbol(
                symbol_id=("src/mocks/" "service.ts" f"::function::{name}"),
                name=name,
                kind="function",
                relative_path=("src/mocks/" "service.ts"),
                start_line=start,
                end_line=end,
            )
        )

    # Candidate List 故意只保留
    # mockListOrders。
    #
    # Gate 仍然必须从完整 SymbolIndex
    # 发现同一 Chunk 有多个函数。
    evidence = make_candidate(
        symbol_id=(MOCK_LIST_ORDERS_ID),
        name="mockListOrders",
        kind="function",
        path=("src/mocks/service.ts"),
        start=21,
        end=25,
        score=0.39,
        source="evidence",
    )

    result = SymbolSeedEligibilityGate().decide(
        candidates=[evidence],
        selection=(
            make_selection(
                [
                    make_file(
                        path=("src/mocks/" "service.ts"),
                        rank=2,
                        start=1,
                        end=25,
                    )
                ]
            )
        ),
        symbol_index=index,
    )

    assert result.seed_symbol_ids == []

    assert any("多个结构型 Symbol" in reason for reason in result.decisions[0].reasons)


def test_rank4_evidence_is_not_seed():

    index = SymbolIndex()

    symbol_id = "src/components/" "UserFilters.vue" "::component::UserFilters"

    index.add_symbol(
        CodeSymbol(
            symbol_id=symbol_id,
            name="UserFilters",
            kind="component",
            relative_path=("src/components/" "UserFilters.vue"),
            start_line=1,
            end_line=14,
        )
    )

    evidence = make_candidate(
        symbol_id=symbol_id,
        name="UserFilters",
        kind="component",
        path=("src/components/" "UserFilters.vue"),
        start=1,
        end=14,
        score=0.4,
        source="evidence",
    )

    result = SymbolSeedEligibilityGate().decide(
        candidates=[evidence],
        selection=(
            make_selection(
                [
                    make_file(
                        path=("src/components/" "UserFilters.vue"),
                        rank=4,
                        start=1,
                        end=14,
                    )
                ]
            )
        ),
        symbol_index=index,
    )

    assert result.seed_symbol_ids == []


def test_non_seed_candidate_does_not_amplify_graph_noise():

    index = SymbolIndex()

    symbols = [
        CodeSymbol(
            symbol_id=LIST_ORDERS_ID,
            name="listOrders",
            kind="function",
            relative_path=("src/api/orders.ts"),
            start_line=5,
            end_line=10,
        ),
        CodeSymbol(
            symbol_id=(MOCK_LIST_ORDERS_ID),
            name="mockListOrders",
            kind="function",
            relative_path=("src/mocks/service.ts"),
            start_line=20,
            end_line=25,
        ),
        CodeSymbol(
            symbol_id=LIST_USERS_ID,
            name="listUsers",
            kind="function",
            relative_path=("src/api/users.ts"),
            start_line=1,
            end_line=5,
        ),
        CodeSymbol(
            symbol_id=(MOCK_LIST_USERS_ID),
            name="mockListUsers",
            kind="function",
            relative_path=("src/mocks/service.ts"),
            start_line=8,
            end_line=15,
        ),
    ]

    for symbol in symbols:
        index.add_symbol(symbol)

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_ORDERS_ID),
            target_symbol_id=(MOCK_LIST_ORDERS_ID),
            relation="calls",
            source_path=("src/api/orders.ts"),
            line_number=7,
        )
    )

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_USERS_ID),
            target_symbol_id=(MOCK_LIST_USERS_ID),
            relation="calls",
            source_path=("src/api/users.ts"),
            line_number=3,
        )
    )

    orders_candidate = make_candidate(
        symbol_id=(LIST_ORDERS_ID),
        name="listOrders",
        kind="function",
        path=("src/api/orders.ts"),
        start=5,
        end=10,
        score=0.47,
        source="evidence",
    )

    users_candidate = make_candidate(
        symbol_id=(LIST_USERS_ID),
        name="listUsers",
        kind="function",
        path=("src/api/users.ts"),
        start=1,
        end=5,
        score=0.3,
        source="name",
        reasons=["符号名称词元匹配: " "list"],
    )

    expanded = SymbolGraphExpander().expand(
        candidates=[
            orders_candidate,
            users_candidate,
        ],
        symbol_index=index,
        symbol_graph=graph,
        seed_symbol_ids={LIST_ORDERS_ID},
    )

    all_ids = {item.symbol_id for item in expanded}

    graph_ids = {item.symbol_id for item in expanded if item.source == "graph"}

    # Weak Candidate 仍然可以保留。
    assert LIST_USERS_ID in all_ids

    # 正确 Seed 能扩出订单依赖。
    assert MOCK_LIST_ORDERS_ID in graph_ids

    # 但是 listUsers 没有 Seed 权限，
    # 所以不能放大成用户调用链。
    assert MOCK_LIST_USERS_ID not in graph_ids
