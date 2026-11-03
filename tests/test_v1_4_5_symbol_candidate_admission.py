from agent_core.codebase.symbol_candidate_admission import (
    SymbolCandidateAdmissionGate,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
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

LIST_ORDERS = "src/api/orders.ts" "::function::listOrders"

MOCK_LIST_ORDERS = "src/mocks/service.ts" "::function::mockListOrders"

REQUEST = "src/api/requestClient.ts" "::function::request"

ORDER = "src/types/domain.ts" "::interface::Order"

USE_MOCK = "src/api/orders.ts" "::variable::useMock"

USER = "src/types/domain.ts" "::interface::User"

MOCK_LIST_USERS = "src/mocks/service.ts" "::function::mockListUsers"


def add_symbol(
    index: SymbolIndex,
    *,
    symbol_id: str,
    name: str,
    kind: str = "function",
):

    index.add_symbol(
        CodeSymbol(
            symbol_id=symbol_id,
            name=name,
            kind=kind,
            relative_path=(
                symbol_id.split(
                    "::",
                    1,
                )[0]
            ),
            start_line=1,
            end_line=3,
        )
    )


def candidate(
    symbol_id: str,
    name: str,
    *,
    kind: str = "function",
) -> SymbolCandidate:

    return SymbolCandidate(
        symbol_id=symbol_id,
        name=name,
        kind=kind,
        relative_path=(
            symbol_id.split(
                "::",
                1,
            )[0]
        ),
        start_line=1,
        end_line=3,
        score=0.5,
        source="evidence",
    )


def test_anchor_is_always_admitted():

    index = SymbolIndex()

    add_symbol(
        index,
        symbol_id=LIST_ORDERS,
        name="listOrders",
    )

    result = SymbolCandidateAdmissionGate().admit(
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            )
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=SymbolGraph(),
    )

    assert [item.symbol_id for item in result.admitted_candidates] == [LIST_ORDERS]


def test_graph_corroborated_candidate_is_admitted():

    index = SymbolIndex()

    add_symbol(
        index,
        symbol_id=LIST_ORDERS,
        name="listOrders",
    )

    add_symbol(
        index,
        symbol_id=MOCK_LIST_ORDERS,
        name="mockListOrders",
    )

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_ORDERS),
            target_symbol_id=(MOCK_LIST_ORDERS),
            relation="calls",
            source_path=("src/api/orders.ts"),
            line_number=8,
        )
    )

    result = SymbolCandidateAdmissionGate().admit(
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            ),
            candidate(
                MOCK_LIST_ORDERS,
                "mockListOrders",
            ),
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=graph,
    )

    ids = {item.symbol_id for item in result.admitted_candidates}

    assert MOCK_LIST_ORDERS in ids

    decision = next(
        item for item in result.decisions if item.symbol_id == MOCK_LIST_ORDERS
    )

    assert decision.reason_kind == "graph_corroborated"


def test_unrelated_candidate_is_rejected():

    index = SymbolIndex()

    add_symbol(
        index,
        symbol_id=LIST_ORDERS,
        name="listOrders",
    )

    add_symbol(
        index,
        symbol_id=USER,
        name="User",
        kind="interface",
    )

    result = SymbolCandidateAdmissionGate().admit(
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            ),
            candidate(
                USER,
                "User",
                kind="interface",
            ),
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=SymbolGraph(),
    )

    assert USER in result.rejected_symbol_ids


def test_admission_does_not_create_graph_only_symbol():

    index = SymbolIndex()

    add_symbol(
        index,
        symbol_id=LIST_ORDERS,
        name="listOrders",
    )

    add_symbol(
        index,
        symbol_id=REQUEST,
        name="request",
    )

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_ORDERS),
            target_symbol_id=(REQUEST),
            relation="calls",
            source_path=("src/api/orders.ts"),
            line_number=9,
        )
    )

    result = SymbolCandidateAdmissionGate().admit(
        # request 故意不在
        # Generated Candidate Pool。
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            )
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=graph,
    )

    assert {item.symbol_id for item in result.admitted_candidates} == {LIST_ORDERS}


def test_reference_neighbor_can_be_supporting_candidate():

    index = SymbolIndex()

    add_symbol(
        index,
        symbol_id=LIST_ORDERS,
        name="listOrders",
    )

    add_symbol(
        index,
        symbol_id=ORDER,
        name="Order",
        kind="interface",
    )

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_ORDERS),
            target_symbol_id=ORDER,
            relation="references",
            source_path=("src/api/orders.ts"),
            line_number=7,
        )
    )

    result = SymbolCandidateAdmissionGate().admit(
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            ),
            candidate(
                ORDER,
                "Order",
                kind="interface",
            ),
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=graph,
    )

    assert ORDER in {item.symbol_id for item in result.admitted_candidates}


def test_unrelated_user_branch_is_pruned():

    index = SymbolIndex()

    for (
        symbol_id,
        name,
    ) in [
        (
            LIST_ORDERS,
            "listOrders",
        ),
        (
            MOCK_LIST_ORDERS,
            "mockListOrders",
        ),
        (
            MOCK_LIST_USERS,
            "mockListUsers",
        ),
    ]:
        add_symbol(
            index,
            symbol_id=symbol_id,
            name=name,
        )

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=(LIST_ORDERS),
            target_symbol_id=(MOCK_LIST_ORDERS),
            relation="calls",
            source_path=("src/api/orders.ts"),
            line_number=8,
        )
    )

    result = SymbolCandidateAdmissionGate().admit(
        candidates=[
            candidate(
                LIST_ORDERS,
                "listOrders",
            ),
            candidate(
                MOCK_LIST_ORDERS,
                "mockListOrders",
            ),
            candidate(
                MOCK_LIST_USERS,
                "mockListUsers",
            ),
        ],
        anchor_symbol_ids={LIST_ORDERS},
        symbol_index=index,
        symbol_graph=graph,
    )

    ids = {item.symbol_id for item in result.admitted_candidates}

    assert MOCK_LIST_ORDERS in ids

    assert MOCK_LIST_USERS not in ids
