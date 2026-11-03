from agent_core.benchmark.symbol_retrieval_evaluator import (
    SymbolGraphProvenanceCollector,
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


def add_symbol(
    index: SymbolIndex,
    symbol_id: str,
):

    index.add_symbol(
        CodeSymbol(
            symbol_id=symbol_id,
            name=symbol_id,
            kind="function",
            relative_path="src/test.ts",
            start_line=1,
            end_line=1,
        )
    )


def test_graph_provenance_keeps_support_for_existing_candidate():

    index = SymbolIndex()

    for symbol_id in [
        "listOrders",
        "mockListOrders",
        "OrdersView",
    ]:
        add_symbol(
            index,
            symbol_id,
        )

    graph = SymbolGraph()

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=("listOrders"),
            target_symbol_id=("mockListOrders"),
            relation="calls",
            source_path=("src/test.ts"),
            line_number=10,
        )
    )

    graph.add_edge(
        SymbolGraphEdge(
            source_symbol_id=("OrdersView"),
            target_symbol_id=("listOrders"),
            relation="calls",
            source_path=("src/test.ts"),
            line_number=20,
        )
    )

    provenance = SymbolGraphProvenanceCollector().collect(
        seed_symbol_ids={"listOrders"},
        symbol_index=index,
        symbol_graph=graph,
    )

    assert {
        (
            item.related_symbol_id,
            item.direction,
            item.relation,
        )
        for item in provenance
    } == {
        (
            "mockListOrders",
            "outgoing",
            "calls",
        ),
        (
            "OrdersView",
            "incoming",
            "calls",
        ),
    }
