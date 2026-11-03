from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)

ORDERS_VIEW_ID = "src/views/OrdersView.vue" "::component" "::OrdersView"

LIST_ORDERS_ID = "src/api/orders.ts" "::function" "::listOrders"

MOCK_LIST_ORDERS_ID = "src/mocks/service.ts" "::function" "::mockListOrders"

REQUEST_ID = "src/api/requestClient.ts" "::function" "::request"


MOCK_LIST_USERS_ID = "src/mocks/service.ts" "::function" "::mockListUsers"


def benchmark_repository_root() -> Path:

    configured = os.getenv("FE_AGENT_BENCHMARK_WEB_ROOT")

    if configured:

        root = Path(configured).resolve()

    else:

        agent_root = Path(__file__).resolve().parents[2]

        root = (agent_root.parent / "fe-agent-benchmark-web").resolve()

    if not root.is_dir():

        pytest.skip(
            "fe-agent-benchmark-web "
            "不在本地；"
            "设置 "
            "FE_AGENT_BENCHMARK_WEB_ROOT "
            "后运行真实仓库 Benchmark"
        )

    return root


@pytest.fixture
def snapshot():

    root = benchmark_repository_root()

    index = RepositoryFileIndexer().build(root)

    return RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("integration-test"),
    )


def test_real_repository_contains_expected_order_symbols(
    snapshot,
):

    for symbol_id in [
        ORDERS_VIEW_ID,
        LIST_ORDERS_ID,
        MOCK_LIST_ORDERS_ID,
        REQUEST_ID,
    ]:

        assert snapshot.symbol_index.get_symbol(symbol_id) is not None, symbol_id


def test_list_orders_has_expected_incoming_order_edge(
    snapshot,
):

    incoming = snapshot.symbol_graph.incoming(LIST_ORDERS_ID)

    source_ids = {edge.source_symbol_id for edge in incoming}

    assert ORDERS_VIEW_ID in source_ids


def test_list_orders_has_expected_outgoing_dependencies(
    snapshot,
):

    outgoing = snapshot.symbol_graph.outgoing(LIST_ORDERS_ID)

    target_ids = {edge.target_symbol_id for edge in outgoing}

    assert MOCK_LIST_ORDERS_ID in target_ids

    assert REQUEST_ID in target_ids


def test_list_orders_does_not_directly_connect_to_user_mock(
    snapshot,
):

    outgoing = snapshot.symbol_graph.outgoing(LIST_ORDERS_ID)

    incoming = snapshot.symbol_graph.incoming(LIST_ORDERS_ID)

    connected_ids = {
        *[edge.target_symbol_id for edge in outgoing],
        *[edge.source_symbol_id for edge in incoming],
    }

    assert MOCK_LIST_USERS_ID not in connected_ids


def test_service_file_can_contain_both_business_symbols_without_graph_merging_them(
    snapshot,
):

    order_mock = snapshot.symbol_index.get_symbol(MOCK_LIST_ORDERS_ID)

    user_mock = snapshot.symbol_index.get_symbol(MOCK_LIST_USERS_ID)

    assert order_mock is not None

    assert user_mock is not None

    assert order_mock.relative_path == user_mock.relative_path == "src/mocks/service.ts"

    # 重点：
    #
    # 两个 Symbol 虽然在一个 File，
    # 但 SymbolGraph 不应因此自动建立关系。

    order_outgoing = {
        edge.target_symbol_id
        for edge in snapshot.symbol_graph.outgoing(MOCK_LIST_ORDERS_ID)
    }

    order_incoming = {
        edge.source_symbol_id
        for edge in snapshot.symbol_graph.incoming(MOCK_LIST_ORDERS_ID)
    }

    assert MOCK_LIST_USERS_ID not in order_outgoing

    assert MOCK_LIST_USERS_ID not in order_incoming
