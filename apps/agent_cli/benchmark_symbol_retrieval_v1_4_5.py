from __future__ import annotations

import json
import os
from pathlib import Path

from agent_core.benchmark.symbol_retrieval_evaluator import (
    SymbolRetrievalBenchmarkCase,
    SymbolRetrievalBenchmarkRunner,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)

ORDERS_VIEW_ID = "src/views/OrdersView.vue" "::component" "::OrdersView"

LIST_ORDERS_ID = "src/api/orders.ts" "::function" "::listOrders"

MOCK_LIST_ORDERS_ID = "src/mocks/service.ts" "::function" "::mockListOrders"

REQUEST_ID = "src/api/requestClient.ts" "::function" "::request"


LIST_USERS_ID = "src/api/users.ts" "::function" "::listUsers"

GET_USER_ID = "src/api/users.ts" "::function" "::getUser"

MOCK_LIST_USERS_ID = "src/mocks/service.ts" "::function" "::mockListUsers"

MOCK_GET_USER_ID = "src/mocks/service.ts" "::function" "::mockGetUser"

USERS_VIEW_ID = "src/views/UsersView.vue" "::component" "::UsersView"

USER_DETAIL_VIEW_ID = "src/views/UserDetailView.vue" "::component" "::UserDetailView"


REQUIRED_ORDER_SYMBOLS = [
    ORDERS_VIEW_ID,
    LIST_ORDERS_ID,
    MOCK_LIST_ORDERS_ID,
    REQUEST_ID,
]


FORBIDDEN_USER_SYMBOLS = [
    LIST_USERS_ID,
    GET_USER_ID,
    MOCK_LIST_USERS_ID,
    MOCK_GET_USER_ID,
    USERS_VIEW_ID,
    USER_DETAIL_VIEW_ID,
]


def resolve_repository_root() -> Path:
    """
    优先使用环境变量。

    如果没有配置，则假设：

        D:/project/
        ├── fe-agent/
        └── fe-agent-benchmark-web/

    两个 Repository 是兄弟目录。
    """

    configured = os.getenv("FE_AGENT_BENCHMARK_WEB_ROOT")

    if configured:

        root = Path(configured).resolve()

    else:

        current_repo = Path(__file__).resolve().parents[2]

        root = (current_repo.parent / "fe-agent-benchmark-web").resolve()

    if not root.is_dir():

        raise RuntimeError(
            "找不到 fe-agent-benchmark-web。\n"
            "可以设置环境变量：\n"
            "FE_AGENT_BENCHMARK_WEB_ROOT"
        )

    return root


def build_case(
    *,
    case_id: str,
    rank_limit: int,
) -> SymbolRetrievalBenchmarkCase:

    return SymbolRetrievalBenchmarkCase(
        case_id=case_id,
        queries=[
            "listOrders",
        ],
        expected_symbol_ids=(REQUIRED_ORDER_SYMBOLS),
        forbidden_symbol_ids=(FORBIDDEN_USER_SYMBOLS),
        rank_limit=rank_limit,
        max_graph_neighbors=20,
    )


def print_evaluation(
    title: str,
    evaluation,
) -> None:

    print()
    print("=" * 80)

    print(title)

    print("=" * 80)

    print(
        json.dumps(
            evaluation.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:

    repository_root = resolve_repository_root()

    print("Benchmark Repository:")

    print(repository_root)

    index = RepositoryFileIndexer().build(repository_root)

    runner = SymbolRetrievalBenchmarkRunner()

    # ==================================
    # Case 1
    #
    # Graph Isolation Benchmark
    #
    # 只允许名称阶段留下最强 Seed：
    #
    # listOrders
    #
    # 专门判断 Graph 自己是否正确。
    # ==================================

    graph_isolation = runner.run(
        case=build_case(
            case_id=("REAL-ORDER-" "GRAPH-ISOLATION"),
            rank_limit=1,
        ),
        file_index=index,
        repository_revision=("real-repository"),
    )

    print_evaluation(
        "Graph Isolation Benchmark",
        graph_isolation,
    )

    # ==================================
    # Case 2
    #
    # Production-shaped Benchmark
    #
    # 使用 SymbolRanker 当前默认规模。
    #
    # 测：
    # Ranker + Graph 的真实联合表现。
    # ==================================

    production = runner.run(
        case=build_case(
            case_id=("REAL-ORDER-" "PRODUCTION-SHAPED"),
            rank_limit=20,
        ),
        file_index=index,
        repository_revision=("real-repository"),
    )

    print_evaluation(
        "Production-shaped Benchmark",
        production,
    )


if __name__ == "__main__":
    main()
