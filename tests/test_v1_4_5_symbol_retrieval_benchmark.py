from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.benchmark.symbol_retrieval_evaluator import (
    SymbolRetrievalBenchmarkCase,
    SymbolRetrievalBenchmarkEvaluator,
    SymbolRetrievalBenchmarkRunner,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
    SymbolRanker,
    SymbolResultFusion,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)

ORDER_LOAD_ID = "src/views/OrdersView.vue" "::function" "::script_setup.loadOrders"

ORDER_LIST_ID = "src/api/orders.ts" "::function" "::listOrders"

MOCK_ORDER_LIST_ID = "src/mocks/service.ts" "::function" "::mockListOrders"


USER_LOAD_ID = "src/views/UsersView.vue" "::function" "::script_setup.loadUsers"

USER_LIST_ID = "src/api/users.ts" "::function" "::listUsers"

MOCK_USER_LIST_ID = "src/mocks/service.ts" "::function" "::mockListUsers"


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
        "src/views/OrdersView.vue",
        """
<template>
  <main>Orders</main>
</template>

<script setup lang="ts">
import { listOrders } from "../api/orders"

async function loadOrders() {
  return listOrders()
}
</script>
""",
    )

    write_file(
        root,
        "src/api/orders.ts",
        """
import {
  mockListOrders
} from "../mocks/service"

export function listOrders() {
  return mockListOrders()
}
""",
    )

    write_file(
        root,
        "src/views/UsersView.vue",
        """
<template>
  <main>Users</main>
</template>

<script setup lang="ts">
import { listUsers } from "../api/users"

async function loadUsers() {
  return listUsers()
}
</script>
""",
    )

    write_file(
        root,
        "src/api/users.ts",
        """
import {
  mockListUsers
} from "../mocks/service"

export function listUsers() {
  return mockListUsers()
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


def order_case() -> SymbolRetrievalBenchmarkCase:

    return SymbolRetrievalBenchmarkCase(
        case_id=("SYMBOL-RETRIEVAL-" "ORDER-CHAIN"),
        queries=[
            "listOrders",
        ],
        expected_symbol_ids=[
            ORDER_LOAD_ID,
            ORDER_LIST_ID,
            MOCK_ORDER_LIST_ID,
        ],
        forbidden_symbol_ids=[
            USER_LOAD_ID,
            USER_LIST_ID,
            MOCK_USER_LIST_ID,
        ],
        # 刻意只让名称阶段保留一个 Seed，
        # 用于测量 Graph Expansion Gain。
        #
        # 这不是修改 Ranker 算法，
        # 只是 Benchmark Retrieval Budget。
        rank_limit=1,
        max_graph_neighbors=10,
    )


# ==================================
# 1. Main Benchmark
#
# Direct:
#   listOrders
#
# Graph:
#   loadOrders
#   mockListOrders
#
# User Chain:
#   不应进入
# ==================================


def test_order_chain_symbol_retrieval_benchmark(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    evaluation = SymbolRetrievalBenchmarkRunner().run(
        case=order_case(),
        file_index=index,
        repository_revision=("benchmark-revision-1"),
    )

    assert evaluation.status == "passed"

    assert evaluation.direct_symbol_ids == [
        ORDER_LIST_ID,
    ]

    assert set(evaluation.expanded_symbol_ids) == {
        ORDER_LOAD_ID,
        ORDER_LIST_ID,
        MOCK_ORDER_LIST_ID,
    }

    assert evaluation.matched_expected_symbol_ids == sorted(
        [
            ORDER_LOAD_ID,
            ORDER_LIST_ID,
            MOCK_ORDER_LIST_ID,
        ]
    )

    assert evaluation.missed_expected_symbol_ids == []

    assert evaluation.forbidden_hits == []

    assert evaluation.unexpected_symbol_ids == []

    assert evaluation.direct_recall == pytest.approx(1 / 3)

    assert evaluation.symbol_recall == pytest.approx(1.0)

    assert evaluation.symbol_precision == pytest.approx(1.0)

    assert evaluation.graph_expansion_gain == pytest.approx(2 / 3)

    assert evaluation.noise_expansion_rate == pytest.approx(0.0)


# ==================================
# 2. Graph 真正补回了必要 Symbol
# ==================================


def test_graph_expansion_recovers_missing_required_symbols(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    evaluation = SymbolRetrievalBenchmarkRunner().run(
        case=order_case(),
        file_index=index,
        repository_revision=("benchmark-revision-2"),
    )

    assert set(evaluation.graph_recovered_expected_symbol_ids) == {
        ORDER_LOAD_ID,
        MOCK_ORDER_LIST_ID,
    }

    assert set(evaluation.graph_added_symbol_ids) == {
        ORDER_LOAD_ID,
        MOCK_ORDER_LIST_ID,
    }


# ==================================
# 3. 同文件 User Symbol 不应因为
#    service.ts 被整体关联而进入订单链
# ==================================


def test_symbol_graph_does_not_pull_unrelated_symbol_from_same_file(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    evaluation = SymbolRetrievalBenchmarkRunner().run(
        case=order_case(),
        file_index=index,
        repository_revision=("benchmark-revision-3"),
    )

    assert MOCK_ORDER_LIST_ID in evaluation.expanded_symbol_ids

    assert MOCK_USER_LIST_ID not in evaluation.expanded_symbol_ids

    assert USER_LIST_ID not in evaluation.expanded_symbol_ids

    assert USER_LOAD_ID not in evaluation.expanded_symbol_ids


# ==================================
# 4. Snapshot Build Cost
#
# 第一版只测量，
# 不设置毫秒阈值。
#
# CI / Windows / macOS 性能不同，
# 现在设绝对时间 Gate 会不稳定。
# ==================================


def test_symbol_snapshot_build_cost_is_recorded(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    evaluation = SymbolRetrievalBenchmarkRunner().run(
        case=order_case(),
        file_index=index,
        repository_revision=("benchmark-revision-4"),
    )

    assert evaluation.symbol_snapshot_build_ms >= 0.0

    assert evaluation.symbol_count >= 6

    assert evaluation.graph_edge_count >= 4


# ==================================
# 5. File Baseline Preservation
#
# Symbol Result Fusion 只能补充，
# 不能重排 / 删除原 File Selection。
# ==================================


def test_symbol_fusion_preserves_file_baseline(
    tmp_path: Path,
):

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("benchmark-revision-5"),
    )

    case = order_case()

    direct = SymbolRanker().rank(
        queries=case.queries,
        symbol_index=(snapshot.symbol_index),
        limit=(case.rank_limit),
    )

    expanded = SymbolGraphExpander().expand(
        candidates=direct,
        symbol_index=(snapshot.symbol_index),
        symbol_graph=(snapshot.symbol_graph),
        max_neighbors=(case.max_graph_neighbors),
    )

    records = {item.relative_path: item for item in index.files}

    baseline_paths = [
        "src/views/OrdersView.vue",
        "src/api/orders.ts",
    ]

    baseline_files = []

    for rank, path in enumerate(
        baseline_paths,
        start=1,
    ):

        record = records[path]

        baseline_files.append(
            HybridRelevantFile(
                relative_path=path,
                category=record.category,
                language=record.language,
                size_bytes=(record.size_bytes),
                retrieval_score=1.0,
                retrieval_rank=rank,
                policy_score=0,
                selection_score=1.0,
                context_role=("implementation_target"),
                role_reason=("benchmark baseline"),
                target_match_type="none",
            )
        )

    baseline = HybridRelevantFileSelection(
        candidate_count=2,
        eligible_count=2,
        selected_count=2,
        total_selected_bytes=sum(item.size_bytes for item in baseline_files),
        selected_files=(baseline_files),
        skipped_by_budget=[],
        skipped_non_indexable=[],
    )

    result = SymbolResultFusion().fuse(
        selection=baseline,
        index=index,
        candidates=expanded,
        max_files=3,
        max_total_bytes=(1_000_000),
    )

    final_paths = [item.relative_path for item in result.selected_files]

    # 原 File Retrieval 是严格前缀。
    assert final_paths[:2] == baseline_paths

    assert final_paths == [
        "src/views/OrdersView.vue",
        "src/api/orders.ts",
        "src/mocks/service.ts",
    ]

    evaluation = SymbolRetrievalBenchmarkEvaluator().evaluate(
        case=case,
        direct_candidates=direct,
        expanded_candidates=(expanded),
        symbol_count=len(snapshot.symbol_index),
        graph_edge_count=len(snapshot.symbol_graph),
        baseline_file_paths=(baseline_paths),
        final_file_paths=(final_paths),
    )

    assert evaluation.baseline_preserved is True

    assert evaluation.status == "passed"


# ==================================
# 6. Benchmark 自身必须能识别噪声
#
# 防止 Evaluator 永远报好成绩。
# ==================================


def test_evaluator_detects_graph_noise():

    from agent_core.codebase.symbol_retrieval_model import (
        SymbolCandidate,
    )

    expected = SymbolCandidate(
        symbol_id="orders::listOrders",
        name="listOrders",
        kind="function",
        relative_path="orders.ts",
        start_line=1,
        end_line=3,
        score=1.0,
        source="name",
    )

    noise = SymbolCandidate(
        symbol_id="users::listUsers",
        name="listUsers",
        kind="function",
        relative_path="users.ts",
        start_line=1,
        end_line=3,
        score=0.5,
        source="graph",
        depth=1,
        seed_symbol_id=(expected.symbol_id),
        relation="calls",
        direction="outgoing",
    )

    case = SymbolRetrievalBenchmarkCase(
        case_id=("BENCHMARK-SELF-CHECK"),
        queries=["listOrders"],
        expected_symbol_ids=[expected.symbol_id],
    )

    evaluation = SymbolRetrievalBenchmarkEvaluator().evaluate(
        case=case,
        direct_candidates=[expected],
        expanded_candidates=[
            expected,
            noise,
        ],
    )

    assert evaluation.symbol_recall == 1.0

    assert evaluation.symbol_precision == 0.5

    assert evaluation.noise_expansion_rate == 1.0

    assert evaluation.graph_noise_symbol_ids == [noise.symbol_id]
