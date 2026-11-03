from __future__ import annotations

from pathlib import Path

from agent_core.benchmark.symbol_discovery_evaluator import (
    SymbolDiscoveryEvaluator,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.retrieval_query import (
    RetrievalQueryBuilder,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
    SymbolRanker,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)

LIST_ORDERS_ID = "src/api/orders.ts" "::function" "::listOrders"


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
export function listOrders() {
  return []
}
""",
    )

    write_file(
        root,
        "src/api/users.ts",
        """
export function listUsers() {
  return []
}
""",
    )


def build_requirement() -> RequirementSpec:

    return RequirementSpec(
        title="订单列表状态筛选",
        summary=("为订单列表增加状态筛选能力"),
        goals=[
            "支持订单状态过滤",
        ],
        acceptance_criteria=[
            ("订单列表只展示" "符合状态条件的数据"),
        ],
    )


def build_task() -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="实现订单状态筛选",
        objective=("支持订单列表按状态过滤"),
        source_step_order=1,
        responsibilities=[
            "定位订单列表的数据加载责任",
            "实现订单状态过滤行为",
        ],
        investigation_questions=[
            ("订单列表的数据" "从哪里加载？"),
        ],
        repository_hints=[],
        target_modules=[],
        expected_outputs=[
            "订单状态筛选可用",
        ],
        verification=[
            "验证订单筛选行为",
        ],
    )


def test_query_builder_does_not_invent_repository_symbol_name():

    query_specs = RetrievalQueryBuilder().build(
        task=build_task(),
        raw_requirement=("给订单列表增加状态筛选"),
        requirement_spec=(build_requirement()),
    )

    query_texts = [query.text for query in query_specs]

    assert query_texts

    # ==================================
    # 重点：
    #
    # Planner / QueryBuilder
    # 不应该凭空猜：
    #
    # listOrders
    # ==================================

    assert not any("listorders" in query.casefold() for query in query_texts)


def test_natural_language_symbol_discovery_baseline(
    tmp_path: Path,
):

    create_repository(tmp_path)

    file_index = RepositoryFileIndexer().build(tmp_path)

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=file_index,
        repository_revision=("test-revision"),
    )

    query_specs = RetrievalQueryBuilder().build(
        task=build_task(),
        raw_requirement=("给订单列表增加状态筛选"),
        requirement_spec=(build_requirement()),
    )

    ranker = SymbolRanker()

    expander = SymbolGraphExpander()

    # ==================================
    # Natural-language Arm
    # ==================================

    natural_direct = ranker.rank(
        queries=[query.text for query in query_specs],
        symbol_index=(snapshot.symbol_index),
    )

    natural_expanded = expander.expand(
        candidates=(natural_direct),
        symbol_index=(snapshot.symbol_index),
        symbol_graph=(snapshot.symbol_graph),
    )

    # ==================================
    # Known-Identifier Control
    # ==================================

    control_direct = ranker.rank(
        queries=["listOrders"],
        symbol_index=(snapshot.symbol_index),
    )

    control_expanded = expander.expand(
        candidates=(control_direct),
        symbol_index=(snapshot.symbol_index),
        symbol_graph=(snapshot.symbol_graph),
    )

    evaluation = SymbolDiscoveryEvaluator().evaluate(
        target_symbol_id=(LIST_ORDERS_ID),
        target_symbol_name=("listOrders"),
        query_specs=(query_specs),
        natural_direct=(natural_direct),
        natural_expanded=(natural_expanded),
        control_direct=(control_direct),
        control_expanded=(control_expanded),
    )

    # QueryBuilder 没有作弊。
    assert evaluation.query_contains_target_identifier is False

    # 控制组证明：
    #
    # SymbolIndex + SymbolRanker
    # 对已知代码标识符是工作的。
    assert evaluation.control_direct_hit is True

    assert evaluation.control_expanded_hit is True

    # ==================================
    # 当前 v1.4.5 Baseline
    #
    # 这里有意记录目前的能力缺口。
    #
    # 将来 Evidence-driven
    # Symbol Discovery 实现后，
    # 这部分测试会升级。
    # ==================================

    assert evaluation.natural_language_direct_hit is False

    assert evaluation.natural_language_expanded_hit is False

    assert evaluation.discovery_gap_detected is True
