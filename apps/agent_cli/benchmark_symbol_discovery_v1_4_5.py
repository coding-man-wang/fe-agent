from __future__ import annotations

import json
import os
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


RAW_REQUIREMENT = "给订单列表增加状态筛选，" "不要修改用户列表"


def resolve_repository_root() -> Path:

    configured = os.getenv("FE_AGENT_BENCHMARK_WEB_ROOT")

    if configured:

        root = Path(configured).resolve()

    else:

        fe_agent_root = Path(__file__).resolve().parents[2]

        root = (fe_agent_root.parent / "fe-agent-benchmark-web").resolve()

    if not root.is_dir():

        raise RuntimeError(
            "找不到 fe-agent-benchmark-web。\n"
            "请设置环境变量：\n"
            "FE_AGENT_BENCHMARK_WEB_ROOT"
        )

    return root


def build_requirement_spec() -> RequirementSpec:

    return RequirementSpec(
        title="订单列表状态筛选",
        summary=("为订单列表增加按状态筛选能力，" "保持现有订单加载行为。"),
        goals=[
            "支持订单列表按状态过滤",
            "保持订单数据加载能力",
        ],
        non_goals=[
            "不修改用户列表功能",
        ],
        acceptance_criteria=[
            ("选择订单状态后，" "只展示匹配状态的订单"),
            ("订单列表仍能正常加载"),
        ],
        constraints=[
            "不要修改用户列表功能",
        ],
    )


def build_task() -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-ORDER-FILTER",
        task_type="code_change",
        title="实现订单列表状态筛选",
        objective=("让订单列表支持按状态过滤，" "并保持现有订单加载行为"),
        source_step_order=1,
        responsibilities=[
            ("定位订单列表的数据加载责任"),
            ("实现订单状态过滤行为"),
        ],
        investigation_questions=[
            ("订单列表的数据从哪里加载？"),
            ("订单状态数据在哪里定义和展示？"),
        ],
        repository_hints=[],
        target_modules=[],
        expected_outputs=[
            "订单列表支持状态筛选",
        ],
        verification=[
            "验证订单列表筛选行为",
        ],
        acceptance_criteria_refs=[
            "选择订单状态后，只展示匹配状态的订单",
            "订单列表仍能正常加载",
        ],
    )


def main() -> None:

    root = resolve_repository_root()

    print("Repository:")

    print(root)

    # ==================================
    # 1. Repository Evidence
    # ==================================

    file_index = RepositoryFileIndexer().build(root)

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=file_index,
        repository_revision=("natural-language-" "discovery-benchmark"),
    )

    # ==================================
    # 2. Natural-language Query
    #
    # 这里禁止人为写：
    #
    # listOrders
    #
    # Query 必须真正来自 Requirement /
    # Task。
    # ==================================

    query_builder = RetrievalQueryBuilder(max_queries=6)

    requirement_spec = build_requirement_spec()

    task = build_task()

    query_specs = query_builder.build(
        task=task,
        raw_requirement=(RAW_REQUIREMENT),
        requirement_spec=(requirement_spec),
    )

    symbol_ranker = SymbolRanker()

    symbol_expander = SymbolGraphExpander()

    natural_direct = symbol_ranker.rank(
        queries=[query.text for query in query_specs],
        symbol_index=(snapshot.symbol_index),
    )

    natural_expanded = symbol_expander.expand(
        candidates=(natural_direct),
        symbol_index=(snapshot.symbol_index),
        symbol_graph=(snapshot.symbol_graph),
    )

    # ==================================
    # 3. Known-Identifier Control
    #
    # 控制组：
    #
    # 已知目标名称 listOrders。
    #
    # 用于证明 SymbolIndex / Ranker /
    # Graph 基础设施是工作的。
    # ==================================

    control_direct = symbol_ranker.rank(
        queries=["listOrders"],
        symbol_index=(snapshot.symbol_index),
    )

    control_expanded = symbol_expander.expand(
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

    print()
    print("=" * 80)

    print("Natural-language " "Symbol Discovery Benchmark")

    print("=" * 80)

    print(
        json.dumps(
            evaluation.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
