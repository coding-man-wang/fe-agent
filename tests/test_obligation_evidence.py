from pathlib import Path

from agent_core.benchmark.obligation_evidence import (
    ObligationEvidenceEnricher,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    ExecutionObligation,
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


def create_bundle(
    tmp_path: Path,
) -> TaskContextBundle:

    orders_content = (tmp_path / "src/api/orders.ts").read_text(encoding="utf-8")

    task = TaskDefinition(
        task_id="TASK-004",
        task_type="code_change",
        title=("订单筛选生产实现"),
        objective=("实现订单筛选完整生产闭环"),
        source_step_order=1,
        responsibilities=[
            "Remote 参数映射",
        ],
        execution_obligations=[
            ExecutionObligation(
                obligation_id=("TASK-004"),
                source_task_id=("TASK-004"),
                objective=("实现订单筛选"),
            ),
            ExecutionObligation(
                obligation_id=("TASK-007"),
                source_task_id=("TASK-007"),
                objective=("Remote 参数映射"),
                responsibilities=["沿用项目 Remote API " "参数传递模式"],
            ),
        ],
    )

    requirement = RequirementSpec(
        title="订单筛选",
        summary="支持订单状态筛选",
    )

    technical_spec = TechnicalSpec(
        summary="实现订单筛选",
    )

    repository = RepositoryProfile(
        project_name="demo",
        root_path=str(tmp_path),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    code_item = CodeContextItem(
        relative_path=("src/api/orders.ts"),
        category="source",
        language="TypeScript",
        content_mode="full_file",
        content=orders_content,
        start_line=1,
        end_line=max(
            1,
            orders_content.count("\n") + 1,
        ),
        relevance_reasons=[
            "implementation target",
        ],
        content_chars=len(orders_content),
    )

    budget = ContextBudgetUsage(
        max_files=2,
        max_total_chars=20_000,
        max_chars_per_file=8_000,
        used_files=1,
        used_chars=len(orders_content),
    )

    return TaskContextBundle(
        requirement=requirement,
        technical_spec=(technical_spec),
        task=task,
        repository=repository,
        engineering_rules=[],
        code_context=[code_item],
        budget=budget,
        change_surface={
            "task_id": "TASK-004",
            "implementation_targets": [
                "src/api/orders.ts",
            ],
            "dependencies": [],
            "references": [],
            "tests": [],
            "create_files": [],
            "unresolved": [],
            "rationale": [],
            "execution_disposition": "change_required",
        },
    )


def create_repository(
    tmp_path: Path,
) -> None:

    api_root = tmp_path / "src/api"

    api_root.mkdir(
        parents=True,
    )

    (api_root / "orders.ts").write_text(
        """
import { request } from './requestClient'
import { mockListOrders } from '@/mocks/service'
import type { Order } from '@/types/domain'

const useMock =
  import.meta.env.VITE_API_MODE !== 'remote'

export function listOrders() {
  if (useMock) {
    return mockListOrders()
  }

  return request<Order[]>({
    method: 'GET',
    url: '/orders'
  })
}
""".strip(),
        encoding="utf-8",
    )

    (api_root / "users.ts").write_text(
        """
import { request } from './requestClient'
import { mockListUsers } from '@/mocks/service'
import type { User } from '@/types/domain'

const useMock =
  import.meta.env.VITE_API_MODE !== 'remote'

export function listUsers(query) {
  if (useMock) {
    return mockListUsers(query)
  }

  return request<User[]>({
    method: 'GET',
    url: '/users',
    params: query
  })
}
""".strip(),
        encoding="utf-8",
    )


def test_adds_same_layer_similar_implementation(
    tmp_path: Path,
):
    create_repository(tmp_path)

    bundle = create_bundle(tmp_path)

    result = ObligationEvidenceEnricher(
        max_added_files=1,
    ).enrich(
        repository_root=(tmp_path),
        bundle=bundle,
    )

    assert result.added_files == [
        "src/api/users.ts",
    ]

    context_paths = {item.relative_path for item in result.bundle.code_context}

    assert "src/api/orders.ts" in context_paths

    assert "src/api/users.ts" in context_paths


def test_enrichment_does_not_expand_write_surface(
    tmp_path: Path,
):
    create_repository(tmp_path)

    bundle = create_bundle(tmp_path)

    original_surface = bundle.change_surface

    result = ObligationEvidenceEnricher().enrich(
        repository_root=(tmp_path),
        bundle=bundle,
    )

    assert result.bundle.change_surface == original_surface

    assert "src/api/users.ts" not in (
        result.bundle.change_surface["implementation_targets"]
    )


def test_non_compressed_task_is_not_enriched(
    tmp_path: Path,
):
    create_repository(tmp_path)

    bundle = create_bundle(tmp_path)

    task = bundle.task.model_copy(
        deep=True,
        update={
            "execution_obligations": [],
        },
    )

    bundle = bundle.model_copy(
        deep=True,
        update={
            "task": task,
        },
    )

    result = ObligationEvidenceEnricher().enrich(
        repository_root=(tmp_path),
        bundle=bundle,
    )

    assert result.added_files == []

    assert len(result.bundle.code_context) == 1
