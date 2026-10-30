import hashlib
from pathlib import Path

from agent_core.benchmark.verification_strategy import (
    VerificationStrategyGuard,
    VerificationStrategyResolver,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def make_task(
    *,
    task_id: str = "TASK-011",
    title: str = "E2E 验证订单筛选",
    verification=None,
):
    return TaskDefinition(
        task_id=task_id,
        task_type="test",
        title=title,
        objective=("验证订单筛选行为"),
        source_step_order=1,
        depends_on=[],
        inputs=[],
        target_modules=[],
        expected_outputs=["订单筛选行为有自动化验证"],
        verification=(
            verification
            or [
                ("切换筛选后页面" "重新加载并显示正确结果"),
                ("Remote 请求失败时" "保留错误状态"),
            ]
        ),
        acceptance_criteria_refs=[],
    )


def make_repository(
    root: Path,
) -> RepositoryProfile:

    return RepositoryProfile(
        project_name="demo",
        root_path=str(root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
        test_stack=[
            "Vitest",
            "Playwright",
        ],
        scripts={
            "test": "vitest run",
            "test:e2e": "playwright test",
        },
        dependencies=[
            "vue",
            "@playwright/test",
            "vitest",
        ],
    )


def make_code_context():
    content = """
import { request } from './requestClient'
import { mockListOrders } from '@/mocks/service'

const useMock =
  import.meta.env.VITE_API_MODE !== 'remote'

export function listOrders(query) {
  if (useMock) {
    return mockListOrders(query)
  }

  return request({
    method: 'GET',
    url: '/orders',
    params: {
      status: query.status,
    },
  })
}
""".strip()

    return [
        CodeContextItem(
            relative_path=("src/api/orders.ts"),
            category="source",
            language="TypeScript",
            content_mode="full_file",
            content=content,
            start_line=1,
            end_line=20,
            relevance_reasons=["test fixture"],
            content_chars=len(content),
        )
    ]


def make_e2e_surface():
    return ResolvedChangeSurface(
        task_id="TASK-011",
        implementation_targets=[],
        dependencies=[],
        references=["src/api/orders.ts"],
        tests=["tests/e2e/orders.spec.ts"],
        create_files=["tests/e2e/orders.spec.ts"],
        unresolved=[],
        rationale=["e2e fixture"],
        execution_disposition=("change_required"),
    )


def make_unit_surface():
    return ResolvedChangeSurface(
        task_id="TASK-010",
        implementation_targets=[],
        dependencies=[],
        references=["src/api/orders.ts"],
        tests=["tests/unit/" "ordersApi.test.ts"],
        create_files=["tests/unit/" "ordersApi.test.ts"],
        unresolved=[],
        rationale=["unit fixture"],
        execution_disposition=("change_required"),
    )


def test_local_mock_e2e_disables_remote_interception(
    tmp_path: Path,
):
    repository = make_repository(tmp_path)

    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=repository,
        code_context=(make_code_context()),
        execution_surface=(make_e2e_surface()),
        requirement_acceptance_criteria=[
            ("切换筛选后" "重新加载订单"),
            ("请求失败时" "显示错误状态"),
        ],
        technical_testing_strategy=["验证 Remote / Mock " "筛选语义一致"],
    )

    assert strategy.selected_layer == "e2e"

    assert strategy.runtime_path == "local_mock"

    assert "e2e_remote_network_interception" in strategy.forbidden_strategies

    capability = strategy.capability(("browser_feature_" "network_interception"))

    assert capability is not None

    assert capability.status == "unavailable"

    assert "/orders" in strategy.remote_endpoint_hints


def test_error_and_remote_contract_are_routed_to_unit(
    tmp_path: Path,
):
    repository = make_repository(tmp_path)

    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=repository,
        code_context=(make_code_context()),
        execution_surface=(make_e2e_surface()),
        requirement_acceptance_criteria=[
            ("Remote 请求参数" "必须携带 status"),
            ("请求失败时" "显示错误状态"),
            ("切换筛选后" "显示正确订单"),
        ],
    )

    remote = next(
        item for item in strategy.proof_routes if ("status" in item.description)
    )

    assert remote.preferred_layer == "unit"

    assert remote.routing_action == "deferred"

    error = next(item for item in strategy.proof_routes if ("错误" in item.description))

    assert error.preferred_layer == "unit"

    assert error.routing_action == "deferred"

    ui = next(
        item for item in strategy.proof_routes if ("正确订单" in item.description)
    )

    assert ui.preferred_layer == "e2e"

    assert ui.routing_action == "required_here"


def test_unit_layer_owns_remote_and_failure_proofs(
    tmp_path: Path,
):
    repository = make_repository(tmp_path)

    task = make_task(
        task_id="TASK-010",
        title="订单筛选单元测试",
    )

    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=task,
        repository=repository,
        code_context=(make_code_context()),
        execution_surface=(make_unit_surface()),
        requirement_acceptance_criteria=[
            ("Remote 请求参数" "必须携带 status"),
            ("请求失败时" "显示错误状态"),
        ],
    )

    assert strategy.selected_layer == "unit"

    routes = {item.description: item.routing_action for item in strategy.proof_routes}

    assert routes[("Remote 请求参数" "必须携带 status")] == "required_here"

    assert routes[("请求失败时" "显示错误状态")] == "required_here"


def make_proposal(
    *,
    relative_path: str,
    content: str,
) -> FilePatchProposal:

    before = "// before\n"

    return FilePatchProposal(
        task_id="TASK-011",
        artifact_id="a" * 64,
        context_hash="b" * 64,
        patches=[
            ValidatedFilePatch(
                action="modify",
                relative_path=(relative_path),
                expected_sha256=(hashlib.sha256(before.encode("utf-8")).hexdigest()),
                after_sha256=(hashlib.sha256(content.encode("utf-8")).hexdigest()),
                before_content=before,
                after_content=content,
                explanation=("test fixture"),
                unified_diff=("--- before\n" "+++ after\n"),
            )
        ],
    )


def test_guard_rejects_page_route_for_local_mock_feature(
    tmp_path: Path,
):
    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=(make_repository(tmp_path)),
        code_context=(make_code_context()),
        execution_surface=(make_e2e_surface()),
    )

    proposal = make_proposal(
        relative_path=("tests/e2e/orders.spec.ts"),
        content="""
import { test } from '@playwright/test'

test('remote error', async ({ page }) => {
  await page.route('**/orders*', route => route.abort())
})
""".strip(),
    )

    decision = VerificationStrategyGuard().evaluate(
        strategy=strategy,
        proposal=proposal,
    )

    assert not decision.allowed

    assert any(item.code == "E2E_PAGE_ROUTE" for item in decision.violations)


def test_guard_allows_visible_behavior_for_local_mock(
    tmp_path: Path,
):
    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=(make_repository(tmp_path)),
        code_context=(make_code_context()),
        execution_surface=(make_e2e_surface()),
    )

    proposal = make_proposal(
        relative_path=("tests/e2e/orders.spec.ts"),
        content="""
import { test, expect } from '@playwright/test'

test('filters orders', async ({ page }) => {
  await page.goto('/orders')
  await page.getByRole('button', { name: 'Pending' }).click()
  await expect(page.getByText('Pending')).toBeVisible()
})
""".strip(),
    )

    decision = VerificationStrategyGuard().evaluate(
        strategy=strategy,
        proposal=proposal,
    )

    assert decision.allowed


def test_explicit_remote_e2e_enables_network_interception(
    tmp_path: Path,
):
    repository = make_repository(tmp_path)

    repository = repository.model_copy(
        deep=True,
        update={
            "scripts": {
                "test": "vitest run",
                "test:e2e": ("cross-env " "VITE_API_MODE=remote " "playwright test"),
            }
        },
    )

    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=repository,
        code_context=(make_code_context()),
        execution_surface=(make_e2e_surface()),
    )

    assert strategy.runtime_path == "remote_http"

    assert "e2e_remote_network_interception" in strategy.allowed_strategies

    assert "e2e_remote_network_interception" not in strategy.forbidden_strategies
