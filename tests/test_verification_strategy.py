import hashlib
from pathlib import Path

import pytest

from agent_core.benchmark.verification_strategy import (
    VerificationStrategyGuard,
    VerificationStrategyResolver,
    VerificationStrategyPatchGenerator,
    VerificationStrategyError,
)
from agent_core.policies.model import EffectivePolicy
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


def mixed_strategy(tmp_path, *, code_context=(), remote=False):
    repository = make_repository(tmp_path)
    if remote:
        repository = repository.model_copy(
            update={
                "scripts": {
                    "test": "vitest run",
                    "test:e2e": "cross-env VITE_API_MODE=remote playwright test",
                }
            }
        )
    return VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(title="用户列表单元测试和端到端验证"),
        repository=repository,
        code_context=code_context,
        execution_surface=make_e2e_surface().model_copy(
            update={
                "tests": ["tests/unit/usersStore.test.ts", "tests/e2e/users.spec.ts"],
                "create_files": [],
            }
        ),
        technical_testing_strategy=[
            "单元测试验证切换 pageSize 时 page 被重置为 1，并调用现有列表加载方法。",
            "组件测试验证选择 20 或 10 后控件值与查询模型 pageSize 保持一致。",
            "Playwright 端到端测试覆盖从默认 10 条切换为 20 条、再切回 10 条的完整流程。",
        ],
    )


def test_mixed_unknown_retains_both_layers_and_does_not_invent_runtime(tmp_path):
    strategy = mixed_strategy(tmp_path)
    assert strategy.selected_layer == "mixed"
    assert strategy.runtime_path == "unknown"
    assert strategy.remote_endpoint_hints == []
    assert (
        strategy.capability("browser_feature_network_interception").status == "unproven"
    )
    assert "unit_module_mock" in strategy.allowed_strategies
    assert "e2e_public_ui_observation" in strategy.allowed_strategies
    assert "e2e_remote_network_interception" in strategy.forbidden_strategies
    proofs = [
        proof for proof in strategy.proof_routes if proof.source == "technical_spec"
    ]
    assert [proof.preferred_layer for proof in proofs] == ["unit", "unit", "e2e"]
    assert all(proof.routing_action == "required_here" for proof in proofs)
    rules = {rule.rule_id: rule.description for rule in strategy.to_policy_rules()}
    assert "TEST-MIXED-LAYER-001" in rules
    assert "(unit) 单元测试" in rules["TEST-PROOF-REQUIRED-001"]
    assert "(e2e) Playwright" in rules["TEST-PROOF-REQUIRED-001"]
    assert "分页状态" in rules["TEST-E2E-NETWORK-001"]
    assert "module mock" in rules["TEST-E2E-NETWORK-001"]


@pytest.mark.parametrize(
    "relative_path", ["tests/e2e/users.spec.ts", "tests/helpers/users.ts"]
)
def test_mixed_unknown_rejects_actual_page_size_http_wait_without_endpoint_hints(
    tmp_path, relative_path
):
    strategy = mixed_strategy(tmp_path)
    # Minimal regression from run 35d937... TASK-005: the Browser runtime was
    # not proven remote, but a mixed task generated this impossible HTTP wait.
    proposal = make_proposal(
        relative_path=relative_path,
        content="""
import { expect, test, type Page } from '@playwright/test'
async function changePageSize(page: Page, value: '10' | '20') {
  const responsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url())
    return url.pathname.endsWith('/users') &&
      url.searchParams.get('page') === '1' &&
      url.searchParams.get('pageSize') === value
  })
  await page.getByLabel('Page size').selectOption(value)
  return responsePromise
}
""",
    )
    decision = VerificationStrategyGuard().evaluate(
        strategy=strategy, proposal=proposal
    )
    assert not decision.allowed
    assert [violation.code for violation in decision.violations] == [
        "E2E_WAIT_FOR_RESPONSE"
    ]


@pytest.mark.parametrize("remote", [False, True])
def test_mixed_network_capability_requires_proven_remote_runtime(tmp_path, remote):
    strategy = mixed_strategy(tmp_path, code_context=make_code_context(), remote=remote)
    assert strategy.runtime_path == ("remote_http" if remote else "local_mock")
    assert strategy.capability("browser_feature_network_interception").status == (
        "available" if remote else "unavailable"
    )
    proposal = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="""
import { test } from '@playwright/test'
test('loads orders', async ({ page }) => {
  await page.waitForResponse('**/orders*')
})
""",
    )
    assert (
        VerificationStrategyGuard()
        .evaluate(strategy=strategy, proposal=proposal)
        .allowed
        is remote
    )


def test_mixed_applies_browser_boundary_per_file_and_keeps_unit_mocks(tmp_path):
    strategy = mixed_strategy(tmp_path)
    unit = make_proposal(
        relative_path="tests/unit/usersStore.test.ts",
        content="""
import { expect, test, vi } from 'vitest'
vi.mock('@/api/users', () => ({ listUsers: vi.fn() }))
test('resets pageSize', () => expect(store.page).toBe(1))
""",
    )
    e2e = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="""
import { expect, test } from '@playwright/test'
test('changes page size', async ({ page }) => {
  await page.goto('/users')
  await page.getByLabel('Page size').selectOption('20')
  await expect(page.getByLabel('Page size')).toHaveValue('20')
  await expect(page.getByText('Page 1 of 2')).toBeVisible()
})
""",
    )
    combined = unit.model_copy(update={"patches": [*unit.patches, *e2e.patches]})
    assert (
        VerificationStrategyGuard()
        .evaluate(strategy=strategy, proposal=combined)
        .allowed
    )
    browser_in_unit = make_proposal(
        relative_path="tests/unit/usersStore.test.ts",
        content=e2e.patches[0].after_content,
    )
    rejected = VerificationStrategyGuard().evaluate(
        strategy=strategy, proposal=browser_in_unit
    )
    assert not rejected.allowed
    assert any(v.code == "UNIT_PLAYWRIGHT_IMPORT" for v in rejected.violations)


def test_single_e2e_unknown_cannot_use_missing_endpoint_evidence_as_permission(
    tmp_path,
):
    strategy = VerificationStrategyResolver().resolve(
        repository_root=tmp_path,
        task=make_task(),
        repository=make_repository(tmp_path),
        code_context=[],
        execution_surface=make_e2e_surface(),
    )
    proposal = make_proposal(
        relative_path="tests/e2e/orders.spec.ts",
        content="await page.waitForResponse('**/users*')",
    )
    assert strategy.runtime_path == "unknown"
    assert (
        not VerificationStrategyGuard()
        .evaluate(strategy=strategy, proposal=proposal)
        .allowed
    )


def test_known_feature_hints_preserve_unrelated_network_setup_compatibility(tmp_path):
    strategy = mixed_strategy(tmp_path, code_context=make_code_context())
    proposal = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="await page.route('**/analytics*', route => route.abort())",
    )
    assert strategy.remote_endpoint_hints == ["/orders"]
    assert (
        VerificationStrategyGuard()
        .evaluate(strategy=strategy, proposal=proposal)
        .allowed
    )


@pytest.mark.parametrize(
    "description",
    [
        "单元测试验证切换 pageSize 时 page 被重置为 1。",
        "组件测试验证选择 20 后控件值与查询模型 pageSize 保持一致。",
        "Unit tests verify pageSize changes reload users.",
        "Component tests verify rendered pagination state.",
    ],
)
def test_explicit_unit_proof_precedes_page_and_ui_keywords(description):
    assert (
        VerificationStrategyResolver()._preferred_layer_for_proof(
            description=description,
            current_layer="mixed",
            runtime_path="unknown",
        )
        == "unit"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [False, True])
async def test_mixed_guard_rejects_before_apply_and_allows_only_bounded_strategy_correction(
    tmp_path, monkeypatch, corrected
):
    from agent_core.benchmark import verification_strategy as module

    strategy = mixed_strategy(tmp_path)
    bad = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="await page.waitForResponse('**/users*')",
    )
    good = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="await expect(page.getByLabel('Page size')).toHaveValue('20')",
    )
    seen_policies = []

    async def generate(self, **kwargs):
        seen_policies.append(kwargs["effective_policy"])
        return good if corrected and len(seen_policies) == 2 else bad

    monkeypatch.setattr(module.PatchGenerator, "generate", generate)
    generator = VerificationStrategyPatchGenerator(
        repository_root=tmp_path, model=object(), max_corrections=1
    )
    arguments = dict(
        artifact=object(),
        plan=object(),
        preparation=object(),
        effective_policy=EffectivePolicy(rules=[]),
        strategy=strategy,
    )
    if corrected:
        result = await generator.generate(**arguments)
        assert result.proposal == good
        assert result.correction_count == 1
    else:
        with pytest.raises(VerificationStrategyError, match="E2E_WAIT_FOR_RESPONSE"):
            await generator.generate(**arguments)
    assert len(seen_policies) == 2
    assert any(
        rule.rule_id == "TEST-STRATEGY-CORRECTION-001"
        for rule in seen_policies[-1].rules
    )


def test_incomplete_legacy_mixed_plan_cannot_grant_unproven_network_permission(
    tmp_path,
):
    strategy = mixed_strategy(tmp_path).model_copy(
        update={
            "allowed_strategies": [],
            "forbidden_strategies": [],
        }
    )
    proposal = make_proposal(
        relative_path="tests/e2e/users.spec.ts",
        content="await page.waitForResponse('**/users*')",
    )
    assert (
        not VerificationStrategyGuard()
        .evaluate(strategy=strategy, proposal=proposal)
        .allowed
    )


@pytest.mark.parametrize(
    "description",
    [
        "Playwright E2E must inject remote HTTP failure",
        "Playwright 端到端测试注入错误并验证 error 状态",
        "E2E verifies request parameter mapping",
    ],
)
def test_explicit_e2e_label_does_not_override_remote_or_failure_capability(description):
    routes = VerificationStrategyResolver()._build_proof_routes(
        task=make_task(),
        selected_layer="e2e",
        runtime_path="local_mock",
        requirement_acceptance_criteria=[description],
        technical_testing_strategy=[],
    )
    proof = next(route for route in routes if route.description == description)
    assert proof.preferred_layer == "unit"
    assert proof.routing_action == "deferred"
    assert not proof.observable_in_current_layer
