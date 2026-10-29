import pytest

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.verification_repair_integrity import (
    VerificationRepairIntegrityError,
    VerificationRepairIntegrityGuard,
)


def build_proposal(
    *,
    content: str,
    path: str = ("tests/e2e/" "orders-status-filter.spec.ts"),
) -> FilePatchProposal:

    return FilePatchProposal(
        task_id="TASK-TEST",
        artifact_id="artifact-1",
        context_hash="context-1",
        patches=[
            ValidatedFilePatch(
                action="create",
                relative_path=path,
                expected_sha256=None,
                after_sha256=("a" * 64),
                before_content=None,
                after_content=content,
                explanation="test",
                unified_diff="diff",
            )
        ],
    )


def test_allows_legitimate_test_repair():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
import { test, expect } from '@playwright/test'

test('shows error', async ({ page }) => {
  await page.route(
    '**/orders*',
    route => route.abort()
  )

  await page.goto('/orders')

  await expect(
    page.getByText('Unable to load orders')
  ).toBeVisible()
})
""")

    repaired = build_proposal(content="""
import { test, expect } from '@playwright/test'

test('shows error', async ({ page }) => {
  await page.route(
    '**/api/orders*',
    route => route.abort()
  )

  await page.goto('/orders')

  await expect(
    page.getByText('Unable to load orders')
  ).toBeVisible()
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is True
    assert result.violations == ()


def test_rejects_new_test_skip():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('shows error', async () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
test.skip('shows error', async () => {
  expect(true).toBe(true)
})
""")

    with pytest.raises(
        VerificationRepairIntegrityError,
        match="repair_test_disabled",
    ):
        guard.enforce(
            failed_proposal=failed,
            repaired_proposal=repaired,
        )


def test_rejects_new_test_todo():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('shows error', async () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
test.todo('shows error')
""")

    with pytest.raises(VerificationRepairIntegrityError):
        guard.enforce(
            failed_proposal=failed,
            repaired_proposal=repaired,
        )


def test_rejects_removed_test_case():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('filters pending', () => {
  expect(true).toBe(true)
})

test('shows error', () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
test('filters pending', () => {
  expect(true).toBe(true)
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is False

    assert any(
        violation.code == "repair_test_case_removed" for violation in result.violations
    )


def test_rejects_renamed_test_case():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('shows remote request error', () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
test('remote mode only', () => {
  expect(true).toBe(true)
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is False

    assert any(
        violation.code == "repair_test_case_removed" for violation in result.violations
    )


def test_rejects_reduced_assertions():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('filters orders', () => {
  expect(1).toBe(1)
  expect(2).toBe(2)
})
""")

    repaired = build_proposal(content="""
test('filters orders', () => {
  expect(1).toBe(1)
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is False

    assert any(
        violation.code == "repair_assertion_reduced" for violation in result.violations
    )


def test_allows_same_assertion_count():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('filters orders', () => {
  expect(value).toBe('pending')
})
""")

    repaired = build_proposal(content="""
test('filters orders', () => {
  expect(value).toBe('paid')
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is True


def test_rejects_new_test_only():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('filters orders', () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
test.only('filters orders', () => {
  expect(true).toBe(true)
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is False

    assert any(
        violation.code == "repair_test_disabled" for violation in result.violations
    )


def test_comments_do_not_trigger_skip_detection():
    guard = VerificationRepairIntegrityGuard()

    failed = build_proposal(content="""
test('filters orders', () => {
  expect(true).toBe(true)
})
""")

    repaired = build_proposal(content="""
// Do not use test.skip(...) here.

test('filters orders', () => {
  expect(true).toBe(true)
})
""")

    result = guard.evaluate(
        failed_proposal=failed,
        repaired_proposal=repaired,
    )

    assert result.allowed is True
