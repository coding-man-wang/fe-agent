import json

import pytest

from agent_core.benchmark.execution_model import (
    VerificationExecutionAttempt,
)
from agent_core.benchmark.verification_repair_diagnostics import (
    analyze_repair_outcome,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)
from agent_core.coding.verification_repair import (
    VerificationRepairError,
    VerificationRepairService,
)
from agent_core.coding.verification_repair_coverage import (
    inspect_repair_input_coverage,
)
from agent_core.coding.verification_repair_integrity import (
    VerificationRepairIntegrityGuard,
)
from agent_core.context.model import (
    TaskContextBundle,
    ContextBudgetUsage,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.sdd.model import (
    RequirementSpec,
    TechnicalSpec,
    TaskDefinition,
)

PROPOSAL_ID = "a" * 64
TEST_PATH = "tests/e2e/users.spec.ts"

FAILED_TEST = """
import { test, expect } from '@playwright/test'

test('switches page size', async ({ page }) => {
  const initialFirstRow: any =
    'User 1 user1@example.com admin disabled'

  await expect(
    page.locator('tbody tr').first()
  ).toHaveText(initialFirstRow)
})
"""

# 只修复 no-explicit-any，不修复错误的 E2E 文本断言。
LINT_ONLY_FIX = FAILED_TEST.replace(
    "initialFirstRow: any",
    "initialFirstRow: string",
)


class FakeLLM:
    def __init__(self, response: dict):
        self.response = response
        self.messages = []

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        self.messages.append(
            {
                "system": system_message,
                "user": json.loads(user_message),
            }
        )
        return json.dumps(self.response)


def make_verification() -> VerificationRunRecord:
    checks = [
        VerificationCheck(
            name="frontend lint",
            command_id="frontend-lint",
            status="failed",
            details="no-explicit-any in users.spec.ts",
        ),
        VerificationCheck(
            name="frontend typecheck",
            command_id="frontend-typecheck",
            status="passed",
        ),
        VerificationCheck(
            name="frontend unit",
            command_id="frontend-unit-test",
            status="passed",
        ),
        VerificationCheck(
            name="frontend e2e",
            command_id="frontend-e2e",
            status="failed",
            details=(
                'Expected: "User 1 user1@example.com '
                'admin disabled"\n'
                'Received: "User 1user1@example.comadmindisabled"'
            ),
        ),
    ]

    return VerificationRunRecord(
        proposal_id=PROPOSAL_ID,
        task_id="TASK-005",
        status="failed",
        result=PatchVerificationResult(
            proposal_id=PROPOSAL_ID,
            task_id="TASK-005",
            status="failed",
            checks=checks,
        ),
    )


def make_proposal() -> FilePatchProposal:
    return FilePatchProposal(
        task_id="TASK-005",
        artifact_id="artifact-1",
        context_hash="context-1",
        patches=[
            ValidatedFilePatch(
                action="create",
                relative_path=TEST_PATH,
                expected_sha256=None,
                after_sha256="a" * 64,
                before_content=None,
                after_content=FAILED_TEST,
                explanation="add e2e coverage",
                unified_diff="diff",
            )
        ],
    )


def make_bundle(tmp_path) -> TaskContextBundle:
    return TaskContextBundle(
        requirement=RequirementSpec(
            title="Pagination",
            summary="Validate page size switching",
        ),
        technical_spec=TechnicalSpec(
            summary="E2E pagination verification",
        ),
        task=TaskDefinition(
            task_id="TASK-005",
            task_type="test",
            title="Verify pagination",
            objective="Validate 10/20 page size",
            source_step_order=5,
        ),
        repository=RepositoryProfile(
            project_name="benchmark",
            root_path=str(tmp_path),
        ),
        budget=ContextBudgetUsage(
            max_files=6,
            max_total_chars=30000,
            max_chars_per_file=8000,
            used_files=0,
            used_chars=0,
        ),
    )


@pytest.mark.asyncio
async def test_fake_llm_receives_all_failed_checks(tmp_path):
    model = FakeLLM(
        {
            "task_id": "TASK-005",
            "repairable": False,
            "reason": "Cannot safely repair in current boundary",
            "changes": [],
        }
    )

    service = VerificationRepairService(model=model)

    result = await service.repair(
        proposal=make_proposal(),
        verification=make_verification(),
        bundle=make_bundle(tmp_path),
        repository_root=tmp_path,
    )

    assert result.proposal is None
    assert len(model.messages) == 1

    message = model.messages[0]
    payload = message["user"]

    assert payload["repair_mode"] == "mixed"

    ids = {check["command_id"] for check in payload["verification_failures"]}

    assert ids == {
        "frontend-lint",
        "frontend-e2e",
    }

    assert "MIXED" in message["system"].upper()

    details = {
        item["command_id"]: item["details"] for item in payload["verification_failures"]
    }

    assert "no-explicit-any" in details["frontend-lint"]
    assert "Received:" in details["frontend-e2e"]


def test_missing_e2e_payload_is_detected():
    verification = make_verification()

    report = inspect_repair_input_coverage(
        verification=verification,
        payload={
            "verification_failures": [
                {
                    "name": "frontend lint",
                    "command_id": "frontend-lint",
                    "status": "failed",
                    "details": "no-explicit-any in users.spec.ts",
                }
            ]
        },
        truncate_details=lambda value: value,
    )

    assert report["complete"] is False
    assert "failed_checks_missing_from_payload" in report["issues"]


@pytest.mark.asyncio
async def test_incomplete_input_blocks_before_llm(
    tmp_path,
    monkeypatch,
):
    model = FakeLLM(
        {
            "task_id": "TASK-005",
            "repairable": False,
            "reason": "not repairable",
            "changes": [],
        }
    )

    service = VerificationRepairService(model=model)

    # 模拟数据构造阶段意外漏掉 E2E 失败。
    monkeypatch.setattr(
        service,
        "_build_payload",
        lambda **kwargs: {
            "verification_failures": [
                {
                    "name": "frontend lint",
                    "command_id": "frontend-lint",
                    "status": "failed",
                    "details": "no-explicit-any in users.spec.ts",
                }
            ]
        },
    )

    with pytest.raises(
        VerificationRepairError,
        match="Input Coverage",
    ):
        await service.repair(
            proposal=make_proposal(),
            verification=make_verification(),
            bundle=make_bundle(tmp_path),
            repository_root=tmp_path,
        )

    assert model.messages == []


@pytest.mark.asyncio
async def test_lint_only_repair_does_not_prove_success(
    tmp_path,
):
    model = FakeLLM(
        {
            "task_id": "TASK-005",
            "repairable": True,
            "reason": "Fix TypeScript lint annotation",
            "changes": [
                {
                    "relative_path": TEST_PATH,
                    "after_content": LINT_ONLY_FIX,
                    "explanation": "replace any with string",
                }
            ],
        }
    )

    original = make_proposal()

    result = await VerificationRepairService(model=model).repair(
        proposal=original,
        verification=make_verification(),
        bundle=make_bundle(tmp_path),
        repository_root=tmp_path,
    )

    assert result.proposal is not None

    # Patch Scope / Test Integrity 可以通过，
    # 但这不等于 E2E 已经通过。
    integrity = VerificationRepairIntegrityGuard().evaluate(
        failed_proposal=original,
        repaired_proposal=result.proposal,
    )

    assert integrity.allowed is True

    def attempt(number, kind, status, e2e_status):
        checks = [
            VerificationCheck(
                name="lint",
                command_id="frontend-lint",
                status=("failed" if number == 1 else "passed"),
            ),
            VerificationCheck(
                name="e2e",
                command_id="frontend-e2e",
                status=e2e_status,
            ),
        ]

        return VerificationExecutionAttempt(
            attempt=number,
            kind=kind,
            status=status,
            proposal_id=str(number) * 64,
            workspace_id=f"workspace-{number}",
            checks=checks,
        )

    # Fake Verification：模拟真实 CASE-006 的结果。
    # 不声称在这个单测里执行过 Playwright。
    report = analyze_repair_outcome(
        [
            attempt(1, "initial", "failed", "failed"),
            attempt(2, "repair", "failed", "failed"),
        ]
    )

    assert report["status"] == "partial_repair"
    assert report["unresolved_checks"] == ["frontend-e2e"]
    assert report["all_checks_passed"] is False
    assert report["task_completion_allowed"] is False
