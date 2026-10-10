from pathlib import Path

import pytest

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.completion import (
    TaskCompletionError,
    TaskCompletionGate,
)
from agent_core.coding.completion_store import (
    TaskCompletionStore,
)
from agent_core.coding.execution_journal_model import (
    FileExecutionRecord,
    PatchExecutionJournal,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)
from tests.test_recovery import (
    AFTER_A,
    AFTER_B,
    create_repository,
    make_proposal,
)


def make_applied_journal(
    proposal,
):
    review = PatchReviewer().prepare(proposal)

    return PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        status="applied",
        files=[
            FileExecutionRecord(
                relative_path=(patch.relative_path),
                expected_sha256=(patch.expected_sha256),
                after_sha256=(patch.after_sha256),
                status="applied",
            )
            for patch in proposal.patches
        ],
    )


def make_passed_verification(
    *,
    proposal_id: str,
    task_id: str,
):
    result = PatchVerificationResult(
        proposal_id=proposal_id,
        task_id=task_id,
        status="passed",
        checks=[
            VerificationCheck(
                name="test",
                status="passed",
                details="ok",
            )
        ],
    )

    return VerificationRunRecord(
        proposal_id=proposal_id,
        task_id=task_id,
        status="succeeded",
        attempts=1,
        result=result,
    )


def create_completion_environment(
    root: Path,
):
    proposal = make_proposal()

    create_repository(
        root,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    journal = make_applied_journal(proposal)

    review = PatchReviewer().prepare(proposal)

    approval_store = ApprovalStore(root)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    completion_store = TaskCompletionStore(root)

    gate = TaskCompletionGate(
        repository_root=root,
        approval_store=(approval_store),
        completion_store=(completion_store),
    )

    verification = make_passed_verification(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
    )

    return (
        proposal,
        journal,
        verification,
        gate,
        completion_store,
        review,
    )


def test_task_completes_after_verification_passed(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        verification,
        gate,
        completion_store,
        review,
    ) = create_completion_environment(tmp_path)

    result = gate.complete(
        proposal=proposal,
        journal=journal,
        verification=verification,
    )

    assert result.status == "completed"

    assert result.execution_status == "applied"

    assert result.verification_status == "passed"

    persisted = completion_store.load(review.proposal_id)

    assert persisted.status == "completed"


def test_task_cannot_complete_if_verification_failed(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        verification,
        gate,
        _,
        _,
    ) = create_completion_environment(tmp_path)

    verification = verification.model_copy(
        deep=True,
        update={
            "status": "failed",
        },
    )

    with pytest.raises(
        TaskCompletionError,
        match="尚未成功",
    ):
        gate.complete(
            proposal=proposal,
            journal=journal,
            verification=verification,
        )


def test_task_cannot_complete_if_repository_changed(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        verification,
        gate,
        _,
        _,
    ) = create_completion_environment(tmp_path)

    path = tmp_path / "src/api/user.ts"

    path.write_bytes(b"human change\n")

    with pytest.raises(
        TaskCompletionError,
        match="仓库状态",
    ):
        gate.complete(
            proposal=proposal,
            journal=journal,
            verification=verification,
        )


def test_completion_is_idempotent(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        verification,
        gate,
        _,
        _,
    ) = create_completion_environment(tmp_path)

    first = gate.complete(
        proposal=proposal,
        journal=journal,
        verification=verification,
    )

    second = gate.complete(
        proposal=proposal,
        journal=journal,
        verification=verification,
    )

    assert first == second
