import difflib

import pytest

from agent_core.coding.approval import (
    ApprovalError,
    ApprovalService,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.reviewer import (
    PatchReviewError,
    PatchReviewer,
    sha256_text,
)

BEFORE = "export function getUserList() {\n" "    return fetch('/api/users')\n" "}\n"

AFTER = (
    "export function getUserList(status: string) {\n"
    "    return fetch(`/api/users?status=${status}`)\n"
    "}\n"
)


def make_proposal(
    *,
    after_content: str = AFTER,
) -> FilePatchProposal:
    relative_path = "src/api/user.ts"

    diff = "".join(
        difflib.unified_diff(
            BEFORE.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )

    return FilePatchProposal(
        task_id="TASK-003",
        artifact_id="a" * 64,
        context_hash="b" * 64,
        patches=[
            ValidatedFilePatch(
                relative_path=relative_path,
                expected_sha256=sha256_text(BEFORE),
                after_sha256=sha256_text(after_content),
                before_content=BEFORE,
                after_content=after_content,
                explanation="增加 status 查询参数",
                unified_diff=diff,
            )
        ],
    )


def test_same_proposal_has_same_id():
    reviewer = PatchReviewer()

    first = reviewer.prepare(make_proposal())

    second = reviewer.prepare(make_proposal())

    assert first.proposal_id == second.proposal_id

    assert len(first.proposal_id) == 64


def test_changed_content_changes_proposal_id():
    reviewer = PatchReviewer()

    first = reviewer.prepare(make_proposal())

    changed_after = AFTER.replace(
        "status: string",
        "status?: string",
    )

    second = reviewer.prepare(make_proposal(after_content=changed_after))

    assert first.proposal_id != second.proposal_id


def test_rejects_tampered_after_hash():
    proposal = make_proposal()

    proposal.patches[0].after_sha256 = "0" * 64

    with pytest.raises(
        PatchReviewError,
        match="修改后文件 Hash",
    ):
        PatchReviewer().prepare(proposal)


def test_rejects_tampered_diff():
    proposal = make_proposal()

    proposal.patches[0].unified_diff = "fake diff"

    with pytest.raises(
        PatchReviewError,
        match="Diff",
    ):
        PatchReviewer().prepare(proposal)


def test_approval_binds_exact_proposal():
    reviewer = PatchReviewer()

    service = ApprovalService()

    first = reviewer.prepare(make_proposal())

    changed_after = AFTER.replace(
        "status: string",
        "status?: string",
    )

    second = reviewer.prepare(make_proposal(after_content=changed_after))

    receipt = service.decide(
        review=first,
        decision="approved",
    )

    assert receipt.decision == "approved"

    service.require_approved(
        review=first,
        receipt=receipt,
    )

    with pytest.raises(
        ApprovalError,
        match="不属于当前 Proposal",
    ):
        service.require_approved(
            review=second,
            receipt=receipt,
        )


def test_rejected_proposal_cannot_proceed():
    review = PatchReviewer().prepare(make_proposal())

    service = ApprovalService()

    receipt = service.decide(
        review=review,
        decision="rejected",
        reason="status 参数应当为可选",
    )

    with pytest.raises(
        ApprovalError,
        match="未获批准",
    ):
        service.require_approved(
            review=review,
            receipt=receipt,
        )


def test_same_proposal_cannot_be_decided_twice():
    review = PatchReviewer().prepare(make_proposal())

    service = ApprovalService()

    service.decide(
        review=review,
        decision="rejected",
        reason="需要调整类型定义",
    )

    with pytest.raises(
        ApprovalError,
        match="已经存在审批决定",
    ):
        service.decide(
            review=review,
            decision="approved",
        )
