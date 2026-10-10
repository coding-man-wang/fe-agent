from pathlib import Path

import pytest

from agent_core.coding.approval import (
    ApprovalError,
)
from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from tests.test_patch_review import (
    make_proposal,
)


def test_approval_survives_restart(
    tmp_path: Path,
):
    review = PatchReviewer().prepare(make_proposal())

    first_store = ApprovalStore(tmp_path)

    first_store.record_decision(
        review=review,
        decision="approved",
    )

    # 新 Store 实例模拟进程重启。
    second_store = ApprovalStore(tmp_path)

    receipt = second_store.require_approved(review=review)

    assert receipt.decision == "approved"

    assert receipt.proposal_id == review.proposal_id


def test_rejected_approval_is_not_accepted(
    tmp_path: Path,
):
    review = PatchReviewer().prepare(make_proposal())

    store = ApprovalStore(tmp_path)

    store.record_decision(
        review=review,
        decision="rejected",
        reason="需要修改参数类型",
    )

    with pytest.raises(
        ApprovalError,
        match="未获批准",
    ):
        store.require_approved(review=review)


def test_missing_approval_is_rejected(
    tmp_path: Path,
):
    review = PatchReviewer().prepare(make_proposal())

    store = ApprovalStore(tmp_path)

    with pytest.raises(
        ApprovalError,
        match="找不到审批记录",
    ):
        store.require_approved(review=review)
