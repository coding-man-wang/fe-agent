from pathlib import Path

import pytest

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.resume import (
    PatchResumeService,
    ResumeError,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from tests.test_recovery import (
    AFTER_A,
    AFTER_B,
    BEFORE_B,
    create_repository,
    make_partial_journal,
    make_proposal,
)


def create_resume_environment(
    root: Path,
):
    proposal = make_proposal()

    path_a, path_b = create_repository(
        root,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    journal_store = ExecutionJournalStore(root)

    journal_store.create(journal)

    approval_store = ApprovalStore(root)

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    service = PatchResumeService(
        repository_root=root,
        approval_store=(approval_store),
        journal_store=(journal_store),
    )

    return (
        proposal,
        journal,
        journal_store,
        service,
        path_a,
        path_b,
    )


def test_resume_pending_files(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        journal_store,
        service,
        path_a,
        path_b,
    ) = create_resume_environment(tmp_path)

    result = service.resume(
        proposal=proposal,
        journal=journal,
    )

    assert result.status == "applied"

    assert all(item.status == "applied" for item in result.files)

    # A 本来就是 AFTER。
    assert path_a.read_bytes() == (AFTER_A.encode("utf-8"))

    # B 被 Resume 修改。
    assert path_b.read_bytes() == (AFTER_B.encode("utf-8"))

    restored = journal_store.load(journal.proposal_id)

    assert restored.status == "applied"


def test_resume_rejected_if_applied_file_diverged(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        _,
        service,
        path_a,
        path_b,
    ) = create_resume_environment(tmp_path)

    path_a.write_bytes(b"human change\n")

    with pytest.raises(
        ResumeError,
        match="不允许 Resume",
    ):
        service.resume(
            proposal=proposal,
            journal=journal,
        )

    # B 不应该产生新的副作用。
    assert path_b.read_bytes() == (BEFORE_B.encode("utf-8"))


def test_resume_rejected_if_pending_file_changed(
    tmp_path: Path,
):
    (
        proposal,
        journal,
        _,
        service,
        _,
        path_b,
    ) = create_resume_environment(tmp_path)

    path_b.write_bytes(b"external change\n")

    with pytest.raises(
        ResumeError,
        match="不允许 Resume",
    ):
        service.resume(
            proposal=proposal,
            journal=journal,
        )


def test_resume_requires_approval(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    journal_store = ExecutionJournalStore(tmp_path)

    journal_store.create(journal)

    # 没有 record approval。
    service = PatchResumeService(
        repository_root=tmp_path,
        approval_store=(ApprovalStore(tmp_path)),
        journal_store=(journal_store),
    )

    with pytest.raises(
        ValueError,
    ):
        service.resume(
            proposal=proposal,
            journal=journal,
        )
