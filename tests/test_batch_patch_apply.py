import difflib
from pathlib import Path

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.batch_apply import (
    BatchPatchApplyService,
)
from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
    sha256_text,
)

BEFORE_A = "export function getUserList() {}\n"

AFTER_A = "export function getUserList(status: string) {}\n"

BEFORE_B = "export const filters = []\n"

AFTER_B = "export const filters = ['status']\n"


def make_patch(
    *,
    path: str,
    before: str,
    after: str,
) -> ValidatedFilePatch:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )

    return ValidatedFilePatch(
        relative_path=path,
        expected_sha256=(sha256_text(before)),
        after_sha256=(sha256_text(after)),
        before_content=before,
        after_content=after,
        explanation="test change",
        unified_diff=diff,
    )


def make_proposal():
    return FilePatchProposal(
        task_id="TASK-003",
        artifact_id="a" * 64,
        context_hash="b" * 64,
        patches=[
            make_patch(
                path="src/api/user.ts",
                before=BEFORE_A,
                after=AFTER_A,
            ),
            make_patch(
                path="src/views/UserList.ts",
                before=BEFORE_B,
                after=AFTER_B,
            ),
        ],
    )


def test_batch_apply_success(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a = tmp_path / "src/api/user.ts"

    path_b = tmp_path / "src/views/UserList.ts"

    path_a.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_b.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_a.write_bytes(BEFORE_A.encode("utf-8"))

    path_b.write_bytes(BEFORE_B.encode("utf-8"))

    approval_store = ApprovalStore(tmp_path)

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    journal_store = ExecutionJournalStore(tmp_path)

    service = BatchPatchApplyService(
        repository_root=tmp_path,
        approval_store=(approval_store),
        journal_store=(journal_store),
    )

    journal = service.apply(proposal=proposal)

    assert journal.status == "applied"

    assert all(item.status == "applied" for item in journal.files)

    assert path_a.read_text(encoding="utf-8") == AFTER_A

    assert path_b.read_text(encoding="utf-8") == AFTER_B


import pytest

from agent_core.coding.batch_apply import (
    BatchPatchApplyError,
)


def test_stage_failure_has_no_side_effects(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a = tmp_path / "src/api/user.ts"

    path_b = tmp_path / "src/views/UserList.ts"

    path_a.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_b.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A 是合法的 Before 版本。
    path_a.write_bytes(BEFORE_A.encode("utf-8"))

    # B 故意制造成与 Proposal 不同的版本。
    changed_b = "export const filters = ['other']\n"

    path_b.write_bytes(changed_b.encode("utf-8"))

    # 先确认测试前提成立：
    # B 当前内容必须和 Proposal 里的 expected hash 不一致。
    assert sha256_text(changed_b) != proposal.patches[1].expected_sha256

    approval_store = ApprovalStore(tmp_path)

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    journal_store = ExecutionJournalStore(tmp_path)

    service = BatchPatchApplyService(
        repository_root=tmp_path,
        approval_store=approval_store,
        journal_store=journal_store,
    )

    with pytest.raises(
        BatchPatchApplyError,
        match="版本已变化",
    ):
        service.apply(proposal=proposal)

    # 因为是 Stage All，
    # B 的预检查失败时 A 不能产生正式副作用。
    assert path_a.read_bytes() == (BEFORE_A.encode("utf-8"))

    assert path_b.read_bytes() == (changed_b.encode("utf-8"))

    # Stage 阶段失败，正式执行 Journal
    # 也不应该已经建立。
    assert journal_store.exists(review.proposal_id) is False


def test_same_proposal_cannot_apply_twice(
    tmp_path: Path,
):
    proposal = make_proposal()

    for path, content in [
        (
            "src/api/user.ts",
            BEFORE_A,
        ),
        (
            "src/views/UserList.ts",
            BEFORE_B,
        ),
    ]:
        target = tmp_path / path

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_bytes(content.encode("utf-8"))

    approval_store = ApprovalStore(tmp_path)

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    journal_store = ExecutionJournalStore(tmp_path)

    service = BatchPatchApplyService(
        repository_root=tmp_path,
        approval_store=(approval_store),
        journal_store=(journal_store),
    )

    service.apply(proposal=proposal)

    with pytest.raises(
        BatchPatchApplyError,
        match="已存在 Execution Journal",
    ):
        service.apply(proposal=proposal)


class FailingSecondCommitService(BatchPatchApplyService):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.commit_count = 0

    def _commit_staged_file(
        self,
        staged,
    ) -> None:
        self.commit_count += 1

        if self.commit_count == 2:
            raise RuntimeError("模拟第二个文件写入失败")

        return super()._commit_staged_file(staged)


def test_partial_apply_is_recorded(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a = tmp_path / "src/api/user.ts"

    path_b = tmp_path / "src/views/UserList.ts"

    path_a.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_b.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_a.write_bytes(BEFORE_A.encode("utf-8"))

    path_b.write_bytes(BEFORE_B.encode("utf-8"))

    approval_store = ApprovalStore(tmp_path)

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    journal_store = ExecutionJournalStore(tmp_path)

    service = FailingSecondCommitService(
        repository_root=tmp_path,
        approval_store=(approval_store),
        journal_store=(journal_store),
    )

    with pytest.raises(
        RuntimeError,
        match="第二个文件",
    ):
        service.apply(proposal=proposal)

    journal = journal_store.load(review.proposal_id)

    assert journal.status == "partially_applied"

    assert journal.files[0].status == "applied"

    assert journal.files[1].status == "pending"

    assert path_a.read_text(encoding="utf-8") == AFTER_A

    assert path_b.read_text(encoding="utf-8") == BEFORE_B
