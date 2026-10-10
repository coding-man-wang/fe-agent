import difflib
from pathlib import Path

import pytest

from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.execution_journal_model import (
    FileExecutionRecord,
    PatchExecutionJournal,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.recovery import (
    RecoveryError,
    RecoveryInspector,
    SafeRollbackService,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
    sha256_text,
)

BEFORE_A = "export function getUserList() {}\n"

AFTER_A = "export function getUserList(" "status: string) {}\n"

BEFORE_B = "export const filters = []\n"

AFTER_B = "export const filters = " "['status']\n"


def make_patch(
    *,
    path: str,
    before: str,
    after: str,
) -> ValidatedFilePatch:
    unified_diff = "".join(
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
        explanation="测试修改",
        unified_diff=(unified_diff),
    )


def make_proposal() -> FilePatchProposal:
    return FilePatchProposal(
        task_id="TASK-003",
        artifact_id="a" * 64,
        context_hash="b" * 64,
        patches=[
            make_patch(
                path=("src/api/user.ts"),
                before=BEFORE_A,
                after=AFTER_A,
            ),
            make_patch(
                path=("src/views/UserList.ts"),
                before=BEFORE_B,
                after=AFTER_B,
            ),
        ],
    )


def create_repository(
    root: Path,
    *,
    content_a: str,
    content_b: str,
) -> tuple[
    Path,
    Path,
]:
    path_a = root / "src/api/user.ts"

    path_b = root / "src/views/UserList.ts"

    path_a.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_b.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 使用 bytes 写入，避免 Windows 自动将
    # \n 转换成 \r\n，确保磁盘字节与
    # Proposal Hash 使用的内容完全一致。
    path_a.write_bytes(content_a.encode("utf-8"))

    path_b.write_bytes(content_b.encode("utf-8"))

    return (
        path_a,
        path_b,
    )


def make_partial_journal(
    proposal: FilePatchProposal,
) -> PatchExecutionJournal:
    review = PatchReviewer().prepare(proposal)

    return PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        status="partially_applied",
        files=[
            FileExecutionRecord(
                relative_path=(proposal.patches[0].relative_path),
                expected_sha256=(proposal.patches[0].expected_sha256),
                after_sha256=(proposal.patches[0].after_sha256),
                status="applied",
            ),
            FileExecutionRecord(
                relative_path=(proposal.patches[1].relative_path),
                expected_sha256=(proposal.patches[1].expected_sha256),
                after_sha256=(proposal.patches[1].after_sha256),
                status="pending",
            ),
        ],
    )


def test_partial_apply_can_resume_and_rollback(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    assessment = RecoveryInspector(tmp_path).inspect(
        proposal=proposal,
        journal=journal,
    )

    assert assessment.can_resume is True

    assert assessment.can_rollback is True

    assert assessment.requires_manual_recovery is False

    states = {item.relative_path: item.current_state for item in assessment.files}

    assert states["src/api/user.ts"] == "after"

    assert states["src/views/UserList.ts"] == "before"


def test_external_change_requires_manual_recovery(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a, _ = create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    # Agent 修改之后，
    # 模拟其他开发者继续修改 A。
    path_a.write_text(
        ("export function " "externalChange() {}\n"),
        encoding="utf-8",
    )

    assessment = RecoveryInspector(tmp_path).inspect(
        proposal=proposal,
        journal=journal,
    )

    assert assessment.can_resume is False

    assert assessment.can_rollback is False

    assert assessment.requires_manual_recovery is True


def test_safe_rollback_restores_applied_file(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a, path_b = create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    journal_store = ExecutionJournalStore(tmp_path)

    journal_store.create(journal)

    service = SafeRollbackService(
        repository_root=(tmp_path),
        journal_store=(journal_store),
    )

    result = service.rollback(
        proposal=proposal,
        journal=journal,
    )

    assert path_a.read_text(encoding="utf-8") == BEFORE_A

    assert path_b.read_text(encoding="utf-8") == BEFORE_B

    restored_journal = journal_store.load(journal.proposal_id)

    assert restored_journal.status == "rolled_back"

    statuses = {item.relative_path: item.status for item in restored_journal.files}

    assert statuses["src/api/user.ts"] == "rolled_back"

    # B 原本没有 Apply，
    # 所以仍然是 pending。
    assert statuses["src/views/UserList.ts"] == "pending"

    assert result.can_resume is False

    assert result.can_rollback is False

    assert result.requires_manual_recovery is False


def test_rollback_rejected_after_external_change(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a, _ = create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = make_partial_journal(proposal)

    journal_store = ExecutionJournalStore(tmp_path)

    journal_store.create(journal)

    # 模拟 Agent Apply 后，
    # 其他开发者再次修改。
    path_a.write_text(
        "human change\n",
        encoding="utf-8",
    )

    service = SafeRollbackService(
        repository_root=(tmp_path),
        journal_store=(journal_store),
    )

    with pytest.raises(
        RecoveryError,
        match=("不允许自动 Rollback"),
    ):
        service.rollback(
            proposal=proposal,
            journal=journal,
        )

    # 不能覆盖用户的新修改。
    assert path_a.read_text(encoding="utf-8") == "human change\n"


def test_applied_batch_can_be_rolled_back(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a, path_b = create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    review = PatchReviewer().prepare(proposal)

    journal = PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        status="applied",
        files=[
            FileExecutionRecord(
                relative_path=(proposal.patches[0].relative_path),
                expected_sha256=(proposal.patches[0].expected_sha256),
                after_sha256=(proposal.patches[0].after_sha256),
                status="applied",
            ),
            FileExecutionRecord(
                relative_path=(proposal.patches[1].relative_path),
                expected_sha256=(proposal.patches[1].expected_sha256),
                after_sha256=(proposal.patches[1].after_sha256),
                status="applied",
            ),
        ],
    )

    inspector = RecoveryInspector(tmp_path)

    before = inspector.inspect(
        proposal=proposal,
        journal=journal,
    )

    assert before.can_resume is False

    assert before.can_rollback is True

    journal_store = ExecutionJournalStore(tmp_path)

    journal_store.create(journal)

    SafeRollbackService(
        repository_root=(tmp_path),
        journal_store=(journal_store),
    ).rollback(
        proposal=proposal,
        journal=journal,
    )

    assert path_a.read_text(encoding="utf-8") == BEFORE_A

    assert path_b.read_text(encoding="utf-8") == BEFORE_B


def test_failed_without_side_effects_needs_no_recovery(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=BEFORE_A,
        content_b=BEFORE_B,
    )

    review = PatchReviewer().prepare(proposal)

    journal = PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        status="failed",
        files=[
            FileExecutionRecord(
                relative_path=(patch.relative_path),
                expected_sha256=(patch.expected_sha256),
                after_sha256=(patch.after_sha256),
                status="pending",
            )
            for patch in proposal.patches
        ],
        error="模拟执行失败",
    )

    assessment = RecoveryInspector(tmp_path).inspect(
        proposal=proposal,
        journal=journal,
    )

    assert assessment.can_resume is False

    assert assessment.can_rollback is False

    assert assessment.requires_manual_recovery is False


def test_rolled_back_state_is_consistent(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=BEFORE_A,
        content_b=BEFORE_B,
    )

    review = PatchReviewer().prepare(proposal)

    journal = PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=proposal.task_id,
        status="rolled_back",
        files=[
            FileExecutionRecord(
                relative_path=(proposal.patches[0].relative_path),
                expected_sha256=(proposal.patches[0].expected_sha256),
                after_sha256=(proposal.patches[0].after_sha256),
                status="rolled_back",
            ),
            FileExecutionRecord(
                relative_path=(proposal.patches[1].relative_path),
                expected_sha256=(proposal.patches[1].expected_sha256),
                after_sha256=(proposal.patches[1].after_sha256),
                status="pending",
            ),
        ],
    )

    assessment = RecoveryInspector(tmp_path).inspect(
        proposal=proposal,
        journal=journal,
    )

    assert assessment.requires_manual_recovery is False

    assert assessment.can_resume is False

    assert assessment.can_rollback is False
