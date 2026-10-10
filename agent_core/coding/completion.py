from pathlib import Path

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.completion_store import (
    TaskCompletionStore,
)
from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.recovery import (
    RecoveryInspector,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.task_completion_model import (
    TaskCompletionRecord,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)


class TaskCompletionError(ValueError):
    pass


class TaskCompletionGate:
    """
    Coding Task 最终完成门禁。

    只有确定性条件全部满足，
    才能产生 completed。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        approval_store: ApprovalStore,
        completion_store: TaskCompletionStore,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise TaskCompletionError("repository_root 不是目录")

        self.approval_store = approval_store

        self.completion_store = completion_store

        self.reviewer = PatchReviewer()

        self.inspector = RecoveryInspector(self.root)

    def complete(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
        verification: VerificationRunRecord,
    ) -> TaskCompletionRecord:

        review = self.reviewer.prepare(proposal)

        proposal_id = review.proposal_id

        # ----------------------------
        # Identity
        # ----------------------------

        if journal.proposal_id != proposal_id:
            raise TaskCompletionError("Execution Journal " "不属于当前 Proposal")

        if verification.proposal_id != proposal_id:
            raise TaskCompletionError("Verification " "不属于当前 Proposal")

        if not (proposal.task_id == journal.task_id == verification.task_id):
            raise TaskCompletionError("Task ID 不一致")

        # ----------------------------
        # Approval
        # ----------------------------

        self.approval_store.require_approved(review=review)

        # ----------------------------
        # Execution
        # ----------------------------

        if journal.status != "applied":
            raise TaskCompletionError("Patch 尚未完整 Apply")

        if not all(item.status == "applied" for item in journal.files):
            raise TaskCompletionError("仍存在未完成的文件修改")

        # ----------------------------
        # Verification
        # ----------------------------

        if verification.status != "succeeded":
            raise TaskCompletionError("Verification Workflow " "尚未成功")

        if verification.result is None:
            raise TaskCompletionError("Verification Result 缺失")

        if verification.result.status != "passed":
            raise TaskCompletionError("Verification 未通过")

        if verification.result.proposal_id != proposal_id:
            raise TaskCompletionError("Verification Result " "Proposal ID 不一致")

        if verification.result.task_id != proposal.task_id:
            raise TaskCompletionError("Verification Result " "Task ID 不一致")

        # ----------------------------
        # Repository Final State
        # ----------------------------

        assessment = self.inspector.inspect(
            proposal=proposal,
            journal=journal,
        )

        if assessment.requires_manual_recovery:
            raise TaskCompletionError("当前仓库状态已经变化: " + assessment.reason)

        if any(item.current_state != "after" for item in assessment.files):
            raise TaskCompletionError("当前仓库不是已验证版本")

        # ----------------------------
        # Idempotent Completion
        # ----------------------------

        if self.completion_store.exists(proposal_id):

            existing = self.completion_store.load(proposal_id)

            if existing.task_id != proposal.task_id:
                raise TaskCompletionError(
                    "已有 Completion Record " "与当前 Task 不一致"
                )

            return existing

        record = TaskCompletionRecord(
            task_id=proposal.task_id,
            proposal_id=proposal_id,
        )

        self.completion_store.save(record)

        return record
