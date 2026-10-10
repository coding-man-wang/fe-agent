from pathlib import Path

from agent_core.coding.approval_store import (
    ApprovalStore,
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
from agent_core.coding.verification_executor import (
    VerificationExecutor,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)
from agent_core.coding.verification_store import (
    VerificationStore,
)


class VerificationWorkflowError(ValueError):
    pass


class VerificationWorkflow:
    """
    将：

    Applied Patch
        ↓
    Repository State Validation
        ↓
    Verification Executor
        ↓
    Durable Verification Record

    串成可靠工作流。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        approval_store: ApprovalStore,
        verification_store: VerificationStore,
        executor: VerificationExecutor,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise VerificationWorkflowError("repository_root 不是目录")

        self.approval_store = approval_store

        self.verification_store = verification_store

        self.executor = executor

        self.reviewer = PatchReviewer()

        self.inspector = RecoveryInspector(self.root)

    def _validate_before_run(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
        plan: VerificationPlan,
    ) -> None:

        review = self.reviewer.prepare(proposal)

        if review.proposal_id != journal.proposal_id:
            raise VerificationWorkflowError("Journal 不属于当前 Proposal")

        if plan.proposal_id != review.proposal_id:
            raise VerificationWorkflowError("Verification Plan " "不属于当前 Proposal")

        if proposal.task_id != journal.task_id or proposal.task_id != plan.task_id:
            raise VerificationWorkflowError("Task ID 不一致")

        # 再确认审批仍然存在。
        self.approval_store.require_approved(review=review)

        if journal.status != "applied":
            raise VerificationWorkflowError("只有 applied 状态才能开始 Verification")

        if not all(item.status == "applied" for item in journal.files):
            raise VerificationWorkflowError("Journal 中仍存在未 applied 文件")

        # 最关键的一步：
        # Journal 说 applied 不够，
        # 当前磁盘也必须仍然是 AFTER。
        assessment = self.inspector.inspect(
            proposal=proposal,
            journal=journal,
        )

        if assessment.requires_manual_recovery:
            raise VerificationWorkflowError(
                "当前仓库状态与 Applied Journal 不一致: " + assessment.reason
            )

        if any(item.current_state != "after" for item in assessment.files):
            raise VerificationWorkflowError("当前仓库并非完整的批准后版本")

    def _load_existing(
        self,
        *,
        proposal_id: str,
    ) -> VerificationRunRecord | None:

        if not self.verification_store.exists(proposal_id):
            return None

        record = self.verification_store.load(proposal_id)

        # 如果上一次进程死在 running，
        # 当前进程不能假装它成功或失败。
        if record.status == "running":
            record = record.model_copy(
                deep=True,
                update={
                    "status": "interrupted",
                    "result": None,
                    "error": (
                        "InterruptedExecution: " "上一次 Verification " "在完成前中断"
                    ),
                },
            )

            self.verification_store.save(record)

        return record

    async def run(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
        plan: VerificationPlan,
    ) -> VerificationRunRecord:

        self._validate_before_run(
            proposal=proposal,
            journal=journal,
            plan=plan,
        )

        existing = self._load_existing(proposal_id=(plan.proposal_id))

        # 已经成功验证，而且当前仓库再次校验
        # 仍然匹配，那么直接返回，避免重复跑测试。
        if existing is not None and existing.status == "succeeded":
            return existing

        attempts = existing.attempts + 1 if existing is not None else 1

        running = VerificationRunRecord(
            proposal_id=(plan.proposal_id),
            task_id=(plan.task_id),
            status="running",
            attempts=attempts,
            result=None,
            error=None,
        )

        self.verification_store.save(running)

        try:

            result = await self.executor.run(plan=plan)

        except Exception as exc:

            failed = running.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "result": None,
                    "error": (f"{type(exc).__name__}: " f"{exc}"),
                },
            )

            self.verification_store.save(failed)

            return failed

        if result.status == "passed":

            completed = running.model_copy(
                deep=True,
                update={
                    "status": "succeeded",
                    "result": result,
                    "error": None,
                },
            )

        else:

            completed = running.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "result": result,
                    "error": None,
                },
            )

        self.verification_store.save(completed)

        return completed
