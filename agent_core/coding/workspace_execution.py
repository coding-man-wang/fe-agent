from pathlib import Path

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.batch_apply import (
    BatchPatchApplyService,
)
from agent_core.coding.completion import (
    TaskCompletionGate,
)
from agent_core.coding.completion_store import (
    TaskCompletionStore,
)
from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
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
from agent_core.coding.verification_policy import (
    VerificationCommandPolicy,
)
from agent_core.coding.verification_store import (
    VerificationStore,
)
from agent_core.coding.verification_workflow import (
    VerificationWorkflow,
)
from agent_core.coding.workspace_execution_model import (
    WorkspaceExecutionResult,
)
from agent_core.git.workspace_binding import (
    WorkspaceBindingService,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class WorkspaceExecutionError(ValueError):
    pass


class WorkspaceExecutionService:
    """
    在经过验证的 Agent Worktree 中执行完整 Coding 流程。

    Proposal
        ↓
    Workspace Binding
        ↓
    Patch Apply
        ↓
    Verification
        ↓
    Completion
    """

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        approval_store: ApprovalStore,
        journal_store: ExecutionJournalStore,
        verification_store: VerificationStore,
        completion_store: TaskCompletionStore,
        verification_policy: VerificationCommandPolicy,
        binding_service: WorkspaceBindingService | None = None,
    ) -> None:

        self.runtime_root = Path(runtime_root).resolve()

        self.approval_store = approval_store

        self.journal_store = journal_store

        self.verification_store = verification_store

        self.completion_store = completion_store

        self.verification_policy = verification_policy

        self.binding_service = binding_service or WorkspaceBindingService()

        self.reviewer = PatchReviewer()

    async def execute(
        self,
        *,
        workspace: GitWorkspaceRecord,
        proposal: FilePatchProposal,
        verification_plan: VerificationPlan,
    ) -> WorkspaceExecutionResult:

        # ----------------------------------
        # 1. 重新计算 Proposal ID
        # ----------------------------------

        review = self.reviewer.prepare(proposal)

        proposal_id = review.proposal_id

        # ----------------------------------
        # 2. 基础 Identity 校验
        # ----------------------------------

        if verification_plan.proposal_id != proposal_id:
            raise WorkspaceExecutionError("Verification Plan " "不属于当前 Proposal")

        if verification_plan.task_id != proposal.task_id:
            raise WorkspaceExecutionError("Verification Plan " "Task ID 不一致")

        # ----------------------------------
        # 3. Workspace Binding
        # ----------------------------------

        binding = await self.binding_service.bind(
            workspace=workspace,
            task_id=proposal.task_id,
            proposal_id=proposal_id,
        )

        execution_root = Path(binding.execution_root).resolve()

        # ----------------------------------
        # 4. Patch Apply
        #
        # 注意：
        # repository_root 现在是 Worktree。
        # ----------------------------------

        apply_service = BatchPatchApplyService(
            repository_root=(execution_root),
            approval_store=(self.approval_store),
            journal_store=(self.journal_store),
        )

        journal = apply_service.apply(proposal=proposal)

        # ----------------------------------
        # 5. Verification Executor
        #
        # 测试命令也在 Worktree 执行。
        # ----------------------------------

        executor = VerificationExecutor(
            repository_root=(execution_root),
            policy=(self.verification_policy),
        )

        verification_workflow = VerificationWorkflow(
            repository_root=(execution_root),
            approval_store=(self.approval_store),
            verification_store=(self.verification_store),
            executor=executor,
        )

        verification = await verification_workflow.run(
            proposal=proposal,
            journal=journal,
            plan=verification_plan,
        )

        # ----------------------------------
        # 6. 验证失败就不能 Completion
        # ----------------------------------

        if verification.status != "succeeded":
            raise WorkspaceExecutionError("Verification 未成功，" "不能完成 Task")

        # ----------------------------------
        # 7. Completion Gate
        #
        # 最终检查的也是 Worktree。
        # ----------------------------------

        completion_gate = TaskCompletionGate(
            repository_root=(execution_root),
            approval_store=(self.approval_store),
            completion_store=(self.completion_store),
        )

        completion = completion_gate.complete(
            proposal=proposal,
            journal=journal,
            verification=(verification),
        )

        return WorkspaceExecutionResult(
            workspace_id=(workspace.workspace_id),
            task_id=proposal.task_id,
            proposal_id=proposal_id,
            execution_root=str(execution_root),
            journal=journal,
            verification=verification,
            completion=completion,
        )
