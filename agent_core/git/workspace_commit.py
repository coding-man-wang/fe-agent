from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.workspace_execution_model import (
    WorkspaceExecutionResult,
)
from agent_core.git.change_set import (
    GitChangeSetInspector,
)
from agent_core.git.change_set_validator import (
    ChangeSetValidator,
)
from agent_core.git.commit_model import (
    GitCommitResult,
)
from agent_core.git.commit_service import (
    AgentGitCommitService,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class WorkspaceCommitError(ValueError):
    pass


class WorkspaceCommitWorkflow:

    def __init__(
        self,
        *,
        commit_service: AgentGitCommitService | None = None,
        change_set_inspector: GitChangeSetInspector | None = None,
        change_set_validator: ChangeSetValidator | None = None,
    ) -> None:

        self.commit_service = commit_service or AgentGitCommitService()

        self.change_set_inspector = change_set_inspector or GitChangeSetInspector()

        self.change_set_validator = change_set_validator or ChangeSetValidator()

        self.reviewer = PatchReviewer()

    async def commit(
        self,
        *,
        workspace: GitWorkspaceRecord,
        execution: WorkspaceExecutionResult,
        proposal: FilePatchProposal,
        message: str,
    ) -> GitCommitResult:

        review = self.reviewer.prepare(proposal)

        proposal_id = review.proposal_id

        # --------------------------
        # Identity
        # --------------------------

        if execution.workspace_id != workspace.workspace_id:
            raise WorkspaceCommitError("Execution 不属于当前 Workspace")

        if execution.task_id != workspace.task_id:
            raise WorkspaceCommitError("Execution Task ID 不一致")

        if execution.proposal_id != workspace.proposal_id:
            raise WorkspaceCommitError("Execution Proposal ID 不一致")

        if proposal.task_id != workspace.task_id:
            raise WorkspaceCommitError("Proposal Task ID 不一致")

        if proposal_id != workspace.proposal_id:
            raise WorkspaceCommitError("Proposal ID 不一致")

        # --------------------------
        # Execution result
        # --------------------------

        if execution.verification.status != "succeeded":
            raise WorkspaceCommitError("Verification 未成功")

        if execution.completion.task_id != workspace.task_id:
            raise WorkspaceCommitError("Completion Task ID 不一致")

        if execution.completion.proposal_id != workspace.proposal_id:
            raise WorkspaceCommitError("Completion Proposal ID 不一致")

        # --------------------------
        # ChangeSet
        # --------------------------

        change_set = await self.change_set_inspector.inspect(workspace=workspace)

        validation = self.change_set_validator.validate(
            proposal=proposal,
            change_set=change_set,
        )

        if not validation.allowed:

            raise WorkspaceCommitError(
                "Git ChangeSet "
                "超出 Proposal 批准范围: "
                f"unexpected="
                f"{validation.unexpected_paths}, "
                f"missing="
                f"{validation.missing_paths}, "
                f"invalid_operations="
                f"{validation.invalid_operations}"
            )

        # --------------------------
        # Commit
        # --------------------------

        return await self.commit_service.commit(
            workspace=workspace,
            message=message,
        )
