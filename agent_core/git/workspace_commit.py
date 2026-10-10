from agent_core.coding.workspace_execution_model import (
    WorkspaceExecutionResult,
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
    ) -> None:

        self.commit_service = commit_service or AgentGitCommitService()

    async def commit(
        self,
        *,
        workspace: GitWorkspaceRecord,
        execution: WorkspaceExecutionResult,
        message: str,
    ) -> GitCommitResult:

        if execution.workspace_id != workspace.workspace_id:
            raise WorkspaceCommitError("Execution 不属于当前 Workspace")

        if execution.task_id != workspace.task_id:
            raise WorkspaceCommitError("Execution Task ID 不一致")

        if execution.proposal_id != workspace.proposal_id:
            raise WorkspaceCommitError("Execution Proposal ID 不一致")

        if execution.verification.status != "succeeded":
            raise WorkspaceCommitError("Verification 未成功")

        if execution.completion.task_id != workspace.task_id:
            raise WorkspaceCommitError("Completion Task ID 不一致")

        if execution.completion.proposal_id != workspace.proposal_id:
            raise WorkspaceCommitError("Completion Proposal ID 不一致")

        return await self.commit_service.commit(
            workspace=workspace,
            message=message,
        )
