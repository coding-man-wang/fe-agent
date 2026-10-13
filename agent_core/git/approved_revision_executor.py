from dataclasses import dataclass

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.coding.workspace_execution import (
    WorkspaceExecutionService,
)
from agent_core.git.commit_service import (
    AgentGitCommitService,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.revision_task_model import (
    PreparedRevisionTask,
)
from agent_core.git.workspace import (
    AgentGitWorkspaceManager,
)
from agent_core.sdd.model import (
    TaskDefinition,
)

@dataclass(frozen=True)
class ApprovedRevisionExecutionResult:

    task_id: str

    base_commit_sha: str

    commit_sha: str

class ApprovedRevisionExecutionError(
    ValueError
):
    pass


class ApprovedRevisionTaskExecutor:

    def __init__(
        self,
        *,
        workspace_manager: (
            AgentGitWorkspaceManager
        ),
        workspace_execution: (
            WorkspaceExecutionService
        ),
        commit_service: (
            AgentGitCommitService
        ),
        approval_store: ApprovalStore,
    ) -> None:

        self.workspace_manager = (
            workspace_manager
        )
        self.workspace_execution = (
            workspace_execution
        )
        self.commit_service = (
            commit_service
        )
        self.approval_store = (
            approval_store
        )

        self.reviewer = (
            PatchReviewer()
        )

    async def execute(
        self,
        *,
        task: TaskDefinition,
        prepared: PreparedRevisionTask,
        pull_request: PullRequestResult,
        verification_plan: VerificationPlan,
    ) -> ApprovedRevisionExecutionResult:

        self._validate_inputs(
            task=task,
            prepared=prepared,
            pull_request=(
                pull_request
            ),
        )

        proposal = prepared.proposal

        review = (
            self.reviewer.prepare(
                proposal
            )
        )

        if (
            review.proposal_id
            != prepared.proposal_id
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Prepared Proposal ID "
                    "校验失败"
                )
            )

        # ----------------------------------
        # Approval 必须已经存在。
        #
        # 注意：
        # Executor 不创建 Approval。
        # ----------------------------------

        receipt = (
            self.approval_store.load(
                prepared.proposal_id
            )
        )

        if (
            receipt.decision
            != "approved"
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Revision Proposal "
                    "尚未获得批准"
                )
            )

        if (
            receipt.task_id
            != task.task_id
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Approval Task ID 不一致"
                )
            )

        # ----------------------------------
        # Execution Workspace
        # ----------------------------------

        workspace = (
            await self.workspace_manager
            .create(
                task_id=task.task_id,
                proposal_id=(
                    prepared.proposal_id
                ),
                base_commit_sha=(
                    prepared.base_commit_sha
                ),
            )
        )

        execution = (
            await self.workspace_execution
            .execute(
                workspace=workspace,
                proposal=proposal,
                verification_plan=(
                    verification_plan
                ),
            )
        )

        # ----------------------------------
        # Commit
        # ----------------------------------

        commit = (
            await self.commit_service.commit(
                workspace=workspace,
                message=(
                    "fix(review): "
                    f"{task.title}"
                ),
            )
        )

        if (
            commit.parent_commit_sha
            != prepared.base_commit_sha
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Revision Commit "
                    "不是基于预期 Commit"
                )
            )

        return (
            ApprovedRevisionExecutionResult(
                task_id=task.task_id,
                base_commit_sha=(
                    prepared.base_commit_sha
                ),
                commit_sha=(
                    commit.commit_sha
                ),
            )
        )

    def _validate_inputs(
        self,
        *,
        task: TaskDefinition,
        prepared: PreparedRevisionTask,
        pull_request: PullRequestResult,
    ) -> None:

        if (
            task.task_id
            != prepared.task_id
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Prepared Revision "
                    "不属于当前 Task"
                )
            )

        if (
            prepared.base_commit_sha
            != pull_request.commit_sha
        ):
            raise (
                ApprovedRevisionExecutionError(
                    "Revision Base Commit "
                    "与当前 PR Head 不一致"
                )
            )