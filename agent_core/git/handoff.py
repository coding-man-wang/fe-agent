import hashlib
import json
from pathlib import Path

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.workspace_execution_model import (
    WorkspaceExecutionResult,
)
from agent_core.git.commit_model import (
    GitCommitResult,
)
from agent_core.git.handoff_model import (
    GitHandoffFile,
    GitHandoffRecord,
)
from agent_core.git.integration_model import (
    GitIntegrationCheckResult,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class GitHandoffError(ValueError):
    pass


class GitHandoffService:

    def __init__(self) -> None:
        self.reviewer = PatchReviewer()

    def prepare(
        self,
        *,
        workspace: GitWorkspaceRecord,
        proposal: FilePatchProposal,
        execution: WorkspaceExecutionResult,
        commit: GitCommitResult,
        integration: GitIntegrationCheckResult,
    ) -> GitHandoffRecord:

        review = self.reviewer.prepare(proposal)

        proposal_id = review.proposal_id

        # --------------------------
        # Identity
        # --------------------------

        if workspace.proposal_id != proposal_id:
            raise GitHandoffError("Workspace Proposal ID 不一致")

        if execution.proposal_id != proposal_id:
            raise GitHandoffError("Execution Proposal ID 不一致")

        if commit.proposal_id != proposal_id:
            raise GitHandoffError("Commit Proposal ID 不一致")

        if not (
            workspace.task_id == proposal.task_id == execution.task_id == commit.task_id
        ):
            raise GitHandoffError("Task ID 不一致")

        if execution.workspace_id != workspace.workspace_id:
            raise GitHandoffError("Execution Workspace ID 不一致")

        if commit.workspace_id != workspace.workspace_id:
            raise GitHandoffError("Commit Workspace ID 不一致")

        # --------------------------
        # Execution / Verification
        # --------------------------

        if execution.verification.status != "succeeded":
            raise GitHandoffError("Verification 未成功")

        if execution.completion.task_id != workspace.task_id:
            raise GitHandoffError("Completion Task ID 不一致")

        if execution.completion.proposal_id != proposal_id:
            raise GitHandoffError("Completion Proposal ID 不一致")

        # --------------------------
        # Commit
        # --------------------------

        if commit.branch_name != workspace.branch_name:
            raise GitHandoffError("Commit Branch 不一致")

        # --------------------------
        # Integration
        # --------------------------

        if integration.agent_commit_sha != commit.commit_sha:
            raise GitHandoffError("Integration 不是针对当前 Commit")

        if integration.base_commit_sha != workspace.base_commit_sha:
            raise GitHandoffError("Integration Base Commit 不一致")

        if integration.status not in {
            "up_to_date",
            "clean",
        }:
            raise GitHandoffError("当前 Commit 存在集成冲突，" "不能生成 Handoff")

        # --------------------------
        # Files
        # --------------------------

        files = [
            GitHandoffFile(
                relative_path=(patch.relative_path),
                before_sha256=(patch.expected_sha256),
                after_sha256=(patch.after_sha256),
            )
            for patch in proposal.patches
        ]

        files.sort(key=lambda item: (item.relative_path))

        handoff_id = self._calculate_handoff_id(
            task_id=workspace.task_id,
            proposal_id=proposal_id,
            workspace_id=(workspace.workspace_id),
            agent_commit_sha=(commit.commit_sha),
            target_branch=(integration.target_branch),
            target_head_sha=(integration.target_head_sha),
        )

        return GitHandoffRecord(
            handoff_id=handoff_id,
            task_id=workspace.task_id,
            proposal_id=proposal_id,
            workspace_id=(workspace.workspace_id),
            source_repository_root=(workspace.repository_root),
            agent_branch=(workspace.branch_name),
            target_branch=(integration.target_branch),
            base_commit_sha=(workspace.base_commit_sha),
            agent_commit_sha=(commit.commit_sha),
            integration_status=(integration.status),
            files=files,
        )

    def _calculate_handoff_id(
        self,
        *,
        task_id: str,
        proposal_id: str,
        workspace_id: str,
        agent_commit_sha: str,
        target_branch: str,
        target_head_sha: str,
    ) -> str:

        payload = {
            "task_id": task_id,
            "proposal_id": proposal_id,
            "workspace_id": workspace_id,
            "agent_commit_sha": (agent_commit_sha),
            "target_branch": (target_branch),
            "target_head_sha": (target_head_sha),
        }

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()
