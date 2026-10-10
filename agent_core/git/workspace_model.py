from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

GitWorkspaceStatus = Literal[
    "creating",
    "ready",
    "failed",
    "removed",
]


class GitWorkspaceRecord(BaseModel):
    """
    一个 Agent Git Workspace 的持久化记录。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    workspace_id: str

    task_id: str

    proposal_id: str

    repository_root: str

    base_commit_sha: str

    branch_name: str

    worktree_path: str

    status: GitWorkspaceStatus

    error: str | None = None
