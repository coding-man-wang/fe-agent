from pydantic import (
    BaseModel,
    ConfigDict,
)


class WorkspaceExecutionContext(BaseModel):
    """
    已经验证过，可以真正用于执行代码修改的 Workspace。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    workspace_id: str

    task_id: str

    proposal_id: str

    source_repository_root: str

    execution_root: str

    base_commit_sha: str

    branch_name: str
