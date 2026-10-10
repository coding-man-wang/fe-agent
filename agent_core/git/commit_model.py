from pydantic import (
    BaseModel,
    ConfigDict,
)


class GitCommitResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    workspace_id: str

    task_id: str

    proposal_id: str

    branch_name: str

    commit_sha: str

    parent_sha: str

    message: str
