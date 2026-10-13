from pydantic import (
    BaseModel,
    ConfigDict,
)


class GitPushResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    handoff_id: str

    task_id: str

    proposal_id: str

    remote_name: str

    branch_name: str

    commit_sha: str

    remote_ref: str
