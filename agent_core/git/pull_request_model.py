from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class PullRequestRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    handoff_id: str

    task_id: str

    proposal_id: str

    remote_name: str

    head_branch: str

    base_branch: str

    commit_sha: str

    title: str = Field(
        min_length=1,
    )

    body: str = Field(
        min_length=1,
    )


class PullRequestResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    handoff_id: str

    task_id: str

    proposal_id: str

    provider: str

    repository: str

    number: int

    url: str

    head_branch: str

    base_branch: str

    commit_sha: str
