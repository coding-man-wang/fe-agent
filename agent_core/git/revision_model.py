from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.git.pull_request_model import (
    PullRequestResult,
)

RevisionTaskStatus = Literal[
    "succeeded",
    "failed",
]


class RevisionTaskResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: str

    status: RevisionTaskStatus

    base_commit_sha: str

    commit_sha: str | None = None

    error: str | None = None


class RevisionWorkflowResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    revision_id: str

    source_follow_up_artifact_id: str

    pull_request_number: int

    branch_name: str

    initial_commit_sha: str

    final_commit_sha: str

    status: Literal[
        "succeeded",
        "failed",
    ]

    tasks: list[RevisionTaskResult] = Field(
        default_factory=list,
    )

    updated_pull_request: PullRequestResult | None = None
