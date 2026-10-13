from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

ReviewState = Literal[
    "pending",
    "approved",
    "changes_requested",
]


class PullRequestReviewEntry(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    reviewer: str = Field(
        min_length=1,
    )

    state: ReviewState

    raw_state: str


class PullRequestReviewStatus(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    provider: str

    repository: str

    pull_request_number: int

    commit_sha: str

    status: ReviewState

    approvals: int

    changes_requested: int

    reviews: list[PullRequestReviewEntry] = Field(
        default_factory=list,
    )
