from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

ReviewFeedbackKind = Literal[
    "inline_comment",
    "conversation_comment",
]


class ReviewFeedbackItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    feedback_id: str

    kind: ReviewFeedbackKind

    reviewer: str = Field(
        min_length=1,
    )

    body: str = Field(
        min_length=1,
    )

    relative_path: str | None = None

    line: int | None = None

    commit_sha: str | None = None

    external_id: str

    url: str | None = None


class ReviewFeedbackArtifact(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    artifact_id: str

    pull_request_provider: str

    repository: str

    pull_request_number: int

    task_id: str

    proposal_id: str

    commit_sha: str

    items: list[ReviewFeedbackItem] = Field(
        default_factory=list,
    )
