from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

MergeReadinessStatus = Literal[
    "blocked_by_ci",
    "waiting_for_review",
    "ready_for_human_merge",
    "blocked",
]


class MergeReadinessResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: MergeReadinessStatus

    reasons: list[str] = Field(
        default_factory=list,
    )

    ci_succeeded: bool

    review_approved: bool
