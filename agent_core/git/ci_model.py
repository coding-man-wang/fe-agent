from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

CICheckState = Literal[
    "pending",
    "succeeded",
    "failed",
]


CIOverallStatus = Literal[
    "no_checks",
    "pending",
    "succeeded",
    "failed",
]


class CICheckResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    source: Literal[
        "commit_status",
        "check_run",
    ]

    name: str = Field(
        min_length=1,
    )

    state: CICheckState

    raw_status: str

    raw_conclusion: str | None = None

    url: str | None = None


class PullRequestCIStatus(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    provider: str

    repository: str

    pull_request_number: int

    commit_sha: str

    status: CIOverallStatus

    checks: list[CICheckResult] = Field(
        default_factory=list,
    )
