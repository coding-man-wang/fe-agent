from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_core.git.ci_model import (
    PullRequestCIStatus,
)

CITrackingOutcome = Literal[
    "succeeded",
    "failed",
    "no_checks",
    "timeout",
]


class CITrackingResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    outcome: CITrackingOutcome

    attempts: int

    final_status: PullRequestCIStatus
