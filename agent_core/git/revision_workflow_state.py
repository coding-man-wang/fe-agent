from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.git.ci_model import (
    PullRequestCIStatus,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_model import (
    PullRequestReviewStatus,
)
from agent_core.git.revision_task_model import (
    PreparedRevisionTask,
)

RevisionWorkflowStatus = Literal[
    "preparing",
    "awaiting_approval",
    "executing",
    "waiting_ci",
    "waiting_review",
    "changes_requested",
    "ready_for_human_merge",
    "failed",
]


RevisionTaskStage = Literal[
    "pending",
    "awaiting_approval",
    "succeeded",
    "failed",
]


class DurableRevisionTaskState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: str

    stage: RevisionTaskStage = "pending"

    base_commit_sha: str

    prepared: PreparedRevisionTask | None = None

    commit_sha: str | None = None

    error: str | None = None


class DurableRevisionWorkflowState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    workflow_id: str

    follow_up_artifact_id: str

    repository: str

    pull_request_number: int

    pull_request: PullRequestResult

    status: RevisionWorkflowStatus

    current_task_index: int = 0

    tasks: list[DurableRevisionTaskState] = Field(default_factory=list)

    latest_ci: PullRequestCIStatus | None = None

    latest_review: PullRequestReviewStatus | None = None

    error: str | None = None
