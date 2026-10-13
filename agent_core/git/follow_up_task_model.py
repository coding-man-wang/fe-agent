from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.sdd.model import (
    TaskDefinition,
)


class FollowUpTaskLink(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    feedback_id: str

    follow_up_task_id: str


class FollowUpTaskArtifact(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    artifact_id: str

    source_analysis_id: str

    source_feedback_artifact_id: str

    parent_task_id: str

    commit_sha: str

    tasks: list[TaskDefinition] = Field(
        default_factory=list,
    )

    links: list[FollowUpTaskLink] = Field(
        default_factory=list,
    )
