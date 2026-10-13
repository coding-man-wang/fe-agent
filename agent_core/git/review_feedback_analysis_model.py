from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

ReviewFeedbackIntent = Literal[
    "code_change",
    "discussion",
    "clarification_needed",
    "non_actionable",
]


class ReviewFeedbackAnalysisItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    feedback_id: str

    intent: ReviewFeedbackIntent

    summary: str = Field(
        min_length=1,
    )

    reasoning: str = Field(
        min_length=1,
    )

    requires_code_change: bool

    requires_reply: bool

    target_hints: list[str] = Field(
        default_factory=list,
    )


class ReviewFeedbackAnalysisOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    source_artifact_id: str

    items: list[ReviewFeedbackAnalysisItem] = Field(
        default_factory=list,
    )


class ReviewFeedbackAnalysisArtifact(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    analysis_id: str

    source_artifact_id: str

    task_id: str

    proposal_id: str

    commit_sha: str

    items: list[ReviewFeedbackAnalysisItem] = Field(
        default_factory=list,
    )
