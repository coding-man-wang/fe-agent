from typing import Literal

from pydantic import BaseModel, Field


class ExecutionPreparation(BaseModel):
    task_id: str

    artifact_id: str

    context_hash: str

    status: Literal["ready_for_review"] = "ready_for_review"

    proposed_paths: list[str] = Field(default_factory=list)

    checked_context_paths: list[str] = Field(default_factory=list)

    checks_passed: list[str] = Field(default_factory=list)
