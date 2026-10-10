from typing import Literal

from pydantic import BaseModel, Field

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)

AnalysisStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
]


class TaskAnalysisRecord(BaseModel):
    task_id: str

    status: AnalysisStatus = "pending"

    attempts: int = Field(
        default=0,
        ge=0,
    )

    result: ReadOnlyCodingResult | None = None

    error: str | None = None

    artifact_id: str | None = None

    context_hash: str | None = None
