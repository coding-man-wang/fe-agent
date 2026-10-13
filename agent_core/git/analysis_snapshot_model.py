from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class AnalysisSnapshotRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    snapshot_id: str

    task_id: str

    repository_root: str

    base_commit_sha: str

    snapshot_path: str

    status: Literal[
        "creating",
        "ready",
        "failed",
        "removed",
    ]

    error: str | None = None
