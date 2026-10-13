from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

IntegrationStatus = Literal[
    "up_to_date",
    "clean",
    "conflict",
]


class GitIntegrationCheckResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    target_branch: str

    base_commit_sha: str

    target_head_sha: str

    agent_commit_sha: str

    status: IntegrationStatus

    target_advanced: bool

    conflicting_files: list[str]
