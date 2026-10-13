from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class GitHandoffFile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str

    before_sha256: str

    after_sha256: str


class GitHandoffRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    handoff_id: str

    task_id: str

    proposal_id: str

    workspace_id: str

    source_repository_root: str

    agent_branch: str

    target_branch: str

    base_commit_sha: str

    agent_commit_sha: str

    integration_status: Literal[
        "up_to_date",
        "clean",
    ]

    files: list[GitHandoffFile] = Field(min_length=1)
