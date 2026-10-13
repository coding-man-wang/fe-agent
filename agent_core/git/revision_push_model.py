from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class RevisionPushResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    provider: str

    repository: str

    pull_request_number: int

    remote_name: str

    branch_name: str

    expected_base_commit_sha: str

    pushed_commit_sha: str

    status: Literal[
        "pushed",
        "already_applied",
    ]
