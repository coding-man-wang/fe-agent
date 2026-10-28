from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ReviewFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str

    action: Literal[
        "modify",
        "create",
    ] = "modify"

    expected_sha256: str | None = None

    after_sha256: str

    explanation: str

    unified_diff: str


class ReviewPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    proposal_id: str

    task_id: str

    artifact_id: str

    context_hash: str

    files: list[ReviewFile] = Field(min_length=1)


class ApprovalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    proposal_id: str

    task_id: str

    decision: Literal[
        "approved",
        "rejected",
    ]

    reason: str = ""

    status: Literal["recorded"] = "recorded"
