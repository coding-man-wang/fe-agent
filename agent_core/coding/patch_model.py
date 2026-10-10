from typing import Literal

from pydantic import BaseModel, Field

PatchAction = Literal["modify",]


class PatchTarget(BaseModel):
    relative_path: str

    action: PatchAction = "modify"

    expected_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    objective: str = Field(min_length=1)

    proposed_approach: str = Field(min_length=1)


class PatchPlan(BaseModel):
    task_id: str

    artifact_id: str

    context_hash: str

    targets: list[PatchTarget] = Field(default_factory=list)

    status: Literal["planned"] = "planned"
