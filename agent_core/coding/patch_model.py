from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

PatchAction = Literal[
    "modify",
    "create",
]


class PatchTarget(BaseModel):
    relative_path: str

    action: PatchAction = "modify"

    expected_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    objective: str = Field(min_length=1)

    proposed_approach: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action_state(
        self,
    ) -> "PatchTarget":

        if self.action == "modify" and self.expected_sha256 is None:
            raise ValueError("modify target " "必须提供 " "expected_sha256")

        if self.action == "create" and self.expected_sha256 is not None:
            raise ValueError("create target " "不能存在 " "expected_sha256")

        return self


class PatchPlan(BaseModel):
    task_id: str

    artifact_id: str

    context_hash: str

    targets: list[PatchTarget] = Field(default_factory=list)

    status: Literal["planned"] = "planned"
