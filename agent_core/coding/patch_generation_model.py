from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from agent_core.coding.patch_model import (
    PatchAction,
)


class GeneratedFileChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)

    after_content: str = Field(min_length=1)

    explanation: str = Field(min_length=1)


class GeneratedPatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    changes: list[GeneratedFileChange] = Field(min_length=1)


class ValidatedFilePatch(BaseModel):
    action: PatchAction = "modify"

    relative_path: str

    expected_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    after_sha256: str

    before_content: str | None = None

    after_content: str

    explanation: str

    unified_diff: str

    @model_validator(mode="after")
    def validate_before_state(
        self,
    ) -> "ValidatedFilePatch":

        if self.action == "modify":
            if self.expected_sha256 is None or self.before_content is None:
                raise ValueError("modify patch " "必须有 before state")

        if self.action == "create":
            if self.expected_sha256 is not None or self.before_content is not None:
                raise ValueError("create patch " "的 before state " "必须是 absent")

        return self


class FilePatchProposal(BaseModel):
    task_id: str

    artifact_id: str

    context_hash: str

    status: Literal["generated_not_applied"] = "generated_not_applied"

    patches: list[ValidatedFilePatch] = Field(min_length=1)
