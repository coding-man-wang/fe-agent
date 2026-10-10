from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    relative_path: str

    expected_sha256: str

    after_sha256: str

    before_content: str

    after_content: str

    explanation: str

    unified_diff: str


class FilePatchProposal(BaseModel):
    task_id: str

    artifact_id: str

    context_hash: str

    status: Literal["generated_not_applied"] = "generated_not_applied"

    patches: list[ValidatedFilePatch] = Field(min_length=1)
