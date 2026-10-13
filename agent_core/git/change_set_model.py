from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

GitChangeKind = Literal[
    "modified",
    "added",
    "deleted",
    "renamed",
    "untracked",
    "other",
]


class GitFileChange(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str

    kind: GitChangeKind

    raw_status: str


class GitChangeSet(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    files: list[GitFileChange] = Field(default_factory=list)

    @property
    def paths(self) -> set[str]:
        return {item.relative_path for item in self.files}


class ChangeSetValidationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    allowed: bool

    expected_paths: list[str]

    actual_paths: list[str]

    unexpected_paths: list[str]

    missing_paths: list[str]

    invalid_operations: list[str]
