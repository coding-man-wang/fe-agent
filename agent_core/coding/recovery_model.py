from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

FileRecoveryState = Literal[
    "absent",
    "before",
    "after",
    "diverged",
]


class FileRecoveryAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str

    action: Literal[
        "modify",
        "create",
    ] = "modify"

    journal_status: Literal[
        "pending",
        "applied",
        "rolled_back",
    ]

    current_state: FileRecoveryState

    current_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )


class RecoveryAssessment(BaseModel):
    """
    对整个 Patch Execution Batch 的恢复判断。

    注意这里不是只给出一个 decision。

    因为 partially applied 状态下，
    Resume 和 Rollback 可能同时都是安全选项。
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str

    task_id: str

    journal_status: str

    can_resume: bool = False

    can_rollback: bool = False

    requires_manual_recovery: bool = False

    files: list[FileRecoveryAssessment] = Field(min_length=1)

    reason: str
