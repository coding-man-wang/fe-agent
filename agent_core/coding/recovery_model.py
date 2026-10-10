from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

FileRecoveryState = Literal[
    "before",
    "after",
    "diverged",
]


class FileRecoveryAssessment(BaseModel):
    """
    单个文件当前磁盘状态与 Proposal / Journal
    对账后的结果。

    before:
        当前文件仍然是 Agent 执行前的版本。

    after:
        当前文件正好是 Agent 批准并写入后的版本。

    diverged:
        当前文件既不是 before，也不是 after。
        通常意味着 Agent 执行之后又发生了外部修改。
    """

    model_config = ConfigDict(extra="forbid")

    relative_path: str

    journal_status: Literal[
        "pending",
        "applied",
        "rolled_back",
    ]

    current_state: FileRecoveryState

    current_sha256: str = Field(
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
