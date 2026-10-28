from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

BatchExecutionStatus = Literal[
    "staged",
    "applying",
    "applied",
    "partially_applied",
    "failed",
    "rolling_back",
    "rolled_back",
    "rollback_failed",
]

FileExecutionStatus = Literal[
    "pending",
    "applied",
    "rolled_back",
]


class FileExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)

    action: Literal[
        "modify",
        "create",
    ] = "modify"

    expected_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    after_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    status: FileExecutionStatus = "pending"


class PatchExecutionJournal(BaseModel):
    """
    一次 Patch Proposal 产生真实文件副作用时的执行日志。

    Journal 的核心职责不是保存 Patch 内容，
    而是回答：

    - 这次执行对应哪个 Proposal？
    - 执行到了哪个阶段？
    - 哪些文件已经产生副作用？
    - 是否发生异常？
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    proposal_id: str = Field(
        min_length=64,
        max_length=64,
    )

    task_id: str = Field(min_length=1)

    status: BatchExecutionStatus

    files: list[FileExecutionRecord] = Field(min_length=1)

    error: str | None = None
