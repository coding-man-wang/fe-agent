from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class TaskCompletionRecord(BaseModel):
    """
    Coding Task 真正完成的持久化结果。

    completed 表示：

    - Patch 已获批准
    - Patch 已完整应用
    - Repository 当前仍然匹配
    - Verification 已通过
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    task_id: str = Field(min_length=1)

    proposal_id: str = Field(
        min_length=64,
        max_length=64,
    )

    status: Literal["completed"] = "completed"

    execution_status: Literal["applied"] = "applied"

    verification_status: Literal["passed"] = "passed"
