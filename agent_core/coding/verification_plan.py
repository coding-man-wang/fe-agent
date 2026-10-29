from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

VerificationExecutionMode = Literal[
    "fail_fast",
    "collect_all",
]


class VerificationPlan(BaseModel):
    """
    一次 Patch 应执行哪些验证命令，
    以及这些验证命令采用什么执行策略。

    fail_fast:
        任意一个 Command 失败后立即停止。

    collect_all:
        即使前面的 Command 失败，
        仍继续执行剩余 Verification，
        尽量收集完整 Failure Evidence。

    注意：

    execution_mode 只控制
    Harness 如何执行已经批准的
    Verification Commands。

    它不允许模型新增任意 Shell。
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str

    task_id: str

    command_ids: list[str] = Field(min_length=1)

    execution_mode: VerificationExecutionMode = "fail_fast"
