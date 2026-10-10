from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class VerificationPlan(BaseModel):
    """
    一次 Patch 应执行哪些验证命令。

    只保存 command_id，
    不保存任意 Shell。
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str

    task_id: str

    command_ids: list[str] = Field(min_length=1)
