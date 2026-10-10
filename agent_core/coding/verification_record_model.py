from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.coding.verification_model import (
    PatchVerificationResult,
)

VerificationRunStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "interrupted",
]


class VerificationRunRecord(BaseModel):
    """
    一次 Proposal 的 Verification Workflow 状态。

    注意：

    VerificationRunRecord.status
        描述 Workflow 本身。

    PatchVerificationResult.status
        描述验证命令最终是 passed 还是 failed。

    两者不是一个概念。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    proposal_id: str = Field(
        min_length=64,
        max_length=64,
    )

    task_id: str = Field(min_length=1)

    status: VerificationRunStatus = "pending"

    attempts: int = Field(
        default=0,
        ge=0,
    )

    result: PatchVerificationResult | None = None

    error: str | None = None
