from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

VerificationStatus = Literal[
    "not_started",
    "running",
    "passed",
    "failed",
]


class VerificationCheck(BaseModel):
    """
    一项确定性的验证检查。

    例如：

    - eslint
    - typecheck
    - unit test
    - build
    - playwright
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)

    status: Literal[
        "passed",
        "failed",
    ]

    details: str = ""

    command_id: str | None = None


class PatchVerificationResult(BaseModel):
    """
    Patch Apply 之后的验证状态。

    注意：

    files applied
    !=
    verification passed
    !=
    task completed

    三者需要分开建模。
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(
        min_length=64,
        max_length=64,
    )

    task_id: str = Field(min_length=1)

    status: VerificationStatus = "not_started"

    checks: list[VerificationCheck] = Field(default_factory=list)
