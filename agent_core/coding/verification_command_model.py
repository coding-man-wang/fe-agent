from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class AllowedVerificationCommand(BaseModel):
    """
    Harness 预定义的安全验证命令。

    模型不能修改 argv。
    """

    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(min_length=1)

    name: str = Field(min_length=1)

    argv: list[str] = Field(min_length=1)

    working_directory: str = "."

    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=1800,
    )

    max_output_chars: int = Field(
        default=20_000,
        ge=100,
        le=200_000,
    )


class VerificationCommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str

    name: str

    status: Literal[
        "passed",
        "failed",
        "timed_out",
    ]

    exit_code: int | None = None

    stdout: str = ""

    stderr: str = ""

    truncated: bool = False
