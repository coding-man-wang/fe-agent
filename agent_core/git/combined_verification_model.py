from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.coding.verification_model import (
    VerificationCheck,
)


class CombinedVerificationPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    integration_id: str

    commit_sha: str

    command_ids: list[str] = Field(
        min_length=1,
    )


class CombinedVerificationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    integration_id: str

    commit_sha: str

    status: Literal[
        "verified_integration",
        "integration_verification_failed",
    ]

    checks: list[VerificationCheck] = Field(
        default_factory=list,
    )
