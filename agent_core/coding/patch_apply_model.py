from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class PatchExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    proposal_id: str

    relative_path: str

    before_sha256: str

    after_sha256: str

    status: Literal["file_written"] = "file_written"

    verification_status: Literal["not_run"] = "not_run"
