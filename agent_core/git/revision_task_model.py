from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)


class PreparedRevisionTask(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    task_id: str

    base_commit_sha: str

    snapshot_id: str

    analysis_artifact_id: str

    proposal_id: str

    proposal: FilePatchProposal

    status: Literal["awaiting_approval"] = "awaiting_approval"


class ExecutedRevisionTask(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    task_id: str

    base_commit_sha: str

    proposal_id: str

    workspace_id: str

    commit_sha: str

    remote_branch: str

    status: Literal["succeeded"] = "succeeded"
