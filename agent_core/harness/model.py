from pydantic import BaseModel, ConfigDict, Field


class HarnessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_investigation_for_code_change: bool = True
    require_change_surface_before_patch: bool = True
    strict_patch_scope: bool = True
    fail_on_unresolved_surface: bool = False


class HarnessDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    allowed: bool
    stage: str
    reasons: list[str] = Field(default_factory=list)
