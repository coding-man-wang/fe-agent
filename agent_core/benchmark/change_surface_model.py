from pydantic import BaseModel, ConfigDict, Field


class ChangeSurfaceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    expected_change_files: list[str] = Field(default_factory=list)
    predicted_change_files: list[str] = Field(default_factory=list)
    matched_change_files: list[str] = Field(default_factory=list)
    missed_change_files: list[str] = Field(default_factory=list)
    unexpected_change_files: list[str] = Field(default_factory=list)
    forbidden_change_hits: list[str] = Field(default_factory=list)
    target_recall: float = Field(ge=0.0, le=1.0)
    target_precision: float = Field(ge=0.0, le=1.0)
    dependency_expected_files: list[str] = Field(default_factory=list)
    dependency_coverage: float = Field(ge=0.0, le=1.0)
    failure_reasons: list[str] = Field(default_factory=list)
