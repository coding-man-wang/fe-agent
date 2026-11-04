from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used_files: int = Field(ge=0)
    context_chars: int = Field(ge=0)

    required_symbol_count: int = Field(ge=0)
    covered_symbol_count: int = Field(ge=0)
    symbol_coverage: float = Field(ge=0, le=1)

    missing_symbol_ids: list[str] = Field(default_factory=list)

    # Integrity 使用现有独立 Evaluator 的检查结果
    budget_preserved: bool
    source_integrity_preserved: bool
    metadata_integrity_preserved: bool

    # 无关文件必须由 Benchmark Oracle 预先标注
    noise_chars: int = Field(ge=0)
    noise_ratio: float = Field(ge=0, le=1)


class ContextComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: Literal["passed", "failed"]

    baseline: ContextQualityMetrics
    enhanced: ContextQualityMetrics

    saved_chars: int
    reduction_ratio: float
    coverage_delta: float

    no_coverage_regression: bool
    issues: list[str] = Field(default_factory=list)
