from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    max_steps: int = Field(default=6, ge=1, le=50)
    max_llm_retries: int = Field(default=1, ge=0, le=3)
    max_tool_calls: int = Field(default=12, ge=1, le=100)
    max_total_tokens: int = Field(default=30000, ge=1)
    max_output_tokens: int = Field(default=2048, ge=16)
    max_input_bytes: int = Field(default=180000, ge=1024)
    max_tool_output_bytes: int = Field(default=16000, ge=512)
    max_argument_bytes: int = Field(default=4096, ge=128)
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    tool_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    max_parallel_tools: int = Field(default=3, ge=1, le=16)
    max_repeated_actions: int = Field(default=2, ge=1)
    require_reported_usage: bool = True
    replay_reasoning: bool = False


class Observation(BaseModel):
    call_id: str
    tool_name: str
    success: bool
    data: Any = None
    error_code: str | None = None
    duration_ms: float = 0

    def output(self) -> str:
        return self.model_dump_json(exclude={'duration_ms'})


class RuntimeEvent(BaseModel):
    step: int
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class RuntimeResult(BaseModel):
    run_id: str
    status: Literal['completed', 'blocked', 'failed']
    reason: str
    steps: int = 0
    llm_attempts: int = 0
    tool_calls: int = 0
    reported_tokens: int = 0
    usage_complete: bool = True
    final_output: str | None = None
    candidate_output: str | None = None  # diagnostic only; never accepted evidence
    observations: list[Observation] = Field(default_factory=list)
    trace: list[RuntimeEvent] = Field(default_factory=list)
    duration_ms: float = 0
