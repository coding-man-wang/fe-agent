from datetime import timezone, datetime
from uuid import uuid4
from typing import Literal, Any

from pydantic import BaseModel, Field

TraceEventType = Literal[
    "run_started",
    "llm_started",
    "llm_completed",
    "llm_failed",
    "retry_scheduled",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "run_completed",
    "run_failed",
]


class TraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))

    run_id: str

    event_type: TraceEventType

    step: int = 0

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 当前事件附加的信息
    data: dict[str, Any] = Field(default_factory=dict)
