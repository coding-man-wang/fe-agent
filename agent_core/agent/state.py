from datetime import timezone, datetime
from typing import Literal, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_core.agent.trace import TraceEvent, TraceEventType
from agent_core.llm.model import Message


class AgentState(BaseModel):
    run_id:str = Field(default_factory=lambda: str(uuid4()))

    status: Literal["create", "running", "completed", "failed"] = "created"

    step:int = 0

    messages: list[Message] = Field(default_factory=list)

    final_output: str | None = None

    error: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    trace: list[TraceEvent] = Field(default_factory=list)

    def add_trace(self, event_type: TraceEventType, *, data: dict[str, Any] | None = None) -> None:
        event = TraceEvent(event_type=event_type, step=self.step, run_id=self.run_id, data=data or {})

        self.trace.append(event)

