import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

HarnessEventType = Literal[
    "investigation_recorded",
    "surface_resolved",
    "context_refined",
    "scope_checked",
    "repository_replanned",
]


class HarnessTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    event_type: HarnessEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class JsonlHarnessTraceStore:
    def __init__(self, *, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: HarnessTraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")

    def read_all(self) -> list[HarnessTraceEvent]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text("utf-8").splitlines():
            if line.strip():
                result.append(HarnessTraceEvent.model_validate_json(line))
        return result
