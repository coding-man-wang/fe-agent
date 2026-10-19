from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


ArtifactType = Literal[
    "investigation",
    "change_surface",
    "replan",
]


class TaskArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str = Field(min_length=1)
    artifact_type: ArtifactType
    depends_on_task_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    payload: dict[str, Any]


class ArtifactGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: Literal["task", "artifact"]
    task_id: str
    artifact_type: ArtifactType | None = None


class ArtifactGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: Literal[
        "produces",
        "consumes",
        "depends_on",
    ]


class ArtifactGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[ArtifactGraphNode] = Field(default_factory=list)
    edges: list[ArtifactGraphEdge] = Field(default_factory=list)
