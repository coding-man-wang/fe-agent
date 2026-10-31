from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SymbolCandidate(BaseModel):
    """符号候选：只携带阅读相关性及来源证据，不包含修改权限。"""

    model_config = ConfigDict(extra="forbid")

    symbol_id: str
    name: str
    kind: str
    relative_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    matched_queries: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    source: Literal["name", "graph"] = "name"
    seed_symbol_id: str | None = None
    relation: Literal["calls", "references"] | None = None
    direction: Literal["outgoing", "incoming"] | None = None
    depth: int = Field(default=0, ge=0, le=1)
