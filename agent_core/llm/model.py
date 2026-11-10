from typing import Literal, List, Any
from agent_core.agent.model import ToolCall
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ChatResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    # Provider-reported counts only. None means unavailable, never estimated.
    prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)
