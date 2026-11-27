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
    input_items: list[dict[str, Any]] | None = None
    max_output_tokens: int | None = Field(default=None, ge=16)
    replay_reasoning: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class ChatResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    output_items: list[dict[str, Any]] | None = None
    response_status: str | None = None
    incomplete_reason: str | None = None
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)
    # Provider-reported counts only. None means unavailable, never estimated.
    prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)
