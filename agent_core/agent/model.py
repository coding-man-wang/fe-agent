from typing import Literal, List, Any
from pydantic import BaseModel,Field

class UserMessage(BaseModel):
    role: Literal["user"]
    content: str

class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    success: bool = True