from __future__ import annotations

import json

from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message


class BenchmarkTextModelError(ValueError):
    pass


def extract_json_object(text: str) -> str:
    """Normalize common provider formatting without weakening downstream schemas."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    # Fast path: already valid JSON object.
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)

    # Provider occasionally adds one sentence around JSON. Extract the outer object,
    # but leave structural validation to the consumer's Pydantic schema.
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise BenchmarkTextModelError("LLM 返回内容中找不到 JSON object")

    candidate = value[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise BenchmarkTextModelError("LLM 返回内容不是合法 JSON object") from exc
    if not isinstance(parsed, dict):
        raise BenchmarkTextModelError("LLM JSON 顶层必须是 object")
    return json.dumps(parsed, ensure_ascii=False)


class LLMTextModelAdapter:
    """Adapt the project's generic LLM.chat contract to coding TextModel.complete."""

    def __init__(self, *, llm: LLM) -> None:
        self.llm = llm

    async def complete(self, *, system_message: str, user_message: str) -> str:
        response = await self.llm.chat(
            ChatRequest(
                messages=[
                    Message(role="system", content=system_message),
                    Message(role="user", content=user_message),
                ],
                tool_choice="none",
            )
        )
        return extract_json_object(response.content)
