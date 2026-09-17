from typing import Any, Dict

from agent_core.agent.errors import ToolNotFoundError
from agent_core.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool
    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())