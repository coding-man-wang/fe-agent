from abc import abstractmethod
from typing import Any
from pathlib import Path

from agent_core.agent.errors import ToolValidationError, ToolExecutionError
from agent_core.tools.base import Tool
from pydantic import BaseModel, Field, ValidationError


class ListFileInput(BaseModel):
    path: str = Field(default=".", description="需要查看的目录路径")


class ListFileTool(Tool):
    @property
    def name(self) -> str:
        return "list_file"

    @property
    def description(self) -> str:
        return "列出指定目录下的文件和目录"

    @property
    def args_schema(self) -> type(ListFileInput):
        return ListFileInput

    async def execute(self, arguments: dict[str, Any]) -> Any:
        try:
            input_data = self.args_schema.model_validate(arguments)
        except ValidationError as e:
            raise ToolValidationError(self.name, str(e)) from e

        try:
            directory = Path(input_data.path)

            if not directory.exists():
                return {
                    "success": False,
                    "error": f"目录不存在: {path}",
                }

            items = []
            for item in directory.iterdir():
                items.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                    }
                )

            return {
                "success": True,
                "path": str(directory),
                "items": items,
            }
        except Exception as e:
            raise ToolExecutionError(self.name, str(e)) from e
