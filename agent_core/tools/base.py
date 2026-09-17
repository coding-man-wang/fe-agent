from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        pass

    @property
    @abstractmethod
    def args_schema(self) -> type(BaseModel):
        pass

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.args_schema.model_json_schema(),
        }
