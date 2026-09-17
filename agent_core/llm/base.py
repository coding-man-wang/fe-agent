from abc import ABC, abstractmethod

from agent_core.llm.model import ChatRequest, ChatResponse


class LLM(ABC):
    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        pass
