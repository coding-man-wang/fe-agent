import json

from agent_core.agent.errors import LLMError
from agent_core.agent.model import ToolCall
from openai import AsyncOpenAI, APIStatusError

from agent_core.llm.model import ChatRequest, ChatResponse
from .base import LLM


class OpenAILLM(LLM):
    def __init__(self, model: str, apikey: str, base_url: str) -> None:
        self.model = model
        self.apikey = apikey
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=apikey, base_url=self.base_url, max_retries=0)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            input_messages = [
                {
                    "role": message.role,
                    "content": message.content
                }
                for message in request.messages
            ]

            response = await self.client.responses.create(
                model=self.model,
                input=input_messages,
                tools=request.tools,
            )

            tool_calls = []
            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append(ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    ))

            return ChatResponse(content=response.output_text, tool_calls=tool_calls, model=self.model)
        except APIConnectionError as e:
            raise LLMError("request timeout", retryable=True) from e
        except APIConnectionError as e:
            raise LLMError("connection failed", retryable=True) from e
        except APIStatusError as e:
            code = e.status_code
            retryable = (
                status_code in {
                    408,
                    409,
                    429,
                } or status_code >= 500
            )
            raise LLMError("request failed", retryable=retryable, status_code=code) from e
        except Exception as e:
            raise LLMError(str(e), retryable=False) from e
