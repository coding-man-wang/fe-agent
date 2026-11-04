import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from agent_core.agent.errors import (
    LLMError,
)
from agent_core.agent.model import (
    ToolCall,
)
from agent_core.llm.model import (
    ChatRequest,
    ChatResponse,
)
from .base import LLM


class OpenAILLM(LLM):

    def __init__(
        self,
        model: str,
        apikey: str,
        base_url: str,
        request_timeout_seconds: float = 60.0,
    ) -> None:

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds " "必须大于 0")

        self.model = model
        self.apikey = apikey
        self.base_url = base_url

        self.request_timeout_seconds = request_timeout_seconds

        self.client = AsyncOpenAI(
            api_key=apikey,
            base_url=self.base_url,
            # Retry 由上层 RetryingLLM 统一控制，
            # 避免 SDK Retry + Agent Retry 叠加。
            max_retries=0,
            # 最关键的修改：
            # 单次 HTTP 请求不能无限等待。
            timeout=request_timeout_seconds,
        )

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        try:

            input_messages = [
                {
                    "role": message.role,
                    "content": message.content,
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

                    tool_calls.append(
                        ToolCall(
                            id=item.call_id,
                            name=item.name,
                            arguments=json.loads(item.arguments),
                        )
                    )

            return ChatResponse(
                content=response.output_text,
                tool_calls=tool_calls,
                model=self.model,
            )

        except APITimeoutError as exc:

            raise LLMError(
                ("LLM request timeout " f"after " f"{self.request_timeout_seconds}s"),
                retryable=True,
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "LLM connection failed",
                retryable=True,
            ) from exc

        except APIStatusError as exc:

            status_code = exc.status_code

            body = exc.body

            error_code = None

            if isinstance(body, dict):

                error_code = body.get("code")

                nested_error = body.get("error")

                if error_code is None and isinstance(nested_error, dict):
                    error_code = nested_error.get("code")

            retryable = status_code in {408, 409, 429} or status_code >= 500

            if status_code == 429 and error_code == "insufficient_quota":
                retryable = False

            raise LLMError(
                f"LLM request failed with status {status_code}",
                retryable=retryable,
                status_code=status_code,
            ) from exc

        except Exception as exc:

            raise LLMError(
                str(exc),
                retryable=False,
            ) from exc
