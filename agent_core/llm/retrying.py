from agent_core.agent.retry import (
    RetryPolicy,
    retry_async,
)
from agent_core.llm.base import (
    LLM,
)
from agent_core.llm.model import (
    ChatRequest,
    ChatResponse,
)


class RetryingLLM(LLM):
    """
    为任意 LLM Adapter 提供统一 Retry 能力。

    Provider Adapter 负责：
        HTTP / SDK
        Provider Error Mapping

    RetryingLLM 负责：
        retry policy
        exponential backoff
        retryable error handling
    """

    def __init__(
        self,
        *,
        llm: LLM,
        retry_policy: RetryPolicy,
    ) -> None:

        self.llm = llm

        self.retry_policy = retry_policy

        # 保留一些常用 metadata，
        # 方便 trace / benchmark 使用。
        self.model = getattr(
            llm,
            "model",
            None,
        )

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        async def operation():
            return await self.llm.chat(request)

        return await retry_async(
            operation,
            self.retry_policy,
        )
