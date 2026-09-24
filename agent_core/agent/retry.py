import asyncio
import random
from collections.abc import Callable, Awaitable
from typing import TypeVar

from pydantic import BaseModel, Field

from agent_core.agent.errors import AgentError


class RetryPolicy(BaseModel):
    # 失败之后，最多重新尝试几次
    max_retries: int = Field(default=2, ge=0)

    # 第一次重试前等待多少秒
    initial_delay: float = Field(default=1.0, ge=0)

    # 指数退避（下一次重试等待时间翻倍）
    backoff_factor: float = Field(default=2.0, ge=0)

    # 单次最大等待时间
    max_delay: float = Field(default=10.0, ge=1)

    # 分散重试
    jitter: bool = True

    def get_delay(self, retry_number) -> float:
        if retry_number < 1:
            raise ValueError("retry_number must be >= 1")

        delay = self.initial_delay * (self.backoff_factor ** (retry_number - 1))

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = random.uniform(0, delay)

        return delay


def is_retryable(error: Exception) -> bool:
    return isinstance(error, AgentError) and error.retryable


T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    retry_count = 0

    while True:
        try:
            return await operation()
        except Exception as e:
            if not is_retryable(e):
                raise

            if retry_count >= policy.max_retries:
                raise

            retry_count += 1

            delay = policy.get_delay(retry_number=retry_count)

            if on_retry is not None:
                on_retry(retry_count, delay, e)

            await asyncio.sleep(delay)
