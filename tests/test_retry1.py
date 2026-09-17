import pytest

from agent_core.agent.errors import LLMError
from agent_core.agent.retry import (
    RetryPolicy,
    retry_async,
)


@pytest.mark.asyncio
async def test_retry_success_first_time():

    call_count = 0

    async def operation():

        nonlocal call_count

        call_count += 1

        return "success"

    policy = RetryPolicy(
        max_retries=2,
        initial_delay=0,
        jitter=False,
    )

    result = await retry_async(
        operation,
        policy,
    )

    assert result == "success"
    assert call_count == 1

@pytest.mark.asyncio
async def test_retry_then_success():

    call_count = 0

    async def operation():

        nonlocal call_count

        call_count += 1

        if call_count < 3:
            raise LLMError(
                "temporary error",
                retryable=True,
            )

        return "success"

    policy = RetryPolicy(
        max_retries=2,
        initial_delay=0,
        jitter=False,
    )

    result = await retry_async(
        operation,
        policy,
    )

    assert result == "success"

    assert call_count == 3

@pytest.mark.asyncio
async def test_non_retryable_error():

    call_count = 0

    async def operation():

        nonlocal call_count

        call_count += 1

        raise LLMError(
            "invalid api key",
            retryable=False,
        )

    policy = RetryPolicy(
        max_retries=10,
        initial_delay=0,
        jitter=False,
    )

    with pytest.raises(LLMError):

        await retry_async(
            operation,
            policy,
        )

    assert call_count == 1

@pytest.mark.asyncio
async def test_retry_exhausted():

    call_count = 0

    async def operation():

        nonlocal call_count

        call_count += 1

        raise LLMError(
            "temporary error",
            retryable=True,
        )

    policy = RetryPolicy(
        max_retries=2,
        initial_delay=0,
        jitter=False,
    )

    with pytest.raises(LLMError):

        await retry_async(
            operation,
            policy,
        )

    assert call_count == 3