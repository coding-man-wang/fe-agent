import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_core.agent.errors import LLMError
from agent_core.agent.retry import RetryPolicy
from agent_core.benchmark.model_usage import MeteredLLM
from agent_core.llm.model import ChatRequest, ChatResponse, Message
from agent_core.llm.openai import OpenAILLM
from agent_core.llm.retrying import RetryingLLM

REQUEST = ChatRequest(messages=[Message(role="user", content="private prompt")])


def response(**changes):
    return ChatResponse(content="private answer", model="provider-model", **changes)


def provider(*results):
    return SimpleNamespace(
        model="configured-model", chat=AsyncMock(side_effect=results)
    )


def test_legacy_chat_response_has_unknown_usage():
    result = ChatResponse(content="legacy")
    assert (
        result.prompt_tokens is result.completion_tokens is result.total_tokens is None
    )


@pytest.mark.asyncio
async def test_meter_preserves_response_and_marks_task_boundaries():
    first = response(prompt_tokens=80, completion_tokens=20, total_tokens=100)
    second = response(prompt_tokens=30, completion_tokens=10, total_tokens=40)
    clock = iter([1.0, 1.25, 2.0, 2.5])
    wrapped = provider(first, second)
    meter = MeteredLLM(wrapped, clock=lambda: next(clock))
    assert await meter.chat(REQUEST) is first
    mark = meter.mark()
    assert await meter.chat(REQUEST) is second
    report = meter.snapshot()
    assert report["calls"] == report["succeeded_calls"] == 2
    assert report["total_tokens"] == 140
    assert report["latency_ms"] == 750
    assert report["usage_complete"] is True
    task = meter.snapshot(mark)
    assert task["calls"] == 1
    assert task["prompt_tokens"] == 30
    assert task["latency_ms"] == 500
    assert "private" not in json.dumps(report)
    assert report["excluded_usage"] == ["embedding", "rerank"]
    report["models"].append("caller mutation")
    assert meter.snapshot()["models"] == ["provider-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "counts",
    [
        {},
        {"prompt_tokens": 3},
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 8},
    ],
)
async def test_missing_or_inconsistent_usage_never_claims_complete_totals(counts):
    meter = MeteredLLM(provider(response(**counts)))
    await meter.chat(REQUEST)
    report = meter.snapshot()
    assert report["usage_complete"] is False
    assert report["usage_missing_calls"] == 1
    assert report["total_tokens"] is None


@pytest.mark.asyncio
async def test_retry_attempts_are_counted_without_logging_error_messages():
    wrapped = provider(
        LLMError("private API key or endpoint error", retryable=True),
        response(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    meter = MeteredLLM(wrapped)
    retrying = RetryingLLM(
        llm=meter,
        retry_policy=RetryPolicy(max_retries=1, initial_delay=0, jitter=False),
    )
    result = await retrying.chat(REQUEST)
    assert result.total_tokens == 6
    report = meter.snapshot()
    assert report["calls"] == 2
    assert report["failed_calls"] == report["succeeded_calls"] == 1
    assert report["error_types"] == ["LLMError"]
    assert report["known_total_tokens"] == 6
    assert report["total_tokens"] is None
    assert "private" not in json.dumps(report)


@pytest.mark.asyncio
async def test_cancellation_is_recorded_and_propagated():
    meter = MeteredLLM(provider(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await meter.chat(REQUEST)
    assert meter.snapshot()["failed_calls"] == 1
    assert meter.snapshot()["usage_complete"] is False


def test_empty_or_invalid_marks_do_not_create_evidence():
    meter = MeteredLLM(provider())
    assert meter.snapshot()["usage_complete"] is False
    assert meter.snapshot()["calls"] == 0
    for mark in (-1, 1, True, "0"):
        with pytest.raises(ValueError):
            meter.snapshot(mark)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage, expected",
    [
        (
            SimpleNamespace(input_tokens=90, output_tokens=12, total_tokens=102),
            (90, 12, 102),
        ),
        (None, (None, None, None)),
        (
            SimpleNamespace(input_tokens=True, output_tokens=-1, total_tokens="100"),
            (None, None, None),
        ),
    ],
)
async def test_openai_adapter_uses_responses_usage_fields_without_network(
    usage, expected
):
    adapter = object.__new__(OpenAILLM)
    adapter.model = "configured-model"
    adapter.request_timeout_seconds = 10
    adapter.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    output=[],
                    output_text="answer",
                    model="provider-version",
                    usage=usage,
                )
            )
        )
    )
    result = await adapter.chat(REQUEST)
    assert (
        result.prompt_tokens,
        result.completion_tokens,
        result.total_tokens,
    ) == expected
    assert result.model == "provider-version"
    adapter.client.responses.create.assert_awaited_once()
