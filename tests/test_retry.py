import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from agent_core.agent.agent import Agent
from agent_core.agent.config import AgentConfig
from agent_core.agent.errors import LLMError, MaxStepsError
from agent_core.agent.model import ToolCall
from agent_core.agent.retry import RetryPolicy, retry_async
from agent_core.llm.model import ChatRequest, ChatResponse, Message
from agent_core.llm.openai import OpenAILLM
from agent_core.tools.registry import ToolRegistry


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.policy = RetryPolicy(max_retries=2, initial_delay=0, jitter=False)

    async def test_first_success(self):
        operation = AsyncMock(return_value="ok")
        self.assertEqual(await retry_async(operation, self.policy), "ok")
        self.assertEqual(operation.await_count, 1)

    async def test_retry_then_success(self):
        operation = AsyncMock(side_effect=[
            LLMError("temporary", retryable=True),
            LLMError("temporary", retryable=True), "ok",
        ])
        self.assertEqual(await retry_async(operation, self.policy), "ok")
        self.assertEqual(operation.await_count, 3)

    async def test_non_retryable(self):
        error = LLMError("unauthorized", status_code=401)
        operation = AsyncMock(side_effect=error)
        with self.assertRaises(LLMError) as raised:
            await retry_async(operation, self.policy)
        self.assertIs(raised.exception, error)
        self.assertEqual(operation.await_count, 1)

    async def test_exhausted(self):
        error = LLMError("temporary", retryable=True)
        operation = AsyncMock(side_effect=error)
        with self.assertRaises(LLMError) as raised:
            await retry_async(operation, self.policy)
        self.assertIs(raised.exception, error)
        self.assertEqual(operation.await_count, 3)

    async def test_zero_retries(self):
        operation = AsyncMock(side_effect=LLMError("temporary", retryable=True))
        with self.assertRaises(LLMError):
            await retry_async(operation, RetryPolicy(max_retries=0))
        self.assertEqual(operation.await_count, 1)

    async def test_programming_error_not_retried(self):
        operation = AsyncMock(side_effect=TypeError("bug"))
        with self.assertRaises(TypeError):
            await retry_async(operation, self.policy)
        self.assertEqual(operation.await_count, 1)

    async def test_cancelled_operation_not_retried(self):
        operation = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await retry_async(operation, self.policy)
        self.assertEqual(operation.await_count, 1)

    async def test_cancel_during_wait(self):
        operation = AsyncMock(side_effect=LLMError("temporary", retryable=True))
        with patch("agent_core.agent.retry.asyncio.sleep", new_callable=AsyncMock) as sleep:
            sleep.side_effect = asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await retry_async(operation, self.policy)
        self.assertEqual(operation.await_count, 1)

    async def test_backoff_and_cap(self):
        operation = AsyncMock(side_effect=[
            LLMError("temporary", retryable=True) for _ in range(5)
        ] + ["ok"])
        policy = RetryPolicy(max_retries=5, initial_delay=1, max_delay=5, jitter=False)
        with patch("agent_core.agent.retry.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await retry_async(operation, policy)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [1, 2, 4, 5, 5])

    async def test_jitter_uses_capped_delay(self):
        operation = AsyncMock(side_effect=[LLMError("temporary", retryable=True), "ok"])
        with patch("agent_core.agent.retry.random.uniform", return_value=0.25) as uniform:
            with patch("agent_core.agent.retry.asyncio.sleep", new_callable=AsyncMock) as sleep:
                await retry_async(operation, RetryPolicy(initial_delay=2, max_delay=1))
        uniform.assert_called_once_with(0, 1)
        sleep.assert_awaited_once_with(0.25)

    async def test_agent_actually_uses_retry(self):
        llm = AsyncMock()
        llm.chat.side_effect = [LLMError("temporary", retryable=True), ChatResponse(content="ok")]
        agent = Agent(llm, ToolRegistry(), AgentConfig(max_steps=1), self.policy)
        result = await agent.run("hello")
        self.assertEqual(result.content, "ok")
        self.assertEqual(llm.chat.await_count, 2)

    async def test_llm_retry_does_not_repeat_tool(self):
        class CountingTool:
            name = "count"
            def __init__(self):
                self.calls = 0
            def to_schema(self):
                return {"type": "function", "name": self.name, "parameters": {"type": "object"}}
            async def execute(self, arguments):
                self.calls += 1
                return {"ok": True}

        tool = CountingTool()
        registry = ToolRegistry()
        registry.register(tool)
        llm = AsyncMock()
        llm.chat.side_effect = [
            ChatResponse(content="", tool_calls=[ToolCall(id="c1", name="count", arguments={})]),
            LLMError("temporary", retryable=True), ChatResponse(content="ok"),
        ]
        agent = Agent(llm, registry, AgentConfig(max_steps=2), self.policy)
        await agent.run("run tool")
        self.assertEqual(tool.calls, 1)
        self.assertEqual(llm.chat.await_count, 3)

    async def test_max_steps_still_enforced(self):
        llm = AsyncMock()
        llm.chat.return_value = ChatResponse(content="", tool_calls=[
            ToolCall(id="c1", name="missing", arguments={}),
        ])
        agent = Agent(llm, ToolRegistry(), AgentConfig(max_steps=2), self.policy)
        with self.assertRaises(MaxStepsError):
            await agent.run("run tool")
        self.assertEqual(llm.chat.await_count, 2)

    def test_policy_rejects_invalid_values(self):
        for data in [
            {"max_retries": -1}, {"initial_delay": -1},
            {"backoff_factor": 0.5}, {"max_delay": float("inf")},
        ]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                RetryPolicy(**data)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_retry_disabled(self):
        with patch("agent_core.llm.openai.AsyncOpenAI") as client:
            OpenAILLM("test", "dummy-test-key", "https://example.invalid/v1")
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.call_args.kwargs["timeout"], 30.0)

    async def test_provider_error_mapping(self):
        # SDK errors only: no network calls.
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        from importlib import import_module
        # The installed SDK may use httpx or httpx2 internally.
        import openai._base_client as sdk_client
        http_module = getattr(sdk_client, "httpx", None)
        if http_module is None:
            http_module = import_module("httpx2")
        request = http_module.Request("POST", "https://example.invalid/v1/responses")
        cases = [
            (APIConnectionError(request=request), True, None),
            (APITimeoutError(request=request), True, None),
        ]
        for status, expected, code in [
            (400, False, "bad_request"), (401, False, "invalid_api_key"),
            (408, True, "timeout"), (409, True, "conflict"),
            (429, True, "rate_limit_exceeded"),
            (429, False, "insufficient_quota"), (503, True, "unavailable"),
        ]:
            response = http_module.Response(status, request=request)
            error = APIStatusError("test", response=response, body={"code": code})
            cases.append((error, expected, status))
        with patch("agent_core.llm.openai.AsyncOpenAI") as factory:
            adapter = OpenAILLM("test", "dummy-test-key", "https://example.invalid/v1")
            for error, expected, status in cases:
                with self.subTest(error=type(error).__name__, status=status):
                    factory.return_value.responses.create = AsyncMock(side_effect=error)
                    with self.assertRaises(LLMError) as raised:
                        await adapter.chat(ChatRequest(messages=[Message(role="user", content="hi")]))
                    self.assertEqual(raised.exception.retryable, expected)
                    self.assertEqual(raised.exception.status_code, status)
                    self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
