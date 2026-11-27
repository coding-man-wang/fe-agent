import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from agent_core.agent.model import ToolCall
from agent_core.llm.model import ChatResponse
from agent_core.runtime.dispatcher import ToolDispatcher
from agent_core.runtime.engine import AgentRuntime
from agent_core.runtime.model import RuntimeConfig
from agent_core.runtime.repository import (
    Snapshot,
    repository_registry,
    validate_investigation,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request.model_copy(deep=True))
        return next(self.responses)


def call(id="c1", name="read_file", **args):
    return ToolCall(id=id, name=name, arguments=args or {"path": "src/orders.ts"})


def response(*calls, content="", tokens=20, **kwargs):
    return ChatResponse(
        content=content, tool_calls=list(calls), total_tokens=tokens, **kwargs
    )


def make(responses, config=None, validator=None):
    config = config or RuntimeConfig()
    snapshot = Snapshot(
        "abc",
        {
            "src/orders.ts": 'export const status = "paid";\n',
            "src/mock.ts": "const listOrders = () => [];\n",
        },
    )
    llm = FakeLLM(responses)
    dispatcher = ToolDispatcher(
        repository_registry(snapshot),
        allowed_names={"read_file", "search_code", "list_files"},
        config=config,
    )
    return (
        AgentRuntime(
            llm=llm, dispatcher=dispatcher, config=config, final_validator=validator
        ),
        llm,
    )


def run(runtime):
    return asyncio.run(
        runtime.run(system_prompt="investigate", user_input="order filter")
    )


def test_multiturn_protocol_and_evidence():
    report = json.dumps(
        {
            "findings": [
                {
                    "claim": "status exists",
                    "call_id": "c2",
                    "path": "src/orders.ts",
                    "line": 1,
                    "quote": 'status = "paid"',
                }
            ],
            "unresolved_questions": [],
        }
    )
    runtime, llm = make(
        [
            response(call("c1", "search_code", query="status")),
            response(call("c2")),
            response(content=report),
        ],
        validator=validate_investigation,
    )
    result = run(runtime)
    assert result.status == "completed"
    assert result.tool_calls == 2 and result.reported_tokens == 60
    items = llm.requests[2].input_items
    assert [i["call_id"] for i in items if i.get("type") == "function_call"] == [
        "c1",
        "c2",
    ]
    assert [i["call_id"] for i in items if i.get("type") == "function_call_output"] == [
        "c1",
        "c2",
    ]
    assert all("parameters" in schema for schema in llm.requests[0].tools)


@pytest.mark.parametrize(
    "calls,reason",
    [
        ([call(), call()], "invalid_or_duplicate_call_id"),
        ([call("c1"), call("c2"), call("c3")], "repeated_action_limit"),
    ],
)
def test_bad_batch_no_side_effect(calls, reason):
    runtime, _ = make([response(*calls)])
    result = run(runtime)
    assert result.reason == reason and result.tool_calls == 0


def test_tool_budget_atomic_batch():
    runtime, _ = make(
        [response(call("c1"), call("c2"))], RuntimeConfig(max_tool_calls=1)
    )
    result = run(runtime)
    assert result.reason == "tool_budget_exhausted" and not result.observations


def test_unauthorized_and_invalid_arguments():
    runtime, _ = make(
        [
            response(
                call("a", "write_file", path="a"),
                call("b", "read_file", path="src/orders.ts", start_line="1"),
                call("c", "read_file", path="../secret"),
            ),
            response(content="done"),
        ]
    )
    result = run(runtime)
    assert [o.error_code for o in result.observations] == [
        "tool_not_allowed",
        "invalid_arguments",
        "tool_reported_failure",
    ]


@pytest.mark.parametrize(
    "resp,config,reason",
    [
        (response(call(), tokens=None), RuntimeConfig(), "token_usage_unavailable"),
        (
            response(call(), tokens=100),
            RuntimeConfig(max_total_tokens=50),
            "token_budget_exhausted",
        ),
        (response(call()), RuntimeConfig(max_steps=1), "step_budget_exhausted"),
        (
            response(content="x", response_status="incomplete"),
            RuntimeConfig(),
            "incomplete_model_response",
        ),
    ],
)
def test_termination(resp, config, reason):
    runtime, _ = make([resp], config)
    result = run(runtime)
    assert result.reason == reason and result.tool_calls == 0


def test_fabricated_evidence_blocked():
    report = json.dumps(
        {
            "findings": [
                {
                    "claim": "invented",
                    "call_id": "c1",
                    "path": "src/orders.ts",
                    "line": 1,
                    "quote": "not in source",
                }
            ],
            "unresolved_questions": [],
        }
    )
    runtime, _ = make(
        [response(call()), response(content=report)], validator=validate_investigation
    )
    assert run(runtime).reason == "final_evidence_invalid"


def test_snapshot_reads_commit_not_dirty_workspace(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (tmp_path / "a.ts").write_text("committed", encoding="utf-8")
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "test")
    (tmp_path / "a.ts").write_text("dirty", encoding="utf-8")
    snap = Snapshot.from_git(tmp_path, ["a.ts"])
    assert snap.files["a.ts"] == "committed"
    for path in [".env", "../a.ts", "missing.ts"]:
        with pytest.raises(ValueError):
            Snapshot.from_git(tmp_path, [path])


def test_run_timeout():
    runtime, llm = make([], RuntimeConfig(timeout_seconds=0.01))

    async def slow(_):
        await asyncio.sleep(1)

    llm.chat = slow
    assert run(runtime).reason == "run_timeout"


def test_parallel_order_and_timeout():
    from pydantic import BaseModel
    from agent_core.tools.base import Tool
    from agent_core.tools.registry import ToolRegistry

    class Args(BaseModel):
        delay: float

    class Slow(Tool):
        name = "slow"
        description = "test"
        args_schema = Args

        async def execute(self, arguments):
            await asyncio.sleep(arguments["delay"])
            return {"delay": arguments["delay"]}

    config = RuntimeConfig(tool_timeout_seconds=0.05, max_parallel_tools=2)
    registry = ToolRegistry()
    registry.register(Slow())
    dispatcher = ToolDispatcher(registry, allowed_names={"slow"}, config=config)
    result = asyncio.run(
        dispatcher.execute_batch(
            [call("a", "slow", delay=0.2), call("b", "slow", delay=0.0)]
        )
    )
    assert [o.call_id for o in result] == ["a", "b"]
    assert result[0].error_code == "tool_timeout" and result[1].success


def test_provider_replay_mismatch():
    runtime, _ = make(
        [
            response(
                call(),
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": "wrong",
                        "name": "read_file",
                        "arguments": "{}",
                    }
                ],
            )
        ]
    )
    assert run(runtime).reason == "invalid_tool_protocol"


def test_openai_adapter_wire_format():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    pytest.importorskip("openai")
    from agent_core.llm.openai import OpenAILLM
    from agent_core.llm.model import ChatRequest, Message

    llm = OpenAILLM(model="fake", apikey="fake", base_url="https://example.invalid/v1")
    raw = {
        "type": "function_call",
        "call_id": "c1",
        "name": "read_file",
        "arguments": '{"path":"src/orders.ts"}',
    }
    item = SimpleNamespace(**raw, model_dump=lambda **_: raw)
    mock = AsyncMock(
        return_value=SimpleNamespace(
            output=[item],
            output_text="",
            status="completed",
            model="fake",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
    )
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=mock))
    items = [{"role": "user", "content": "test"}]
    req = ChatRequest(
        messages=[Message(role="user", content="test")],
        input_items=items,
        tool_choice="auto",
        max_output_tokens=100,
        replay_reasoning=True,
    )
    res = asyncio.run(llm.chat(req))
    kwargs = mock.call_args.kwargs
    assert (
        kwargs["input"] == items
        and kwargs["tool_choice"] == "auto"
        and kwargs["max_output_tokens"] == 100
    )
    assert kwargs["store"] is False and kwargs["include"] == [
        "reasoning.encrypted_content"
    ]
    assert res.output_items == [raw] and res.total_tokens == 15


def test_duplicate_across_turns():
    runtime, _ = make([response(call()), response(call())])
    result = run(runtime)
    assert result.reason == "invalid_or_duplicate_call_id" and result.tool_calls == 1


def test_input_budget():
    runtime, llm = make([], RuntimeConfig(max_input_bytes=1024))
    result = asyncio.run(runtime.run(system_prompt="x" * 2000, user_input="task"))
    assert result.reason == "input_budget_exhausted" and not llm.requests


def test_unknown_usage_opt_out():
    runtime, _ = make(
        [response(content="done", tokens=None)],
        RuntimeConfig(require_reported_usage=False),
    )
    result = run(runtime)
    assert result.status == "completed" and not result.usage_complete


def test_output_budget():
    config = RuntimeConfig(max_tool_output_bytes=512)
    snapshot = Snapshot("abc", {"src/orders.ts": "x" * 1000})
    dispatcher = ToolDispatcher(
        repository_registry(snapshot), allowed_names={"read_file"}, config=config
    )
    result = asyncio.run(dispatcher.execute_batch([call()]))
    assert result[0].error_code == "tool_output_too_large"


def test_unresolved_report_blocked():
    report = json.dumps(
        {
            "findings": [
                {
                    "claim": "status exists",
                    "call_id": "c1",
                    "path": "src/orders.ts",
                    "line": 1,
                    "quote": "status",
                }
            ],
            "unresolved_questions": ["mock behavior unknown"],
        }
    )
    runtime, _ = make(
        [response(call()), response(content=report)], validator=validate_investigation
    )
    result = run(runtime)
    assert result.reason == "final_evidence_invalid" and result.final_output is None
    assert result.candidate_output == report


def test_retryable_model_failure_reuses_existing_retry_policy():
    from agent_core.agent.errors import LLMError

    runtime, llm = make([], RuntimeConfig(max_llm_retries=1))
    attempts = 0

    async def flaky(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMError("temporary", retryable=True)
        return response(content="done")

    llm.chat = flaky
    result = run(runtime)
    assert result.status == "completed" and result.llm_attempts == 2
    assert not result.usage_complete and any(
        e.event == "llm_retry" for e in result.trace
    )


def test_nonretryable_model_failure():
    from agent_core.agent.errors import LLMError

    runtime, llm = make([])

    async def fail(_):
        raise LLMError("bad request", retryable=False)

    llm.chat = fail
    result = run(runtime)
    assert result.status == "failed" and result.llm_attempts == 1


def test_compact_evidence_reference():

    report = json.dumps(
        {
            "findings": [
                {
                    "claim": "status exists",
                    "call_id": "E001",
                    "path": "src/orders.ts",
                    "line": 1,
                    "quote": "status",
                }
            ],
            "unresolved_questions": [],
        }
    )

    config = RuntimeConfig(
        compact_evidence=True,
    )

    runtime, llm = make(
        [
            response(call("provider_long_call_001")),
            response(content=report),
        ],
        config=config,
        validator=validate_investigation,
    )

    result = run(runtime)

    assert result.status == "completed"

    observation = result.observations[0]

    assert observation.call_id == ("provider_long_call_001")

    assert observation.evidence_id == "E001"

    outputs = [
        item
        for item in llm.requests[1].input_items
        if item.get("type") == "function_call_output"
    ]

    assert len(outputs) == 1

    assert outputs[0]["call_id"] == ("provider_long_call_001")

    payload = json.loads(outputs[0]["output"])

    assert payload["evidence_id"] == "E001"

    assert "call_id" not in payload


def test_invalid_compact_evidence_reference():

    report = json.dumps(
        {
            "findings": [
                {
                    "claim": "status exists",
                    "call_id": "E999",
                    "path": "src/orders.ts",
                    "line": 1,
                    "quote": "status",
                }
            ],
            "unresolved_questions": [],
        }
    )

    runtime, _ = make(
        [
            response(call()),
            response(content=report),
        ],
        config=RuntimeConfig(
            compact_evidence=True,
        ),
        validator=validate_investigation,
    )

    result = run(runtime)

    assert result.status == "blocked"

    assert result.reason == ("final_evidence_invalid")

    assert result.candidate_output == report


def test_finalize_when_tool_budget_exhausted():

    runtime, llm = make(
        [
            response(call()),
            response(content="done"),
        ],
        config=RuntimeConfig(
            max_steps=6,
            max_tool_calls=1,
            force_finalization=True,
        ),
    )

    result = run(runtime)

    assert result.status == "completed"

    assert result.tool_calls == 1

    assert len(llm.requests) == 2

    assert llm.requests[0].tool_choice == "auto"

    assert llm.requests[1].tool_choice == "none"

    assert any(event.event == "finalization_started" for event in result.trace)


def test_finalize_on_last_model_step():

    runtime, llm = make(
        [
            response(call()),
            response(content="done"),
        ],
        config=RuntimeConfig(
            max_steps=2,
            force_finalization=True,
        ),
    )

    result = run(runtime)

    assert result.status == "completed"

    assert result.steps == 2

    assert llm.requests[1].tool_choice == "none"
