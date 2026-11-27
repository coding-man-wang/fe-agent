import asyncio
import json
from collections import Counter
from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from agent_core.agent.retry import RetryPolicy, retry_async
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message
from .dispatcher import ToolDispatcher
from .model import RuntimeConfig, RuntimeEvent, RuntimeResult, Observation
from .protocol import function_output, replay_items

FinalValidator = Callable[[str, list[Observation]], None]


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: LLM,
        dispatcher: ToolDispatcher,
        config: RuntimeConfig,
        final_validator: FinalValidator | None = None,
    ):
        if dispatcher.config != config:
            raise ValueError("dispatcher and runtime must share the same policy")
        self.llm = llm
        self.dispatcher = dispatcher
        self.config = config
        self.final_validator = final_validator

    async def run(self, *, system_prompt: str, user_input: str) -> RuntimeResult:
        config = self.config
        started = perf_counter()
        result = RuntimeResult(
            run_id=str(uuid4()), status="blocked", reason="not_started"
        )
        items = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        seen_ids: set[str] = set()
        actions: Counter[str] = Counter()

        def trace(event: str, **data):
            result.trace.append(RuntimeEvent(step=result.steps, event=event, data=data))

        def finish(status, reason, output=None):
            result.status, result.reason, result.final_output = status, reason, output
            result.duration_ms = round((perf_counter() - started) * 1000, 2)
            trace("run_finished", status=status, reason=reason)
            return result

        trace("run_started")
        try:
            async with asyncio.timeout(config.timeout_seconds):
                schemas = self.dispatcher.schemas()
                for step in range(1, config.max_steps + 1):
                    if result.reported_tokens >= config.max_total_tokens:
                        return finish("blocked", "token_budget_exhausted")
                    size = len(
                        json.dumps(
                            {"input": items, "tools": schemas}, ensure_ascii=False
                        ).encode()
                    )
                    if size > config.max_input_bytes:
                        return finish("blocked", "input_budget_exhausted")
                    remaining = config.max_total_tokens - result.reported_tokens
                    if remaining < 16:
                        return finish("blocked", "token_budget_exhausted")
                    result.steps = step
                    trace("llm_started", input_bytes=size)
                    request = ChatRequest(
                        messages=[
                            Message(role="system", content=system_prompt),
                            Message(role="user", content=user_input),
                        ],
                        input_items=items,
                        tools=schemas or None,
                        tool_choice="auto" if schemas else None,
                        max_output_tokens=min(config.max_output_tokens, remaining),
                        replay_reasoning=config.replay_reasoning,
                    )

                    async def call_model():
                        result.llm_attempts += 1
                        return await self.llm.chat(request)

                    def on_retry(number, delay, error):
                        # A failed request can still be billed; usage is incomplete.
                        result.usage_complete = False
                        trace(
                            "llm_retry",
                            number=number,
                            delay=delay,
                            error_type=type(error).__name__,
                        )

                    response = await retry_async(
                        call_model,
                        RetryPolicy(
                            max_retries=config.max_llm_retries,
                            initial_delay=0.25,
                            max_delay=1.0,
                        ),
                        on_retry,
                    )
                    usage = response.total_tokens
                    if (
                        usage is None
                        and response.prompt_tokens is not None
                        and response.completion_tokens is not None
                    ):
                        usage = response.prompt_tokens + response.completion_tokens
                    if usage is None:
                        result.usage_complete = False
                    else:
                        result.reported_tokens += usage
                    trace(
                        "llm_completed",
                        reported_tokens=usage,
                        tool_call_count=len(response.tool_calls),
                    )
                    if usage is None and config.require_reported_usage:
                        return finish("blocked", "token_usage_unavailable")
                    if result.reported_tokens > config.max_total_tokens:
                        return finish("blocked", "token_budget_exhausted")
                    if response.response_status not in (None, "completed"):
                        return finish("blocked", "incomplete_model_response")
                    calls = response.tool_calls
                    if not calls:
                        if not response.content.strip():
                            return finish("blocked", "empty_final_output")
                        if self.final_validator is not None:
                            try:

                                self.final_validator(
                                    response.content,
                                    result.observations,
                                )

                            except Exception as exc:

                                # 保留未通过验收的模型候选输出。
                                # 仅供诊断、审计以及后续修复使用。
                                result.candidate_output = response.content

                                trace(
                                    "final_validation_failed",
                                    validation_code=getattr(
                                        exc,
                                        "code",
                                        "unknown_validation_error",
                                    ),
                                    error_type=type(exc).__name__,
                                )

                                return finish(
                                    "blocked",
                                    "final_evidence_invalid",
                                )
                        return finish("completed", "model_finished", response.content)
                    # Reserve an additional turn for consuming observations.
                    if step == config.max_steps:
                        return finish("blocked", "step_budget_exhausted")
                    if result.reported_tokens >= config.max_total_tokens:
                        return finish("blocked", "token_budget_exhausted")
                    if result.tool_calls + len(calls) > config.max_tool_calls:
                        return finish("blocked", "tool_budget_exhausted")
                    ids = [c.id for c in calls]
                    if (
                        any(not i or len(i) > 256 for i in ids)
                        or len(set(ids)) != len(ids)
                        or seen_ids.intersection(ids)
                    ):
                        return finish("blocked", "invalid_or_duplicate_call_id")
                    keys = [
                        json.dumps(
                            [c.name, c.arguments], sort_keys=True, ensure_ascii=False
                        )
                        for c in calls
                    ]
                    next_actions = actions + Counter(keys)
                    if any(
                        n > config.max_repeated_actions for n in next_actions.values()
                    ):
                        return finish("blocked", "repeated_action_limit")
                    try:
                        replay = replay_items(response)
                    except (ValueError, TypeError, KeyError):
                        return finish("blocked", "invalid_tool_protocol")
                    if (
                        len(json.dumps([*items, *replay], ensure_ascii=False).encode())
                        > config.max_input_bytes
                    ):
                        return finish("blocked", "input_budget_exhausted")
                    items.extend(replay)
                    seen_ids.update(ids)
                    actions = next_actions
                    result.tool_calls += len(calls)
                    observations = await self.dispatcher.execute_batch(calls)
                    result.observations.extend(observations)
                    for observation in observations:
                        trace(
                            "tool_result",
                            call_id=observation.call_id,
                            tool=observation.tool_name,
                            success=observation.success,
                            error_code=observation.error_code,
                            duration_ms=observation.duration_ms,
                        )
                        items.append(
                            function_output(observation.call_id, observation.output())
                        )
                return finish("blocked", "step_budget_exhausted")
        except TimeoutError:
            return finish("blocked", "run_timeout")
        except Exception as exc:
            trace("runtime_error", error_type=type(exc).__name__)
            return finish("failed", "runtime_error")
