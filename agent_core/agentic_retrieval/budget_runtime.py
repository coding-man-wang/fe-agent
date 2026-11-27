"""v1.6.2 bounded runtime. Existing v1.5 runtime stays unchanged.

Reuses the repository's dispatcher, retry policy and Responses replay protocol.
Finalization is a separate, tool-free request built from validated source windows.
Token estimates are advisory; provider-reported totals remain authoritative.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from math import ceil
from time import perf_counter
from uuid import uuid4

from agent_core.agent.retry import RetryPolicy, retry_async
from agent_core.llm.model import ChatRequest, Message
from agent_core.runtime.engine import AgentRuntime
from agent_core.runtime.model import RuntimeEvent, RuntimeResult
from agent_core.runtime.protocol import function_output, replay_items


def serialized_bytes(items, schemas):
    return len(json.dumps({"input": items, "tools": schemas}, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8"))


def estimated_input_tokens(size: int, observed_ratio: float = 0.0) -> int:
    # Conservative starting heuristic, calibrated upward using actual input usage.
    # Not a tokenizer and not a billing limit; provider framing/caching can differ.
    return ceil(size * max(0.30, observed_ratio * 1.12)) + 128


def evidence_data(observation):
    if not observation.success or not isinstance(observation.data, dict):
        return None
    if observation.tool_name == "read_file":
        data = observation.data
    elif observation.tool_name == "retrieve_evidence":
        data = observation.data.get("evidence")
    else:
        return None
    return data if isinstance(data, dict) and "lines" in data else None


def model_observation(observation):
    payload = observation.model_dump(mode="json", exclude={"duration_ms"}, exclude_none=True)
    if observation.evidence_id:
        payload.pop("call_id", None)
        data = payload["data"]
        data = data["evidence"] if observation.tool_name == "retrieve_evidence" else data
        data.pop("lines", None)
        data.pop("registered_evidence_id", None)
        data["evidence_id"] = observation.evidence_id
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class BudgetedInvestigationRuntime(AgentRuntime):
    """Investigation-specific scheduling; no behavioral change to legacy AgentRuntime."""

    def __init__(self, *, llm, dispatcher, config, final_validator, finalize_when, final_payload,
                 final_system_prompt=None, exploration_output_tokens=2048, reasoning_effort=None):
        super().__init__(llm=llm, dispatcher=dispatcher, config=config,
                         final_validator=final_validator)
        self.progress_policy = finalize_when
        self.final_payload = final_payload
        if type(exploration_output_tokens) is not int or exploration_output_tokens < 16:
            raise ValueError("invalid exploration output budget")
        if reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("invalid reasoning effort")
        self.exploration_output_tokens = min(exploration_output_tokens, config.max_output_tokens)
        self.final_system_prompt = final_system_prompt
        self.reasoning_effort = reasoning_effort

    async def run(self, *, system_prompt: str, user_input: str) -> RuntimeResult:
        config = self.config
        started = perf_counter()
        result = RuntimeResult(run_id=str(uuid4()), status="blocked", reason="not_started")
        items = [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_input}]
        seen_ids, evidence_ids = set(), set()
        actions = Counter()
        observed_ratio = 0.0
        pending_final_reason = None

        def trace(event, **data):
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
                    remaining = config.max_total_tokens - result.reported_tokens
                    # The projection is rebuilt from accepted observations, never
                    # from a model summary or unexecuted tool call.
                    final_input = self.final_payload(tuple(result.observations))
                    final_prompt = self.final_system_prompt or system_prompt
                    final_items = [{"role": "system", "content": final_prompt},
                                   {"role": "user", "content": final_input}]
                    explore_size = serialized_bytes(items, schemas)
                    final_size = serialized_bytes(final_items, [])
                    explore_est = estimated_input_tokens(explore_size, observed_ratio)
                    final_est = estimated_input_tokens(final_size, observed_ratio)
                    reserve = final_est + config.max_output_tokens
                    progress_reason = self.progress_policy(tuple(result.observations)) if self.progress_policy else None
                    reason = pending_final_reason or progress_reason
                    if not reason:
                        if not schemas or step == config.max_steps:
                            reason = "last_model_step"
                        elif result.tool_calls >= config.max_tool_calls:
                            reason = "tool_budget_exhausted"
                        elif explore_size > config.max_input_bytes:
                            reason = "input_budget_reserve"
                        elif remaining < explore_est + self.exploration_output_tokens + reserve:
                            reason = "finalization_token_reserve"
                    force_final = bool(reason)
                    request_items = final_items if force_final else items
                    request_schemas = [] if force_final else schemas
                    size = final_size if force_final else explore_size
                    input_est = final_est if force_final else explore_est
                    if size > config.max_input_bytes:
                        return finish("blocked", "input_budget_exhausted")
                    if remaining < input_est + 16:
                        trace("budget_preflight_blocked", remaining_tokens=remaining,
                              estimated_input_tokens=input_est)
                        return finish("blocked", "token_budget_exhausted")
                    phase_limit = config.max_output_tokens if force_final else self.exploration_output_tokens
                    output_limit = min(phase_limit, remaining - input_est)
                    if not force_final:
                        # Preserve the estimated final report budget rather than
                        # spending all remaining output capacity on exploration.
                        output_limit = min(output_limit, remaining - input_est - reserve)
                        if output_limit < 16:
                            return finish("blocked", "token_budget_exhausted")
                    result.steps = step
                    if force_final:
                        trace("finalization_started", reason=reason,
                              projected_input_bytes=size, history_input_bytes=explore_size)
                    trace("llm_started", input_bytes=size, estimated_input_tokens=input_est,
                          finalization_reserve_tokens=reserve, remaining_tokens=remaining,
                          max_output_tokens=output_limit, phase="finalization" if force_final else "exploration")
                    request = ChatRequest(
                        messages=[Message(role="system", content=final_prompt if force_final else system_prompt),
                                  Message(role="user", content=final_input if force_final else user_input)],
                        input_items=request_items, tools=request_schemas or None,
                        tool_choice="auto" if request_schemas else None,
                        max_output_tokens=output_limit, replay_reasoning=config.replay_reasoning,
                        reasoning_effort=self.reasoning_effort)

                    async def call_model():
                        result.llm_attempts += 1
                        return await self.llm.chat(request)

                    def on_retry(number, delay, error):
                        result.usage_complete = False
                        trace("llm_retry", number=number, delay=delay, error_type=type(error).__name__)

                    response = await retry_async(call_model, RetryPolicy(
                        max_retries=config.max_llm_retries, initial_delay=0.25, max_delay=1.0), on_retry)
                    usage = response.total_tokens
                    if usage is None and response.prompt_tokens is not None and response.completion_tokens is not None:
                        usage = response.prompt_tokens + response.completion_tokens
                    if usage is None:
                        result.usage_complete = False
                    else:
                        result.reported_tokens += usage
                    if response.prompt_tokens is not None:
                        observed_ratio = max(observed_ratio, response.prompt_tokens / max(1, size))
                    trace("llm_completed", reported_tokens=usage, prompt_tokens=response.prompt_tokens,
                          completion_tokens=response.completion_tokens,
                          reasoning_tokens=getattr(response, "reasoning_tokens", None),
                          response_status=response.response_status,
                          incomplete_reason=getattr(response, "incomplete_reason", None),
                          tool_call_count=len(response.tool_calls))
                    if usage is None and config.require_reported_usage:
                        return finish("blocked", "token_usage_unavailable")
                    if result.reported_tokens > config.max_total_tokens:
                        result.candidate_output = response.content or None
                        return finish("blocked", "token_budget_exhausted")
                    if response.response_status not in (None, "completed"):
                        result.candidate_output = response.content or None
                        # A text-only truncated draft in exploration may receive
                        # one independent final request, never a partial tool batch.
                        if (not force_final and not response.tool_calls
                                and response.response_status == "incomplete"
                                and response.incomplete_reason == "max_output_tokens"
                                and step < config.max_steps):
                            pending_final_reason = "exploration_output_exhausted"
                            trace("exploration_draft_truncated")
                            continue
                        return finish("blocked", "incomplete_model_response")
                    calls = response.tool_calls
                    if force_final and calls:
                        trace("finalization_violation", tool_call_count=len(calls))
                        return finish("blocked", "tool_calls_during_finalization")
                    if not calls:
                        if not response.content.strip():
                            return finish("blocked", "empty_final_output")
                        try:
                            validated = self.final_validator(response.content, result.observations)
                        except ValueError as exc:
                            result.candidate_output = response.content
                            trace("final_validation_failed", validation_code=getattr(exc, "code", "unknown_validation_error"),
                                  error_type=type(exc).__name__)
                            return finish("blocked", "final_evidence_invalid")
                        canonical = validated.model_dump_json() if hasattr(validated, "model_dump_json") else response.content
                        result.candidate_output = None
                        return finish("completed", "model_finished", canonical)
                    if result.tool_calls + len(calls) > config.max_tool_calls:
                        # Never execute a partial batch; next request is independent
                        # so the unexecuted provider call items are not replayed.
                        pending_final_reason = "tool_batch_budget_exhausted"
                        trace("tool_batch_rejected", proposed_calls=len(calls))
                        continue
                    ids = [c.id for c in calls]
                    if any(not i or len(i) > 256 for i in ids) or len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                        return finish("blocked", "invalid_or_duplicate_call_id")
                    keys = [json.dumps([c.name, c.arguments], sort_keys=True, ensure_ascii=False) for c in calls]
                    next_actions = actions + Counter(keys)
                    if any(n > config.max_repeated_actions for n in next_actions.values()):
                        pending_final_reason = "repeated_action_limit"
                        continue
                    try:
                        replay = replay_items(response)
                    except (ValueError, TypeError, KeyError):
                        return finish("blocked", "invalid_tool_protocol")
                    if serialized_bytes([*items, *replay], schemas) > config.max_input_bytes:
                        pending_final_reason = "input_budget_reserve"
                        continue
                    items.extend(replay)
                    seen_ids.update(ids)
                    actions = next_actions
                    result.tool_calls += len(calls)
                    observations = await self.dispatcher.execute_batch(calls)
                    for obs, call in zip(observations, calls, strict=True):
                        if obs.error_code == "invalid_arguments":
                            schema = next((item for item in schemas if item.get("name") == call.name), None)
                            properties = (schema or {}).get("parameters", {}).get("properties")
                            if isinstance(properties, dict):
                                unknown = set(call.arguments) - properties.keys()
                                if unknown:
                                    # Only short identifier-shaped field names, never values.
                                    safe = sorted(k for k in unknown if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", k))[:10]
                                    trace("unknown_tool_fields", tool=call.name, unknown_fields=safe,
                                          unknown_field_count=len(unknown))
                        data = evidence_data(obs)
                        if data is not None:
                            evidence_id = data.get("registered_evidence_id")
                            if not isinstance(evidence_id, str) or evidence_id in evidence_ids:
                                return finish("failed", "invalid_evidence_registration")
                            evidence_ids.add(evidence_id)
                            obs.evidence_id = evidence_id
                        result.observations.append(obs)
                        trace("tool_result", call_id=obs.call_id, evidence_id=obs.evidence_id,
                              tool=obs.tool_name, success=obs.success, error_code=obs.error_code,
                              duration_ms=obs.duration_ms)
                        items.append(function_output(obs.call_id, model_observation(obs)))
                return finish("blocked", "step_budget_exhausted")
        except TimeoutError:
            return finish("blocked", "run_timeout")
        except Exception as exc:
            trace("runtime_error", error_type=type(exc).__name__)
            return finish("failed", "runtime_error")
