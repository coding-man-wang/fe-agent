"""Provider attempt metering without prompts, outputs, credentials or estimates."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, ChatResponse


class MeteredLLM(LLM):
    """Wrap the provider BELOW RetryingLLM to count every provider attempt.

    Marks partition calls by their start order; finish pending calls before taking
    a final task/run snapshot. Latency is summed provider-call time, not wall time.
    Failed attempts may have been billed, so their unknown usage makes full totals
    unavailable. This deliberately excludes embedding and rerank API calls.
    """

    def __init__(self, llm: LLM, *, clock: Callable[[], float] = perf_counter):
        self.llm = llm
        self.model = getattr(llm, "model", None)
        self._clock = clock
        self._calls: list[dict] = []

    def mark(self) -> int:
        return len(self._calls)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        entry = {
            "status": "pending",
            "latency_ms": 0.0,
            "model": self.model,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "error_type": None,
        }
        self._calls.append(entry)
        started = self._clock()
        try:
            response = await self.llm.chat(request)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(response, field, None)
                entry[field] = value if type(value) is int and value >= 0 else None
            entry["model"] = getattr(response, "model", None) or self.model
            entry["status"] = "succeeded"
            return response
        except BaseException as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            entry["latency_ms"] = max(0.0, (self._clock() - started) * 1000)

    def snapshot(self, start_index: int = 0) -> dict:
        if type(start_index) is not int or not 0 <= start_index <= len(self._calls):
            raise ValueError("Invalid model usage mark")
        calls = self._calls[start_index:]

        def complete(call):
            return (
                call["status"] == "succeeded"
                and all(
                    call[key] is not None
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                and call["prompt_tokens"] + call["completion_tokens"]
                == call["total_tokens"]
            )

        missing = sum(not complete(call) for call in calls)
        report = {
            "schema_version": 1,
            "scope": "llm_provider_attempts",
            "excluded_usage": ["embedding", "rerank"],
            "calls": len(calls),
            "succeeded_calls": sum(call["status"] == "succeeded" for call in calls),
            "failed_calls": sum(call["status"] == "failed" for call in calls),
            "in_flight_calls": sum(call["status"] == "pending" for call in calls),
            "latency_ms": round(sum(call["latency_ms"] for call in calls), 3),
            "usage_complete": bool(calls) and missing == 0,
            "usage_missing_calls": missing,
            "models": sorted(
                {call["model"] for call in calls if isinstance(call["model"], str)}
            ),
            "error_types": sorted(
                {call["error_type"] for call in calls if call["error_type"]}
            ),
        }
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            known = sum(call[field] for call in calls if call[field] is not None)
            report["known_" + field] = known
            report[field] = known if missing == 0 else None
        return report
