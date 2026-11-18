"""D5-E: strict evaluation of existing Full Execution A/B evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_d5_effectiveness(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    gates = report.get("gates")
    metrics = report.get("metrics")

    if not isinstance(gates, dict) or not isinstance(metrics, dict):
        return {
            "version": "v1.4.9-P4-D5-E",
            "status": "insufficient_evidence",
            "reason": "missing_ab_evaluation",
            "token_savings_confirmed": False,
            "production_ready": False,
        }

    quality_valid = (
        report.get("quality_passed") is True
        and bool(gates)
        and all(value is True for value in gates.values())
    )

    baseline_usage = metrics.get("baseline_model_usage")
    active_usage = metrics.get("active_model_usage")

    def complete_usage(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and value.get("usage_complete") is True
            and value.get("usage_missing_calls") == 0
            and value.get("failed_calls") == 0
            and value.get("in_flight_calls") == 0
            and type(value.get("total_tokens")) is int
            and type(value.get("prompt_tokens")) is int
            and type(value.get("completion_tokens")) is int
            and value["prompt_tokens"] + value["completion_tokens"]
            == value["total_tokens"]
        )

    usage_valid = (
        metrics.get("token_usage_comparable") is True
        and complete_usage(baseline_usage)
        and complete_usage(active_usage)
    )

    saved = metrics.get("llm_tokens_saved")

    if not usage_valid:
        saved = None

    if (
        usage_valid
        and type(saved) is int
        and saved != (baseline_usage["total_tokens"] - active_usage["total_tokens"])
    ):
        usage_valid = False
        saved = None

    if not quality_valid:
        status = "quality_not_proven"
    elif not usage_valid:
        status = "usage_not_comparable"
    elif saved is None:
        status = "insufficient_evidence"
    elif saved < 0:
        status = "token_regression"
    elif saved == 0:
        status = "no_token_saving"
    elif report.get("status") == "passed":
        status = "saving_confirmed"
    else:
        status = "ab_result_not_passed"

    before = metrics.get("baseline_context_chars")
    after = metrics.get("active_context_chars")

    context_chars_saved = (
        before - after if type(before) is int and type(after) is int else None
    )

    return {
        "version": "v1.4.9-P4-D5-E",
        "status": status,
        "ab_status": report.get("status"),
        "baseline_run_id": report.get("baseline_run_id"),
        "active_run_id": report.get("active_run_id"),
        "quality_verified": quality_valid,
        "model_usage_comparable": usage_valid,
        "context_chars_saved": context_chars_saved,
        "llm_tokens_saved": saved,
        "llm_token_saving_ratio": (
            saved / baseline_usage["total_tokens"]
            if (saved is not None and baseline_usage["total_tokens"] > 0)
            else None
        ),
        "failure_reasons": list(report.get("failure_reasons") or []),
        "token_savings_confirmed": (status == "saving_confirmed"),
        "usage_scope": "llm_provider_attempts",
        "excluded_usage": ["embedding", "rerank"],
        "production_ready": False,
    }
