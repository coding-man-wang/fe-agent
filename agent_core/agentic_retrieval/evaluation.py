"""Post-run metrics. Expected paths are never passed to the Controller."""
from __future__ import annotations

from collections import defaultdict
from math import ceil
from statistics import mean

from pydantic import Field

from .contracts import Contract, PathText
from .controller import AgenticResult


class EvaluationOracle(Contract):
    # Optional offline expected evidence, not a retrieval policy.
    required_paths: tuple[PathText, ...] = Field(default=())
    expected_blocked: bool = False


def evaluate(result: AgenticResult, oracle: EvaluationOracle | None = None) -> dict:
    actual = set(result.evidence_paths)
    expected = set(oracle.required_paths) if oracle is not None else set()
    cited = {citation.path for answer in result.report.answers for citation in answer.citations} if result.report is not None else set()
    completed = result.runtime.status == "completed" and result.report is not None
    recall = len(actual & expected) / len(expected) if expected else None
    cited_recall = len(cited & expected) / len(expected) if expected else None
    no_progress_calls = sum(event["status"] in {"no_matches", "no_new_candidates", "admission_budget_exhausted"}
                            for event in result.retrieval_events)
    false_admission = (completed and (oracle.expected_blocked or not expected <= cited)) if oracle is not None else None
    return {"mode": result.mode, "revision": result.revision, "task_id": result.scope.task_id,
            "completed": completed, "status": result.runtime.status, "reason": result.runtime.reason,
            "validation_code": result.validation_code,
            "required_file_recall": recall, "cited_required_file_recall": cited_recall,
            "false_admission": false_admission,
            "criteria_with_validated_answers": len(result.report.answers) if result.report is not None else 0,
            "criteria_total": len(result.scope.criteria),
            "admitted_files": len(result.admitted_paths), "evidence_files": len(actual),
            "new_evidence_files": len(actual - set(result.initial_paths)),
            "steps": result.runtime.steps, "llm_attempts": result.runtime.llm_attempts,
            "tool_calls": result.runtime.tool_calls, "no_progress_search_calls": no_progress_calls,
            "retrieval_calls": len(result.retrieval_events),
            "reported_tokens": result.runtime.reported_tokens,
            "usage_complete": result.runtime.usage_complete, "runtime_ms": result.runtime.duration_ms}


def summarize(metrics: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in metrics:
        grouped[item["mode"]].append(item)
    summary = {}
    for mode, rows in grouped.items():
        durations = sorted(row["runtime_ms"] for row in rows)
        known_tokens = [row["reported_tokens"] for row in rows if row["usage_complete"]]
        recalls = [row["required_file_recall"] for row in rows if row["required_file_recall"] is not None]
        false_admissions = [row["false_admission"] for row in rows if row["false_admission"] is not None]
        summary[mode] = {"runs": len(rows), "completion_rate": mean(row["completed"] for row in rows),
                         "mean_required_file_recall": mean(recalls) if recalls else None,
                         "false_admission_rate": mean(false_admissions) if false_admissions else None,
                         "mean_reported_tokens_complete_usage": mean(known_tokens) if known_tokens else None,
                         "complete_usage_runs": len(known_tokens),
                         "mean_runtime_ms": mean(durations), "p95_runtime_ms": durations[ceil(.95 * len(durations)) - 1],
                         "mean_tool_calls": mean(row["tool_calls"] for row in rows)}
    return summary
