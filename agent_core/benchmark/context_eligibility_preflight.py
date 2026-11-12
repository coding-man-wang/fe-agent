"""D3: conservative, read-only file-drop eligibility analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from agent_core.change_surface.model import ResolvedChangeSurface

SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".vue", ".mjs", ".cjs"}


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _classify_path(path: str) -> str:
    """Conservative category hint, not a repository-index proof."""
    normalized = _normalize(path).casefold()
    parts = PurePosixPath(normalized).parts
    name = parts[-1]

    if (
        any(p in {"test", "tests", "__tests__"} for p in parts[:-1])
        or ".test." in name
        or ".spec." in name
    ):
        return "test"

    if PurePosixPath(name).suffix in SOURCE_SUFFIXES:
        return "possible_source"

    return "unknown"


def analyze_context_eligibility(
    *,
    context_files: Sequence[str],
    surface: ResolvedChangeSurface,
    obligation_evidence_files: Sequence[str] = (),
    verification_strategy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze only possible file-drop eligibility.

    Does NOT prove a candidate can be dropped.
    Does NOT run Retrieval, Rerank, or LLM.
    Does NOT grant write permission.
    """
    paths = [_normalize(p) for p in context_files]

    def inconclusive(reason: str) -> dict[str, Any]:
        return {
            "version": "v1.4.9-P4-B-D3",
            "status": "inconclusive",
            "reason": reason,
            "context_files": paths,
            "historical_advisory_only": True,
            "applied": False,
        }

    if not paths or any(not p for p in paths):
        return inconclusive("missing_context_evidence")

    if len(paths) != len(set(paths)):
        return inconclusive("duplicate_context_paths")

    writable_tests: list[str] = []

    if verification_strategy is not None:
        value = verification_strategy.get("writable_test_files")
        if not isinstance(value, list) or any(not isinstance(p, str) for p in value):
            return inconclusive("invalid_verification_strategy")
        writable_tests = value

    protected = {
        _normalize(p)
        for p in (
            *surface.implementation_targets,
            *surface.tests,
            *surface.dependencies,
            *surface.references,
            *obligation_evidence_files,
            *writable_tests,
        )
    }

    protected_in_context = []
    non_source_files = []
    possible_source_files = []
    unknown_files = []

    for path in paths:
        if path in protected:
            protected_in_context.append(path)
            continue

        category = _classify_path(path)

        if category == "test":
            non_source_files.append(path)
        elif category == "possible_source":
            possible_source_files.append(path)
        else:
            unknown_files.append(path)

    if possible_source_files:
        status = "candidate_possible"
        reason = "requires_graph_rerank_and_guard_validation"
    elif unknown_files:
        status = "inconclusive"
        reason = "unclassified_context_files"
    else:
        status = "no_source_drop_candidate"
        reason = "no_unprotected_source_in_final_context"

    return {
        "version": "v1.4.9-P4-B-D3",
        "status": status,
        "reason": reason,
        "task_id": surface.task_id,
        "context_files": paths,
        "protected_in_context": protected_in_context,
        "non_source_files": non_source_files,
        "possible_source_files": possible_source_files,
        "unknown_files": unknown_files,
        "surface_missing_from_context": sorted(protected - set(paths)),
        "historical_advisory_only": True,
        "requires_shadow_validation": True,
        "applied": False,
        "production_ready": False,
    }


def recommend_experiment(report: dict) -> dict:
    """给出实验启动建议，不替代 Runtime Guard。"""

    status = report.get("status")

    decisions = {
        "no_source_drop_candidate": (
            "skip_recommended",
            "No eligible source file for current drop strategies",
        ),
        "candidate_possible": (
            "live_validation_required",
            "Fresh Shadow and Guard validation required",
        ),
        "inconclusive": (
            "manual_review_required",
            "Insufficient evidence for eligibility decision",
        ),
    }

    decision, reason = decisions.get(
        status,
        (
            "manual_review_required",
            "Unknown eligibility status",
        ),
    )

    return {
        "version": "v1.4.9-P4-B-D3-C",
        "eligibility_status": status,
        "decision": decision,
        "reason": reason,
        "experiment_executed": False,
        "applied": False,
        "production_ready": False,
    }
