"""Fail-closed comparison of independently executed Full Execution A/B arms."""

from __future__ import annotations

from collections.abc import Mapping
import posixpath
import re

IDENTITY_KEYS = (
    "case_sha256",
    "prepared_input_sha256",
    "upstream_artifacts_sha256",
    "model_config_sha256",
    "canary_sha256",
    "approved_revision",
)
QUALITY_FLAGS = (
    "mandatory_context_complete",
    "retained_symbol_evidence",
    "metadata_unchanged",
    "scope_unchanged",
)


def _dict(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value):
    return isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value)


def _sha(value, length=64):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None
    )


def _path(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    # Records may be compared on another OS. Conservative Windows-style aliases
    # are treated as the same path; this does not access or trust live directories.
    return posixpath.normpath(value.replace("\\", "/")).casefold().rstrip("/")


def _tasks(run):
    tasks = run.get("tasks")
    if not isinstance(tasks, list) or any(not isinstance(t, Mapping) for t in tasks):
        return {}
    result = {
        t.get("task_id"): dict(t) for t in tasks if isinstance(t.get("task_id"), str)
    }
    return result if len(result) == len(tasks) else {}


def _covers(path, scopes):
    path = path.replace("\\", "/").strip("/")
    return any(
        path == scope.replace("\\", "/").strip("/")
        or path.startswith(scope.replace("\\", "/").strip("/") + "/")
        for scope in scopes
    )


def _quality(run):
    evaluation = _dict(run.get("evaluation"))
    hidden = _dict(run.get("hidden_acceptance"))
    checks = hidden.get("checks")
    tasks = _tasks(run)
    if (
        run.get("status") != "completed"
        or run.get("error")
        or evaluation.get("status") != "passed"
        or any(
            evaluation.get(key) is not True
            for key in (
                "verification_passed",
                "hidden_acceptance_passed",
                "impact_coverage_passed",
                "impact_verification_passed",
            )
        )
        or any(
            evaluation.get(key) != []
            for key in (
                "missed_change_files",
                "unexpected_change_files",
                "forbidden_change_hits",
                "failed_task_ids",
                "failure_reasons",
                "missed_impact_files",
                "missing_impact_verification_command_ids",
            )
        )
        or hidden.get("status") != "passed"
        or not isinstance(checks, list)
        or not checks
        or any(_dict(check).get("status") != "passed" for check in checks)
        or not tasks
    ):
        return False
    expected = evaluation.get("expected_change_files")
    allowed = evaluation.get("allowed_change_files")
    actual = evaluation.get("actual_change_files")
    if (
        not all(_strings(value) for value in (expected, allowed, actual))
        or not expected
    ):
        return False
    if any(not _covers(path, [*expected, *allowed]) for path in actual):
        return False
    if any(not any(_covers(path, [scope]) for path in actual) for scope in expected):
        return False
    for task in tasks.values():
        if task.get("status") == "failed" or task.get("error"):
            return False
        expected_ids = task.get("expected_execution_obligation_ids", [])
        satisfied = task.get("satisfied_execution_obligation_ids", [])
        if (
            not _strings(expected_ids)
            or not _strings(satisfied)
            or not set(expected_ids) <= set(satisfied)
            or task.get("unresolved_execution_obligation_ids", []) != []
        ):
            return False
        if (
            task.get("status") == "completed"
            and task.get("task_type") in {"code_change", "test", "verification"}
            and task.get("verification_status") != "succeeded"
        ):
            return False
    task = tasks.get("TASK-002", {})
    return (
        task.get("status") == "completed"
        and task.get("verification_status") == "succeeded"
        and task.get("completion_status") == "completed"
    )


def _usage_complete(usage):
    counts = [
        usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    return (
        usage.get("scope") == "llm_provider_attempts"
        and usage.get("usage_complete") is True
        and type(usage.get("calls")) is int
        and usage["calls"] > 0
        and usage.get("usage_missing_calls") == 0
        and usage.get("failed_calls") == 0
        and usage.get("in_flight_calls") == 0
        and usage.get("succeeded_calls") == usage["calls"]
        and all(type(count) is int and count >= 0 for count in counts)
        and counts[0] + counts[1] == counts[2]
    )


def compare_full_execution_ab(*, baseline, active) -> dict:
    """Accept records or JSON summaries. Missing evidence never counts as a pass.

    A passing result is experimental evidence for this pair, not authorization
    to enable production Active. Token counts cover LLM provider attempts only.
    """
    baseline, active = _dict(baseline), _dict(active)
    b_identity = _dict(baseline.get("context_experiment_identity"))
    a_identity = _dict(active.get("context_experiment_identity"))
    b_tasks, a_tasks = _tasks(baseline), _tasks(active)
    b_task, a_task = b_tasks.get("TASK-002", {}), a_tasks.get("TASK-002", {})
    report = _dict(a_task.get("context_guarded_active"))
    b_root, a_root = _path(baseline.get("final_workspace_root")), _path(
        active.get("final_workspace_root")
    )
    b_workspaces = {
        t.get("workspace_id")
        for t in b_tasks.values()
        if isinstance(t.get("workspace_id"), str) and t["workspace_id"]
    }
    a_workspaces = {
        t.get("workspace_id")
        for t in a_tasks.values()
        if isinstance(t.get("workspace_id"), str) and t["workspace_id"]
    }
    gates = {
        "frozen_identity_matched": (
            bool(b_identity)
            and b_identity == a_identity
            and all(
                _sha(b_identity.get(key), 40 if key == "approved_revision" else 64)
                for key in IDENTITY_KEYS
            )
            and baseline.get("case_id") == active.get("case_id") == "CASE-006"
            and baseline.get("base_commit_sha")
            == active.get("base_commit_sha")
            == b_identity.get("approved_revision")
            and bool(baseline.get("baseline_run_id"))
            and baseline.get("baseline_run_id") == active.get("baseline_run_id")
        ),
        "independent_runs": (
            bool(baseline.get("run_id"))
            and bool(active.get("run_id"))
            and baseline["run_id"] != active["run_id"]
            and bool(b_root)
            and bool(a_root)
            and b_root != a_root
            and not b_root.startswith(a_root + "/")
            and not a_root.startswith(b_root + "/")
            and bool(b_workspaces)
            and bool(a_workspaces)
            # Workspace IDs hash task/proposal/base, not the run. Identical
            # successful patches legitimately share IDs in distinct run roots.
        ),
        "arm_modes_valid": (
            baseline.get("context_policy_mode") == "baseline"
            and active.get("context_policy_mode") == "active_experiment"
            and not any(
                _dict(t.get("context_guarded_active")).get("applied") is True
                for t in b_tasks.values()
            )
            and not any(
                _dict(t.get("context_guarded_active")).get("applied") is True
                for key, t in a_tasks.items()
                if key != "TASK-002"
            )
        ),
        "same_task_plan": (
            bool(b_tasks)
            and b_tasks.keys() == a_tasks.keys()
            and all(
                b_tasks[key].get("task_type") == a_tasks[key].get("task_type")
                and b_tasks[key].get("expected_execution_obligation_ids", [])
                == a_tasks[key].get("expected_execution_obligation_ids", [])
                for key in b_tasks
            )
        ),
        "oracle_matched": all(
            _dict(baseline.get("evaluation")).get(key, [])
            == _dict(active.get("evaluation")).get(key, [])
            for key in (
                "expected_change_files",
                "allowed_change_files",
                "expected_impact_files",
                "required_impact_verification_command_ids",
            )
        ),
        "baseline_quality_passed": _quality(baseline),
        "active_quality_passed": _quality(active),
        "actual_active_consumed": (
            report.get("status") == "active"
            and report.get("applied") is True
            and report.get("active_execution_enabled") is True
            and report.get("production_ready") is False
            and report.get("audit_recorded") is True
            and report.get("snapshot_matched") is True
            and report.get("task_id") == "TASK-002"
            and report.get("case_id") == active.get("case_id")
            and report.get("repository_revision") == active.get("base_commit_sha")
            and report.get("canary_sha256") == a_identity.get("canary_sha256")
            and _sha(report.get("consumed_bundle_sha256"))
        ),
        "context_quality_preserved": all(
            report.get(flag) is True for flag in QUALITY_FLAGS
        ),
    }
    before, after, removed = (
        report.get(key)
        for key in ("baseline_files", "candidate_files", "removed_files")
    )
    obligations = b_task.get("obligation_evidence_files", [])
    gates["context_files_verified"] = (
        all(_strings(value) for value in (before, after, removed, obligations))
        and bool(removed)
        and len(set(before)) == len(before)
        and len(set(after)) == len(after)
        and list(before) == b_task.get("selected_context_files")
        and list(after) == a_task.get("selected_context_files")
        and list(after) == [p for p in before if p not in removed]
        and set(removed) == set(before) - set(after)
        and set(obligations) <= set(after)
    )
    before_chars, after_chars = report.get("baseline_chars"), report.get(
        "candidate_chars"
    )
    gates["context_reduced"] = (
        type(before_chars) is int
        and type(after_chars) is int
        and 0 < after_chars < before_chars
    )
    b_usage, a_usage = _dict(baseline.get("model_usage")), _dict(
        active.get("model_usage")
    )
    usage_available = _usage_complete(b_usage) and _usage_complete(a_usage)
    models_matched = (
        _strings(b_usage.get("models"))
        and _strings(a_usage.get("models"))
        and bool(b_usage.get("models"))
        and b_usage["models"] == a_usage["models"]
    )
    usage_available = (
        usage_available
        and models_matched
        and all(
            gates[key]
            for key in (
                "frozen_identity_matched",
                "independent_runs",
                "arm_modes_valid",
                "same_task_plan",
                "oracle_matched",
                "actual_active_consumed",
            )
        )
    )
    quality_passed = all(gates.values())
    reasons = [name for name, passed in gates.items() if not passed]
    savings = None
    if usage_available:
        savings = b_usage["total_tokens"] - a_usage["total_tokens"]
        if savings < 0:
            reasons.append("llm_token_regression")
    else:
        reasons.append("model_usage_unavailable_or_not_comparable")
    status = (
        "failed"
        if not quality_passed or (savings is not None and savings < 0)
        else ("passed" if usage_available else "inconclusive")
    )
    return {
        "version": "v1.4.9-P4-B-D2-AB",
        "status": status,
        "baseline_run_id": baseline.get("run_id"),
        "active_run_id": active.get("run_id"),
        "gates": gates,
        "quality_passed": quality_passed,
        "failure_reasons": reasons,
        "metrics": {
            "baseline_context_chars": before_chars,
            "active_context_chars": after_chars,
            "baseline_context_files": before,
            "active_context_files": after,
            "baseline_model_usage": b_usage,
            "active_model_usage": a_usage,
            "token_usage_comparable": usage_available,
            "llm_tokens_saved": savings,
            "llm_token_saving_ratio": (
                savings / b_usage["total_tokens"]
                if savings is not None and b_usage["total_tokens"] > 0
                else None
            ),
            "baseline_actual_change_files": _dict(baseline.get("evaluation")).get(
                "actual_change_files"
            ),
            "active_actual_change_files": _dict(active.get("evaluation")).get(
                "actual_change_files"
            ),
            "usage_scope": "llm_provider_attempts",
            "excluded_usage": ["embedding", "rerank"],
        },
        "production_ready": False,
        "production_active_enabled": False,
    }
