from copy import deepcopy

import pytest

from agent_core.benchmark.full_execution_ab import (
    compare_full_execution_ab,
    IDENTITY_KEYS,
)
from agent_core.benchmark.execution_model import FullExecutionRunRecord


def usage(total):
    return {
        "scope": "llm_provider_attempts",
        "usage_complete": True,
        "calls": 2,
        "succeeded_calls": 2,
        "failed_calls": 0,
        "in_flight_calls": 0,
        "usage_missing_calls": 0,
        "prompt_tokens": total - 10,
        "completion_tokens": 10,
        "total_tokens": total,
        "models": ["same-model"],
        "latency_ms": 12,
    }


@pytest.fixture
def arms():
    baseline = {
        "run_id": "baseline-run",
        "baseline_run_id": "frozen-prepared-run",
        "case_id": "CASE-006",
        "base_commit_sha": "a" * 40,
        "final_commit_sha": "b" * 40,
        "final_workspace_root": "D:/runs/baseline",
        "status": "completed",
        "error": None,
        "context_policy_mode": "baseline",
        "context_experiment_identity": {
            key: "a" * (40 if key == "approved_revision" else 64)
            for key in IDENTITY_KEYS
        },
        "model_usage": usage(200),
        "tasks": [
            {
                "task_id": "TASK-002",
                "task_type": "code_change",
                "status": "completed",
                "verification_status": "succeeded",
                "completion_status": "completed",
                "selected_context_files": [
                    "src/target.ts",
                    "src/evidence.ts",
                    "src/noise.ts",
                ],
                "obligation_evidence_files": ["src/evidence.ts"],
                "workspace_id": "workspace-b",
                "expected_execution_obligation_ids": ["obligation-1"],
                "satisfied_execution_obligation_ids": ["obligation-1"],
                "unresolved_execution_obligation_ids": [],
            }
        ],
        "evaluation": {
            "status": "passed",
            "expected_change_files": ["src/target.ts"],
            "allowed_change_files": ["tests"],
            "actual_change_files": ["src/target.ts"],
            "missed_change_files": [],
            "unexpected_change_files": [],
            "forbidden_change_hits": [],
            "failed_task_ids": [],
            "failure_reasons": [],
            "missed_impact_files": [],
            "missing_impact_verification_command_ids": [],
            "verification_passed": True,
            "hidden_acceptance_passed": True,
            "impact_coverage_passed": True,
            "impact_verification_passed": True,
        },
        "hidden_acceptance": {
            "proposal_id": "a" * 64,
            "task_id": "hidden",
            "status": "passed",
            "checks": [{"name": "functional-oracle", "status": "passed"}],
        },
    }
    active = deepcopy(baseline)
    active.update(
        run_id="active-run",
        final_workspace_root="D:/runs/active",
        context_policy_mode="active_experiment",
        model_usage=usage(150),
    )
    active["tasks"][0]["workspace_id"] = "workspace-a"
    active["tasks"][0]["selected_context_files"] = ["src/target.ts", "src/evidence.ts"]
    active["tasks"][0]["context_guarded_active"] = {
        "status": "active",
        "applied": True,
        "active_execution_enabled": True,
        "production_ready": False,
        "audit_recorded": True,
        "snapshot_matched": True,
        "task_id": "TASK-002",
        "case_id": "CASE-006",
        "repository_revision": "a" * 40,
        "canary_sha256": "a" * 64,
        "consumed_bundle_sha256": "c" * 64,
        "mandatory_context_complete": True,
        "retained_symbol_evidence": True,
        "metadata_unchanged": True,
        "scope_unchanged": True,
        "baseline_files": ["src/target.ts", "src/evidence.ts", "src/noise.ts"],
        "candidate_files": ["src/target.ts", "src/evidence.ts"],
        "removed_files": ["src/noise.ts"],
        "baseline_chars": 1000,
        "candidate_chars": 800,
    }
    return baseline, active


def compare(arms):
    return compare_full_execution_ab(baseline=arms[0], active=arms[1])


def test_successful_pair_has_actual_usage_and_keeps_production_closed(arms):
    report = compare(arms)
    assert report["status"] == "passed"
    assert report["quality_passed"] is True
    assert report["metrics"]["llm_tokens_saved"] == 50
    assert report["metrics"]["llm_token_saving_ratio"] == 0.25
    assert report["production_ready"] is False
    assert report["production_active_enabled"] is False


@pytest.mark.parametrize("key", IDENTITY_KEYS)
def test_frozen_input_mismatch_cannot_pass(arms, key):
    arms[1]["context_experiment_identity"][key] = "f" * (
        40 if key == "approved_revision" else 64
    )
    assert compare(arms)["status"] == "failed"
    assert "frozen_identity_matched" in compare(arms)["failure_reasons"]
    assert compare(arms)["metrics"]["llm_tokens_saved"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "baseline-run"),
        ("final_workspace_root", "d:\\runs\\BASELINE\\"),
        ("final_workspace_root", "D:/runs/baseline/nested"),
    ],
)
def test_reused_run_or_workspace_is_rejected(arms, field, value):
    arms[1][field] = value
    assert compare(arms)["gates"]["independent_runs"] is False


def test_same_content_addressed_workspace_id_is_safe_in_independent_run_roots(arms):
    arms[1]["tasks"][0]["workspace_id"] = arms[0]["tasks"][0]["workspace_id"]
    assert compare(arms)["gates"]["independent_runs"] is True
    assert compare(arms)["status"] == "passed"


@pytest.mark.parametrize("arm", [0, 1])
def test_bad_baseline_or_active_quality_cannot_pass(arms, arm):
    arms[arm]["evaluation"]["hidden_acceptance_passed"] = False
    assert compare(arms)["status"] == "failed"


def test_summary_success_cannot_hide_failed_actual_hidden_check(arms):
    arms[1]["hidden_acceptance"]["checks"][0]["status"] = "failed"
    assert compare(arms)["status"] == "failed"


@pytest.mark.parametrize("actual", [[], ["src/target.ts", "src/forbidden.ts"]])
def test_rechecks_expected_and_allowed_scope_instead_of_trusting_status(arms, actual):
    arms[1]["evaluation"]["actual_change_files"] = actual
    assert compare(arms)["status"] == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "mandatory_context_complete",
        "retained_symbol_evidence",
        "metadata_unchanged",
        "scope_unchanged",
        "snapshot_matched",
        "audit_recorded",
    ],
)
def test_missing_active_safety_evidence_fails_closed(arms, field):
    del arms[1]["tasks"][0]["context_guarded_active"][field]
    assert compare(arms)["status"] == "failed"


def test_fallback_is_not_actual_active(arms):
    arms[1]["tasks"][0]["context_guarded_active"].update(
        status="fallback", applied=False
    )
    assert compare(arms)["gates"]["actual_active_consumed"] is False


def test_missing_obligation_or_mandatory_file_fails(arms):
    arms[1]["tasks"][0]["satisfied_execution_obligation_ids"] = []
    assert compare(arms)["status"] == "failed"
    arms[1]["tasks"][0]["satisfied_execution_obligation_ids"] = ["obligation-1"]
    arms[0]["tasks"][0]["obligation_evidence_files"] = ["src/noise.ts"]
    assert compare(arms)["gates"]["context_files_verified"] is False


def test_runtime_cannot_silently_shrink_obligations_or_expand_oracle(arms):
    arms[1]["tasks"][0]["expected_execution_obligation_ids"] = []
    assert compare(arms)["gates"]["same_task_plan"] is False
    arms[1]["evaluation"]["allowed_change_files"].append("src")
    assert compare(arms)["gates"]["oracle_matched"] is False


@pytest.mark.parametrize("arm", [0, 1])
def test_unknown_usage_is_inconclusive_and_never_reports_savings(arms, arm):
    arms[arm]["model_usage"] = {}
    report = compare(arms)
    assert report["status"] == "inconclusive"
    assert report["quality_passed"] is True
    assert report["metrics"]["llm_tokens_saved"] is None


def test_token_regression_is_reported(arms):
    arms[1]["model_usage"] = usage(300)
    report = compare(arms)
    assert report["status"] == "failed"
    assert "llm_token_regression" in report["failure_reasons"]


def test_actual_model_version_mismatch_makes_usage_incomparable(arms):
    arms[1]["model_usage"]["models"] = ["different-version"]
    assert compare(arms)["status"] == "inconclusive"


def test_missing_and_duplicate_task_records_fail_closed(arms):
    arms[1]["tasks"].append(deepcopy(arms[1]["tasks"][0]))
    assert compare(arms)["status"] == "failed"
    assert compare_full_execution_ab(baseline={}, active={})["status"] == "failed"


def test_accepts_pydantic_records(arms):
    baseline, active = (FullExecutionRunRecord.model_validate(value) for value in arms)
    assert (
        compare_full_execution_ab(baseline=baseline, active=active)["status"]
        == "passed"
    )
