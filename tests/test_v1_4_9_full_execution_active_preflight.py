from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.full_execution_active_preflight import (
    evaluate_active_preflight,
)
from apps.agent_cli.benchmark_active_context_v1_4_9 import (
    REQUIRED_CANARY_GATES,
)


@pytest.fixture
def case():
    root = Path(__file__).resolve().parents[1]

    return BenchmarkCaseLoader(root / "benchmark").load("CASE-006")


def make_canary(revision: str):
    cases = []

    for case_id in (
        "ORDER-001",
        "USER-001",
        "API-001",
    ):
        cases.append(
            {
                "case_id": case_id,
                "status": "passed",
                "completion_gates": {name: True for name in REQUIRED_CANARY_GATES},
            }
        )

    return {
        "version": "v1.4.9-P2-C",
        "repository_revision": revision,
        "status": "passed",
        "safety_tests_passed": True,
        "shadow_consumer_passed": True,
        "active_experiment_eligible": True,
        "production_ready": False,
        "total_cases": 3,
        "passed_cases": 3,
        "cases": cases,
    }


def valid_inputs(case):
    return {
        "case": case,
        "task_id": "TASK-002",
        "approved_revision": case.base_commit_sha,
        "current_revision": case.base_commit_sha,
        "repository_dirty": False,
        "canary_report": make_canary(case.base_commit_sha),
        "canary_sha256": "a" * 64,
    }


def test_valid_preflight_does_not_apply_active(case):
    result = evaluate_active_preflight(**valid_inputs(case))

    assert result.status == "passed"
    assert result.reason == "active_preflight_valid"

    # 最关键的安全断言：
    assert result.applied is False
    assert result.active_execution_enabled is False
    assert result.production_ready is False


def test_unauthorized_task_is_denied(case):
    args = valid_inputs(case)
    args["task_id"] = "TASK-005"

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "task_not_allowlisted"


def test_other_case_is_denied(case):
    args = valid_inputs(case)
    args["case"] = case.model_copy(update={"case_id": "CASE-007"})

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "case_not_allowlisted"


def test_invalid_approved_revision_is_denied(case):
    args = valid_inputs(case)
    args["approved_revision"] = "main"

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "invalid_approved_revision"


def test_stale_repository_revision_is_denied(case):
    args = valid_inputs(case)
    args["current_revision"] = "b" * 40

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "repository_revision_mismatch"


def test_dirty_repository_is_denied(case):
    args = valid_inputs(case)
    args["repository_dirty"] = True

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "repository_dirty"


def test_missing_canary_is_denied(case):
    args = valid_inputs(case)
    args["canary_report"] = None

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "canary_unavailable"


def test_canary_with_failed_gate_is_denied(case):
    args = valid_inputs(case)

    report = deepcopy(args["canary_report"])

    report["cases"][0]["completion_gates"]["required_symbol_coverage"] = False

    args["canary_report"] = report

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "canary_invalid"


def test_canary_from_other_revision_is_denied(case):
    args = valid_inputs(case)

    report = deepcopy(args["canary_report"])
    report["repository_revision"] = "c" * 40

    args["canary_report"] = report

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "canary_invalid"


def test_unreviewed_case_oracle_is_denied(case):
    args = valid_inputs(case)

    expectation = case.expectation.model_copy(
        deep=True,
    )

    expectation.allowed_change_files = [
        path
        for path in expectation.allowed_change_files
        if path != "tests/unit/UsersView.test.ts"
    ]

    args["case"] = case.model_copy(
        deep=True,
        update={"expectation": expectation},
    )

    result = evaluate_active_preflight(**args)

    assert result.status == "denied"
    assert result.reason == "case_oracle_not_reviewed"


@pytest.mark.asyncio
async def test_cli_preflight_does_not_enter_execution(
    monkeypatch,
):
    from apps.agent_cli import (
        benchmark_v1_1_run as cli,
    )

    calls = []

    monkeypatch.setattr(
        cli,
        "_parse_args",
        lambda: SimpleNamespace(
            case_id="CASE-006",
            context_active_preflight=True,
            active_task_id="TASK-002",
            approved_revision="a" * 40,
            context_shadow=False,
            refresh_upstream=False,
        ),
    )

    def fake_preflight(**kwargs):
        calls.append(kwargs["args"].case_id)

    monkeypatch.setattr(
        cli,
        "_run_active_preflight",
        fake_preflight,
    )

    def forbidden_execution(**kwargs):
        pytest.fail("Preflight 不应创建 Benchmark Runner")

    monkeypatch.setattr(
        cli,
        "create_benchmark_runner",
        forbidden_execution,
    )

    await cli.main()

    assert calls == ["CASE-006"]
