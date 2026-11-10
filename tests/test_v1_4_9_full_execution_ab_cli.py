from types import SimpleNamespace
import sys

import pytest

from apps.agent_cli import benchmark_v1_1_run as cli
from test_v1_4_9_guarded_preview_cli import args, fake_prepared_runtime


def test_active_and_ab_require_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    parsed = cli._parse_args()
    assert parsed.context_guarded_active is parsed.context_ab is False


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["context_guarded_active", "context_ab"])
async def test_active_modes_cannot_be_combined_with_preview(monkeypatch, flag):
    monkeypatch.setattr(cli, "_parse_args", lambda: args(**{flag: True}))
    monkeypatch.setattr(
        cli,
        "create_benchmark_runner",
        lambda **_: pytest.fail("must reject before models"),
    )
    with pytest.raises(SystemExit, match="不能与"):
        await cli.main()


def setup_ab(monkeypatch, *, baseline_status="passed", compare_status="passed"):
    calls, captured, passed, observer = fake_prepared_runtime(monkeypatch)
    monkeypatch.setattr(
        cli, "_parse_args", lambda: args(context_guarded_preview=False, context_ab=True)
    )
    identity = {"approved_revision": "a" * 40, "canary_sha256": "b" * 64}
    identity_calls = []

    def identify(**kwargs):
        identity_calls.append(kwargs)
        return dict(identity)

    monkeypatch.setattr(cli, "_experiment_identity", identify)
    arms = []
    records = {}

    async def execute(**kwargs):
        arms.append(kwargs)
        mode = kwargs["mode"]
        record = SimpleNamespace(
            run_id=mode,
            status="completed",
            evaluation=SimpleNamespace(
                status=baseline_status if mode == "baseline" else "passed"
            ),
        )
        records[mode] = record
        return record

    monkeypatch.setattr(cli, "_execute_prepared_run", execute)
    comparisons, reports = [], []

    def compare(**kwargs):
        comparisons.append(kwargs)
        return {"status": compare_status, "quality_passed": True, "failure_reasons": []}

    monkeypatch.setattr(cli, "compare_full_execution_ab", compare)
    monkeypatch.setattr(
        cli, "_write_ab_report", lambda _, report: reports.append(report)
    )
    return SimpleNamespace(
        calls=calls,
        passed=passed,
        observer=observer,
        identity=identity,
        identity_calls=identity_calls,
        arms=arms,
        records=records,
        comparisons=comparisons,
        reports=reports,
    )


@pytest.mark.asyncio
async def test_ab_freezes_inputs_runs_independent_arms_and_compares(monkeypatch):
    state = setup_ab(monkeypatch)
    await cli.main()
    assert [arm["mode"] for arm in state.arms] == ["baseline", "active_experiment"]
    assert state.arms[0]["identity"] == state.arms[1]["identity"] == state.identity
    assert "preflight" not in state.arms[0]
    assert state.arms[1]["preflight"] is state.passed
    assert state.arms[1]["observer"] is state.observer
    assert state.calls.count("preflight") == 3
    assert len(state.identity_calls) == 3
    assert state.comparisons == [
        {
            "baseline": state.records["baseline"],
            "active": state.records["active_experiment"],
        }
    ]
    assert state.reports[-1]["status"] == "passed"


@pytest.mark.asyncio
async def test_failed_baseline_never_starts_active(monkeypatch):
    state = setup_ab(monkeypatch, baseline_status="failed")
    with pytest.raises(SystemExit):
        await cli.main()
    assert [arm["mode"] for arm in state.arms] == ["baseline"]
    assert state.reports[-1]["reason"] == "baseline_failed"
    assert not state.comparisons


@pytest.mark.asyncio
async def test_baseline_initialization_failure_keeps_failed_report(monkeypatch):
    state = setup_ab(monkeypatch)

    async def unavailable(**kwargs):
        state.arms.append(kwargs)
        raise RuntimeError("cannot start baseline")

    monkeypatch.setattr(cli, "_execute_prepared_run", unavailable)
    with pytest.raises(SystemExit):
        await cli.main()
    assert [arm["mode"] for arm in state.arms] == ["baseline"]
    assert state.reports[-1]["reason"] == "baseline_execution_unavailable"
    assert not state.comparisons


@pytest.mark.asyncio
async def test_changed_frozen_inputs_stop_before_active(monkeypatch):
    state = setup_ab(monkeypatch)
    values = iter([state.identity, {**state.identity, "canary_sha256": "c" * 64}])
    monkeypatch.setattr(cli, "_experiment_identity", lambda **_: next(values))
    with pytest.raises(SystemExit):
        await cli.main()
    assert len(state.arms) == 1
    assert state.reports[-1]["reason"] == "admission_or_inputs_changed"


@pytest.mark.asyncio
async def test_drift_during_active_invalidates_comparison(monkeypatch):
    state = setup_ab(monkeypatch)
    values = iter(
        [state.identity, state.identity, {**state.identity, "canary_sha256": "c" * 64}]
    )
    monkeypatch.setattr(cli, "_experiment_identity", lambda **_: next(values))
    with pytest.raises(SystemExit):
        await cli.main()
    assert len(state.arms) == 2
    assert state.reports[-1]["status"] == "failed"
    assert "frozen_inputs_changed_during_active" in state.reports[-1]["failure_reasons"]


@pytest.mark.asyncio
async def test_revoked_canary_after_active_invalidates_result(monkeypatch):
    state = setup_ab(monkeypatch)
    checks = []

    def preflight(**kwargs):
        checks.append(kwargs)
        if len(checks) == 3:
            raise SystemExit(2)
        return state.passed

    monkeypatch.setattr(cli, "_checked_active_preflight", preflight)
    with pytest.raises(SystemExit):
        await cli.main()
    assert len(checks) == 3
    assert len(state.arms) == 2
    assert state.reports[-1]["status"] == "failed"
    assert "frozen_inputs_changed_during_active" in state.reports[-1]["failure_reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [True, False])
async def test_standalone_active_requires_actual_candidate_consumption(
    monkeypatch, applied
):
    state = setup_ab(monkeypatch)
    monkeypatch.setattr(
        cli,
        "_parse_args",
        lambda: args(context_guarded_preview=False, context_guarded_active=True),
    )

    async def execute(**kwargs):
        state.arms.append(kwargs)
        return SimpleNamespace(
            run_id="active-run",
            status="completed",
            evaluation=SimpleNamespace(status="passed"),
            model_dump_json=lambda **_: "{}",
            tasks=[
                SimpleNamespace(
                    task_id="TASK-002",
                    context_guarded_active={
                        "status": "active" if applied else "fallback",
                        "applied": applied,
                        "active_execution_enabled": applied,
                        "production_ready": False,
                    },
                )
            ],
        )

    monkeypatch.setattr(cli, "_execute_prepared_run", execute)
    if applied:
        await cli.main()
        assert len(state.identity_calls) == 2
    else:
        with pytest.raises(SystemExit, match="was not applied"):
            await cli.main()
    assert state.arms[0]["mode"] == "active_experiment"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "inconclusive"])
async def test_no_success_exit_without_complete_ab_evidence(monkeypatch, status):
    state = setup_ab(monkeypatch, compare_status=status)
    with pytest.raises(SystemExit):
        await cli.main()
    assert state.reports[-1]["status"] == status
