import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_core.benchmark.case_loader import BenchmarkCaseLoader
from agent_core.benchmark.full_execution_active_preflight import ActivePreflightDecision
from apps.agent_cli import benchmark_v1_1_run as cli


def args(**changes):
    values = dict(
        case_id="CASE-006",
        context_active_preflight=False,
        context_guarded_preview=True,
        context_shadow=False,
        refresh_upstream=False,
        active_task_id="TASK-002",
        approved_revision="a" * 40,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def decision():
    return ActivePreflightDecision(
        version="v1.4.9-P4-B-D2-A",
        status="passed",
        reason="active_preflight_valid",
        case_id="CASE-006",
        task_id="TASK-002",
        approved_revision="a" * 40,
        current_revision="a" * 40,
        canary_sha256="b" * 64,
    )


def forbid_initialization(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Guarded Preview gate must run before runtime initialization")

    for name in (
        "create_benchmark_runner",
        "FullExecutionBenchmarkService",
        "NodeModulesWorkspaceProvisioner",
        "create_full_execution_shadow_observer",
    ):
        monkeypatch.setattr(cli, name, forbidden)


def test_flag_is_opt_in_and_explains_baseline_consumption(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    assert cli._parse_args().context_guarded_preview is False
    monkeypatch.setattr(sys, "argv", ["benchmark", "--context-guarded-preview"])
    assert cli._parse_args().context_guarded_preview is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_flag", ["context_active_preflight", "context_shadow", "refresh_upstream"]
)
async def test_conflicting_modes_stop_before_runtime(monkeypatch, other_flag):
    monkeypatch.setattr(cli, "_parse_args", lambda: args(**{other_flag: True}))
    forbid_initialization(monkeypatch)
    with pytest.raises(SystemExit, match="不能与"):
        await cli.main()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["active_task_id", "approved_revision"])
async def test_preview_requires_explicit_identity_before_runtime(monkeypatch, missing):
    monkeypatch.setattr(cli, "_parse_args", lambda: args(**{missing: None}))
    forbid_initialization(monkeypatch)
    with pytest.raises(SystemExit, match="必须显式提供"):
        await cli.main()


@pytest.mark.asyncio
async def test_ordinary_execution_rejects_stray_active_arguments(monkeypatch):
    monkeypatch.setattr(cli, "_parse_args", lambda: args(context_guarded_preview=False))
    forbid_initialization(monkeypatch)
    with pytest.raises(SystemExit, match="Active 参数只能与"):
        await cli.main()


@pytest.mark.asyncio
async def test_missing_canary_denies_before_runtime(monkeypatch, tmp_path, capsys):
    case = BenchmarkCaseLoader(Path(__file__).resolve().parents[1] / "benchmark").load(
        "CASE-006"
    )
    monkeypatch.setattr(
        cli, "_parse_args", lambda: args(approved_revision=case.base_commit_sha)
    )
    monkeypatch.setattr(
        cli, "__file__", str(tmp_path / "apps" / "agent_cli" / "run.py")
    )
    monkeypatch.setattr(
        cli, "BenchmarkCaseLoader", lambda *_: SimpleNamespace(load=lambda _: case)
    )
    monkeypatch.setattr(
        cli,
        "git_command",
        lambda _root, *command: (
            case.base_commit_sha if command[0] == "rev-parse" else ""
        ),
    )
    forbid_initialization(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        await cli.main()
    assert exc.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "canary_unavailable"
    assert report["active_execution_enabled"] is False


def test_shared_preflight_binds_exact_canary_bytes(monkeypatch, tmp_path, capsys):
    canary = tmp_path / "var" / "canary" / "context_policy_v1_4_9.json"
    canary.parent.mkdir(parents=True)
    raw = b'{"example": "canary"}\n'
    canary.write_bytes(raw)
    case = object()
    monkeypatch.setattr(
        cli, "BenchmarkCaseLoader", lambda *_: SimpleNamespace(load=lambda _: case)
    )
    monkeypatch.setattr(
        cli,
        "git_command",
        lambda _root, *command: "a" * 40 if command[0] == "rev-parse" else "",
    )
    seen = {}
    passed = decision()

    def evaluate(**kwargs):
        seen.update(kwargs)
        return passed

    monkeypatch.setattr(cli, "evaluate_active_preflight", evaluate)
    actual = cli._checked_active_preflight(
        args=args(),
        fe_agent_root=tmp_path,
        benchmark_repository=tmp_path / "repository",
    )
    assert actual is passed
    assert seen["canary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert seen["canary_report"] == {"example": "canary"}
    assert seen["repository_dirty"] is False
    assert json.loads(capsys.readouterr().out)["production_ready"] is False


def fake_prepared_runtime(monkeypatch, *, prepared_available=True):
    calls = []
    passed = decision()

    def preflight(**_kwargs):
        calls.append("preflight")
        return passed

    monkeypatch.setattr(cli, "_parse_args", args)
    monkeypatch.setattr(cli, "_checked_active_preflight", preflight)
    monkeypatch.setattr(
        cli,
        "NodeModulesWorkspaceProvisioner",
        lambda **_: SimpleNamespace(
            require_source_ready=lambda: calls.append("dependencies")
        ),
    )
    baseline = SimpleNamespace(
        task_plan=object(),
        requirement_spec=object(),
        technical_spec=object(),
        project_context=SimpleNamespace(engineering_rules=[]),
    )
    case = SimpleNamespace(case_id="CASE-006")
    runner = SimpleNamespace(
        case_loader=SimpleNamespace(load=lambda _: case),
        run_store=SimpleNamespace(load=lambda _: baseline),
        requirement_analyzer=SimpleNamespace(llm=object()),
    )

    def create_runner(**_kwargs):
        calls.append("runner")
        return runner

    monkeypatch.setattr(cli, "create_benchmark_runner", create_runner)
    prepared = SimpleNamespace(
        validate_for_case=lambda _: calls.append("validate_prepared"),
        source="reused",
        baseline_run_id="baseline-1",
        base_commit_sha="a" * 40,
        surface_evaluation=SimpleNamespace(
            status="passed", target_recall=1, target_precision=1
        ),
        task_surfaces={},
    )
    monkeypatch.setattr(
        cli,
        "PreparedExecutionInputStore",
        lambda **_: SimpleNamespace(
            load_optional=lambda **_: prepared if prepared_available else None
        ),
    )
    monkeypatch.setattr(
        cli, "RepositoryFileIndexer", lambda: SimpleNamespace(build=lambda _: object())
    )
    monkeypatch.setattr(
        cli,
        "HistoricalPreparedExecutionFinder",
        lambda **_: SimpleNamespace(find_latest_passed=lambda **_: None),
    )

    async def forbidden_prepare(**_kwargs):
        pytest.fail("Guarded Preview must not silently refresh upstream")

    monkeypatch.setattr(cli, "_prepare_fresh", forbidden_prepare)
    monkeypatch.setattr(cli, "_repository_profile", lambda **_: object())
    monkeypatch.setattr(cli, "_coordinator_for_baseline", lambda **_: object())
    monkeypatch.setattr(cli, "_build_context_selection_service", lambda **_: object())
    observer = object()
    monkeypatch.setattr(cli, "create_full_execution_shadow_observer", lambda: observer)
    captured = {}

    class Execution:
        def __init__(self, **kwargs):
            calls.append("execution")
            captured.update(kwargs)

        async def run(self, **kwargs):
            calls.append("run")
            return SimpleNamespace(model_dump_json=lambda **_: "{}", run_id="preview-1")

    monkeypatch.setattr(cli, "FullExecutionBenchmarkService", Execution)
    return calls, captured, passed, observer


@pytest.mark.asyncio
async def test_passed_preview_wires_observer_and_admission_into_full_execution(
    monkeypatch,
):
    calls, captured, passed, observer = fake_prepared_runtime(monkeypatch)
    await cli.main()
    assert calls == [
        "preflight",
        "dependencies",
        "runner",
        "validate_prepared",
        "execution",
        "run",
    ]
    assert captured["context_guarded_preflight"] is passed
    assert captured["context_shadow_observer"] is observer
    assert passed.active_execution_enabled is False
    assert passed.production_ready is False


@pytest.mark.asyncio
async def test_preview_requires_existing_prepared_input(monkeypatch):
    calls, captured, *_ = fake_prepared_runtime(monkeypatch, prepared_available=False)
    with pytest.raises(SystemExit, match="需要既有 passed"):
        await cli.main()
    assert calls == ["preflight", "dependencies", "runner"]
    assert not captured
