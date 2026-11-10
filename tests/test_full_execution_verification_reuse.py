"""Run the actual service verification loop without external commands/models."""

from types import SimpleNamespace

import pytest

import agent_core.benchmark.full_execution as execution
from agent_core.benchmark.case_model import BenchmarkCase, BenchmarkExpectation
from agent_core.codebase.model import RepositoryProfile
from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TaskPlan,
    TechnicalSpec,
)


@pytest.fixture
def run_verifications(tmp_path, monkeypatch):
    agent_root = tmp_path / "agent"
    repository_root = tmp_path / "repository"
    agent_root.mkdir()
    repository_root.mkdir()
    calls = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        async def run(self, *, plan):
            calls.append(plan)
            return PatchVerificationResult(
                proposal_id=plan.proposal_id,
                task_id=plan.task_id,
                status="passed",
                checks=[
                    VerificationCheck(name=command, command_id=command, status="passed")
                    for command in plan.command_ids
                ],
            )

    async def hidden(**kwargs):
        return PatchVerificationResult(
            proposal_id="f" * 64,
            task_id="hidden",
            status="passed",
            checks=[VerificationCheck(name="hidden-acceptance", status="passed")],
        )

    async def changed_files(**kwargs):
        return []

    monkeypatch.setattr(execution, "VerificationExecutor", Executor)
    monkeypatch.setattr(
        execution, "HiddenAcceptanceRunner", lambda **_: SimpleNamespace(run=hidden)
    )
    monkeypatch.setattr(
        execution,
        "GitDiffCollector",
        lambda: SimpleNamespace(changed_files=changed_files),
    )
    monkeypatch.setattr(
        execution.NodeModulesWorkspaceProvisioner, "provision", lambda *_: None
    )
    real_build = execution.FinalVerificationPlanner.build

    async def run(revision="a" * 40, *, change_commands=False):
        planned_count = 0

        def build(planner, **kwargs):
            nonlocal planned_count
            planned_count += 1
            plan = real_build(planner, **kwargs)
            if change_commands and planned_count == 2:
                plan = plan.model_copy(
                    update={
                        "added_command_ids": ["frontend-typecheck"],
                        "final_command_ids": [
                            *plan.final_command_ids,
                            "frontend-typecheck",
                        ],
                    }
                )
            return plan

        monkeypatch.setattr(execution.FinalVerificationPlanner, "build", build)
        service = execution.FullExecutionBenchmarkService(
            fe_agent_root=agent_root,
            benchmark_repository=repository_root,
            baseline_run_id="prepared",
            llm=object(),
            coordinator=object(),
            context_selection_service=object(),
        )
        monkeypatch.setattr(service, "_create_run_scoped_coordinator", lambda: object())
        tasks = [
            TaskDefinition(
                task_id=f"VERIFY-{number}",
                task_type="verification",
                title="Verify unchanged revision",
                objective="Run deterministic checks",
                source_step_order=number,
                depends_on=["VERIFY-1"] if number == 2 else [],
            )
            for number in (1, 2)
        ]
        return await service.run(
            case=BenchmarkCase(
                case_id="CASE-006",
                title="Verification reuse",
                category="single_file_change",
                difficulty="easy",
                repository_name="fixture",
                base_commit_sha=revision,
                requirement="Verify the existing code",
                expectation=BenchmarkExpectation(
                    expected_behaviors=["Checks pass"],
                    verification_command_ids=["frontend-lint"],
                ),
            ),
            task_plan=TaskPlan(summary="Two verification tasks", tasks=tasks),
            requirement=RequirementSpec(
                title="Verify", summary="Check current revision"
            ),
            technical_spec=TechnicalSpec(summary="Deterministic verification"),
            repository=RepositoryProfile(
                project_name="fixture", root_path=str(repository_root)
            ),
            engineering_rules=[],
            task_surfaces={},
        )

    return run, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("change_commands", [False, True])
async def test_same_revision_reuses_only_identical_successful_verification(
    run_verifications, change_commands
):
    run, calls = run_verifications
    record = await run(change_commands=change_commands)
    assert record.status == "completed", record.error
    assert record.evaluation.status == "passed"
    assert record.evaluation.failed_task_ids == []
    assert len(calls) == (2 if change_commands else 1)
    first, second = record.tasks
    assert first.status == "completed"
    assert second.context_revision == first.context_revision == "a" * 40
    if change_commands:
        assert second.status == "completed"
        assert calls[1].command_ids == ["frontend-lint", "frontend-typecheck"]
    else:
        assert second.status == "skipped_redundant_verification"
        assert second.planned_verification.final_command_ids == ["frontend-lint"]
        assert second.executed_verification_command_ids == []
        assert second.verification_checks == []
        assert second.verification_status is None  # Does not invent another execution.


@pytest.mark.asyncio
async def test_new_revision_run_cannot_reuse_previous_verification_evidence(
    run_verifications,
):
    run, calls = run_verifications
    first = await run("a" * 40)
    second = await run("b" * 40)
    assert first.evaluation.status == second.evaluation.status == "passed"
    assert len(calls) == 2
    assert calls[0].proposal_id != calls[1].proposal_id
    assert first.tasks[0].context_revision != second.tasks[0].context_revision
