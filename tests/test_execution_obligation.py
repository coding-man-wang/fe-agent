from agent_core.benchmark.execution_obligation import (
    ExecutionObligationGate,
)
from agent_core.coding.model import (
    ExecutionObligationCoverage,
    ReadOnlyCodingResult,
)
from agent_core.sdd.model import (
    ExecutionObligation,
    TaskDefinition,
)


def make_task():
    return TaskDefinition(
        task_id="TASK-004",
        task_type="code_change",
        title="compressed production change",
        objective="complete order filtering",
        source_step_order=1,
        depends_on=[],
        inputs=[],
        target_modules=[],
        expected_outputs=[],
        verification=[],
        acceptance_criteria_refs=[],
        execution_obligations=[
            ExecutionObligation(
                obligation_id="TASK-004",
                source_task_id="TASK-004",
                objective="filter state",
            ),
            ExecutionObligation(
                obligation_id="TASK-007",
                source_task_id="TASK-007",
                objective=("Remote runtime must " "send selected status"),
            ),
        ],
    )


def test_all_obligations_satisfied():

    task = make_task()

    analysis = ReadOnlyCodingResult(
        task_id="TASK-004",
        summary="ok",
        findings=[],
        proposed_changes=[],
        verification_plan=[],
        missing_information=[],
        evidence_gaps=[],
        obligation_coverage=[
            ExecutionObligationCoverage(
                obligation_id="TASK-004",
                status=("addressed_by_" "proposed_change"),
                related_paths=["src/views/OrdersView.vue"],
                reason="handled",
            ),
            ExecutionObligationCoverage(
                obligation_id="TASK-007",
                status=("addressed_by_" "proposed_change"),
                related_paths=["src/api/orders.ts"],
                reason="handled",
            ),
        ],
        execution_status="not_executed",
    )

    decision = ExecutionObligationGate().evaluate(
        task=task,
        analysis=analysis,
    )

    assert decision.allowed
    assert decision.unresolved_obligation_ids == []


def test_unresolved_obligation_blocks_execution():

    task = make_task()

    analysis = ReadOnlyCodingResult(
        task_id="TASK-004",
        summary="remote unresolved",
        findings=[],
        proposed_changes=[],
        verification_plan=[],
        missing_information=[],
        evidence_gaps=[],
        obligation_coverage=[
            ExecutionObligationCoverage(
                obligation_id="TASK-004",
                status=("satisfied_by_" "current_code"),
                related_paths=["src/views/OrdersView.vue"],
                reason="already satisfied",
            ),
            ExecutionObligationCoverage(
                obligation_id="TASK-007",
                status="unresolved",
                related_paths=["src/api/orders.ts"],
                reason=("Remote status mapping " "has not been proven"),
            ),
        ],
        execution_status="not_executed",
    )

    decision = ExecutionObligationGate().evaluate(
        task=task,
        analysis=analysis,
    )

    assert not decision.allowed

    assert decision.unresolved_obligation_ids == ["TASK-007"]
