from apps.agent_cli.benchmark_policy_canary_v1_4_9 import (
    REQUIRED_GATES,
    case_gates_passed,
)


def test_empty_completion_gates_cannot_pass():
    assert not case_gates_passed(
        {
            "case_id": "ORDER-001",
            "status": "passed",
            "completion_gates": {},
        }
    )


def test_missing_gate_cannot_pass():
    gates = {name: True for name in REQUIRED_GATES}

    gates.pop("required_symbol_coverage")

    assert not case_gates_passed(
        {
            "case_id": "ORDER-001",
            "status": "passed",
            "completion_gates": gates,
        }
    )


def test_all_required_gates_can_pass():
    gates = {name: True for name in REQUIRED_GATES}

    assert case_gates_passed(
        {
            "case_id": "ORDER-001",
            "status": "passed",
            "completion_gates": gates,
        }
    )


def test_one_failed_gate_cannot_pass():
    gates = {name: True for name in REQUIRED_GATES}

    gates["required_file_recall"] = False

    assert not case_gates_passed(
        {
            "case_id": "ORDER-001",
            "status": "passed",
            "completion_gates": gates,
        }
    )
