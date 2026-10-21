from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.execution_evaluator import FullExecutionEvaluator
from agent_core.benchmark.execution_model import FullExecutionTaskRecord
from agent_core.coding.verification_model import PatchVerificationResult


def make_case() -> BenchmarkCase:
    return BenchmarkCase.model_validate(
        {
            "schema_version": 2,
            "case_id": "CASE-006",
            "title": "x",
            "category": "cross_file_change",
            "difficulty": "medium",
            "repository_name": "repo",
            "base_commit_sha": "a" * 40,
            "requirement": "x",
            "expectation": {
                "required_context_files": ["src/stores/users.ts"],
                "useful_context_files": [],
                "forbidden_context_files": [],
                "expected_change_files": [
                    "src/stores/users.ts",
                    "src/views/UsersView.vue",
                ],
                "allowed_change_files": ["tests/unit/usersStore.test.ts"],
                "forbidden_change_files": ["src/api", "src/mocks"],
                "expected_behaviors": ["x"],
                "verification_command_ids": ["frontend-unit-test"],
            },
        }
    )


def hidden(status: str = "passed") -> PatchVerificationResult:
    return PatchVerificationResult(
        proposal_id="b" * 64,
        task_id="CASE-006",
        status=status,
    )


def test_full_execution_evaluator_passes_exact_change_set():
    evaluation = FullExecutionEvaluator().evaluate(
        case=make_case(),
        actual_change_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        task_records=[
            FullExecutionTaskRecord(
                task_id="TASK-002",
                task_type="code_change",
                status="completed",
                verification_status="succeeded",
            )
        ],
        hidden_acceptance=hidden(),
    )
    assert evaluation.status == "passed"
    assert evaluation.missed_change_files == []
    assert evaluation.unexpected_change_files == []
    assert evaluation.forbidden_change_hits == []


def test_full_execution_evaluator_detects_directory_forbidden_hit():
    evaluation = FullExecutionEvaluator().evaluate(
        case=make_case(),
        actual_change_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
            "src/api/users.ts",
        ],
        task_records=[
            FullExecutionTaskRecord(
                task_id="TASK-002",
                task_type="code_change",
                status="completed",
                verification_status="succeeded",
            )
        ],
        hidden_acceptance=hidden(),
    )
    assert evaluation.status == "failed"
    assert evaluation.forbidden_change_hits == ["src/api/users.ts"]
    assert "forbidden_change" in evaluation.failure_reasons
