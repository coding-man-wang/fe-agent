from pathlib import Path

import pytest

from agent_core.benchmark.prepared_execution import (
    PreparedExecutionInput,
    PreparedExecutionInputError,
    PreparedExecutionInputStore,
)
from agent_core.benchmark.change_surface_model import ChangeSurfaceEvaluation
from agent_core.benchmark.case_model import BenchmarkCase


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
                "required_context_files": [
                    "src/stores/users.ts",
                ],
                "useful_context_files": [],
                "forbidden_context_files": [],
                "expected_change_files": [
                    "src/stores/users.ts",
                ],
                "allowed_change_files": [],
                "forbidden_change_files": [],
                "expected_behaviors": [
                    "should satisfy benchmark behavior",
                ],
                "verification_command_ids": [
                    "frontend-lint",
                ],
            },
        }
    )


def make_prepared() -> PreparedExecutionInput:
    evaluation = ChangeSurfaceEvaluation(
        status="passed",
        expected_change_files=[],
        predicted_change_files=[],
        matched_change_files=[],
        missed_change_files=[],
        unexpected_change_files=[],
        forbidden_change_hits=[],
        target_recall=1.0,
        target_precision=1.0,
        dependency_expected_files=[],
        dependency_coverage=1.0,
        failure_reasons=[],
    )
    return PreparedExecutionInput(
        case_id="CASE-006",
        base_commit_sha="a" * 40,
        baseline_run_id="b" * 32,
        source="historical_passed_surface",
        surface_evaluation=evaluation,
    )


def test_prepared_store_roundtrip(tmp_path: Path):
    (tmp_path / ".fe-agent").mkdir()
    case = make_case()
    prepared = make_prepared()
    store = PreparedExecutionInputStore(fe_agent_root=tmp_path)
    store.save(prepared)
    loaded = store.load_optional(case=case)
    assert loaded is not None
    assert loaded.baseline_run_id == "b" * 32
    assert loaded.surface_evaluation.status == "passed"


def test_prepared_rejects_different_base_commit(tmp_path: Path):
    (tmp_path / ".fe-agent").mkdir()
    case = make_case().model_copy(update={"base_commit_sha": "c" * 40})
    store = PreparedExecutionInputStore(fe_agent_root=tmp_path)
    store.save(make_prepared())
    with pytest.raises(PreparedExecutionInputError):
        store.load_optional(case=case)
