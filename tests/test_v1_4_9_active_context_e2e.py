from copy import deepcopy

from apps.agent_cli.benchmark_active_context_v1_4_9 import (
    REQUIRED_CANARY_GATES,
    quality_gate,
    validate_canary_report,
)

REVISION = "revision-001"
EXPECTED_IDS = {
    "ORDER-001",
    "USER-001",
    "API-001",
}


def make_canary():
    return {
        "status": "passed",
        "repository_revision": REVISION,
        "safety_tests_passed": True,
        "shadow_consumer_passed": True,
        "active_experiment_eligible": True,
        "production_ready": False,
        "total_cases": 3,
        "passed_cases": 3,
        "cases": [
            {
                "case_id": case_id,
                "status": "passed",
                "completion_gates": {key: True for key in REQUIRED_CANARY_GATES},
            }
            for case_id in sorted(EXPECTED_IDS)
        ],
    }


def valid(report):
    return validate_canary_report(
        report,
        revision=REVISION,
        expected_ids=EXPECTED_IDS,
    )


def test_valid_canary_is_accepted():
    assert valid(make_canary()) is True


def test_wrong_revision_is_rejected():
    report = make_canary()
    report["repository_revision"] = "another-revision"

    assert valid(report) is False


def test_missing_canary_gate_is_rejected():
    report = make_canary()
    report["cases"][0]["completion_gates"].pop("required_symbol_coverage")

    assert valid(report) is False


def test_shadow_consumer_failure_is_rejected():
    report = make_canary()
    report["shadow_consumer_passed"] = False

    assert valid(report) is False


def test_duplicate_case_does_not_replace_missing_case():
    report = make_canary()
    report["cases"][2]["case_id"] = "ORDER-001"

    assert valid(report) is False


def make_quality():
    return {
        "required_file_recall": 1.0,
        "required_symbol_coverage": 1.0,
        "symbol_status": "passed",
        "source_integrity": True,
        "metadata_integrity": True,
        "budget_integrity": True,
        "irrelevant_chars": 500,
        "context_chars": 4000,
    }


def test_optimized_context_can_reduce_chars():
    baseline = make_quality()
    actual = deepcopy(baseline)

    actual["context_chars"] = 2900
    actual["irrelevant_chars"] = 0

    assert quality_gate(baseline, actual) is True


def test_missing_symbol_blocks_active():
    baseline = make_quality()
    actual = deepcopy(baseline)

    actual["context_chars"] = 2000
    actual["required_symbol_coverage"] = 0.8

    assert quality_gate(baseline, actual) is False


def test_source_integrity_failure_blocks_active():
    baseline = make_quality()
    actual = deepcopy(baseline)

    actual["context_chars"] = 2000
    actual["source_integrity"] = False

    assert quality_gate(baseline, actual) is False


def test_context_growth_blocks_active():
    baseline = make_quality()
    actual = deepcopy(baseline)

    actual["context_chars"] = 4500

    assert quality_gate(baseline, actual) is False
