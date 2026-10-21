import pytest

from agent_core.benchmark.acceptance import (
    HiddenAcceptanceError,
    HiddenAcceptanceRegistry,
)


def test_case_006_hidden_acceptance_is_registered():
    spec = HiddenAcceptanceRegistry().get("CASE-006")
    assert spec.fixture_name == "CASE-006.test.ts"
    assert spec.relative_path.endswith("CASE-006.acceptance.test.ts")


def test_unknown_case_requires_explicit_hidden_acceptance():
    with pytest.raises(HiddenAcceptanceError):
        HiddenAcceptanceRegistry().get("CASE-999")
