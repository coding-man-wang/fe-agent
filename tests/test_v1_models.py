import pytest
from pydantic import ValidationError

from agent_core.change_surface.model import (
    InvestigationArtifact,
    ObservedRepositoryFile,
    ResolvedChangeSurface,
)


def test_observed_role_requires_evidence():
    with pytest.raises(ValidationError):
        ObservedRepositoryFile(
            relative_path="src/stores/users.ts",
            observed_role="candidate_target",
            evidence=[],
            confidence=0.9,
        )


def test_surface_rejects_same_file_in_two_roles():
    with pytest.raises(ValidationError):
        ResolvedChangeSurface(
            task_id="TASK-002",
            implementation_targets=["src/stores/users.ts"],
            dependencies=["src/stores/users.ts"],
        )


def test_investigation_file_unique():
    with pytest.raises(ValidationError):
        InvestigationArtifact(
            task_id="TASK-001",
            summary="facts",
            observed_files=[
                ObservedRepositoryFile(
                    relative_path="src/stores/users.ts",
                    observed_role="dependency",
                    evidence=["fact"],
                    confidence=0.8,
                ),
                ObservedRepositoryFile(
                    relative_path="src/stores/users.ts",
                    observed_role="reference",
                    evidence=["fact2"],
                    confidence=0.7,
                ),
            ],
        )
