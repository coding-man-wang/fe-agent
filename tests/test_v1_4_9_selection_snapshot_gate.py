import hashlib

import pytest

from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.benchmark.selection_snapshot_gate import (
    ShadowSnapshotGate,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)

REVISION = "test-revision-001"
REQUIREMENT = "修改 demo 功能"


@pytest.fixture
def snapshot_case(tmp_path):
    source = tmp_path / "src" / "demo.ts"
    source.parent.mkdir(parents=True)

    source.write_text(
        "export const value = 1\n",
        encoding="utf-8",
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=23,
        selected_files=[
            HybridRelevantFile(
                relative_path="src/demo.ts",
                category="source",
                language="TypeScript",
                size_bytes=23,
                retrieval_score=1.0,
                retrieval_rank=1,
                policy_score=0,
                selection_score=1.0,
            )
        ],
    )

    evidence = capture_selection_evidence(
        repository_root=tmp_path,
        selection=selection,
    )

    report = {
        "policy_version": "v1.4.9-P1",
        "mode": "shadow",
        "task_id": "TEST-001",
        "repository_revision": REVISION,
        "requirement_sha256": hashlib.sha256(REQUIREMENT.encode("utf-8")).hexdigest(),
        "status": "complete",
        "applied": False,
        "selection_unchanged": True,
        "baseline_file_count": 1,
        "proposed_file_count": 1,
        "effective_proposed_drops": [],
        "evidence_binding": {
            "selection_sha256": (evidence.selection_sha256),
            "source_sha256_before": (evidence.source_sha256),
            "source_sha256_after": (evidence.source_sha256),
            "source_matched": True,
            "selection_matched": True,
            "changed_paths": [],
        },
    }

    return tmp_path, source, selection, report


def check(snapshot_case, **overrides):
    root, _, selection, report = snapshot_case

    arguments = {
        "report": report,
        "repository_root": root,
        "selection": selection,
        "task_id": "TEST-001",
        "raw_requirement": REQUIREMENT,
        "expected_revision": REVISION,
        "current_revision": REVISION,
    }

    arguments.update(overrides)

    return ShadowSnapshotGate().validate(**arguments)


def test_valid_snapshot_is_accepted(snapshot_case):
    decision = check(snapshot_case)

    assert decision.allowed is True
    assert decision.reason == "snapshot_valid"


def test_modified_source_is_rejected(snapshot_case):
    _, source, _, _ = snapshot_case

    source.write_text(
        "export const value = 2\n",
        encoding="utf-8",
    )

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == ("source_snapshot_changed")


def test_selection_change_is_rejected(snapshot_case):
    _, _, selection, _ = snapshot_case

    selection.selected_files[0].retrieval_score = 0.5

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == ("selection_snapshot_changed")


def test_requirement_change_is_rejected(snapshot_case):
    decision = check(
        snapshot_case,
        raw_requirement="另一个任务",
    )

    assert decision.allowed is False
    assert decision.reason == "requirement_changed"


def test_repository_revision_change_is_rejected(
    snapshot_case,
):
    decision = check(
        snapshot_case,
        current_revision="new-revision",
    )

    assert decision.allowed is False
    assert decision.reason == ("repository_revision_changed")


def test_incomplete_shadow_is_rejected(snapshot_case):
    _, _, _, report = snapshot_case

    report["status"] = "abstained"

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == ("shadow_not_complete")


def test_unknown_drop_path_is_rejected(snapshot_case):
    _, _, _, report = snapshot_case

    report["effective_proposed_drops"] = ["src/not-selected.ts"]
    report["proposed_file_count"] = 0

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == ("proposal_path_mismatch")


def test_proposal_count_mismatch_is_rejected(
    snapshot_case,
):
    _, _, _, report = snapshot_case

    report["proposed_file_count"] = 99

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == "proposal_count_mismatch"


def test_malformed_drop_is_rejected_safely(
    snapshot_case,
):
    _, _, _, report = snapshot_case

    report["effective_proposed_drops"] = [{"unexpected": "object"}]

    decision = check(snapshot_case)

    assert decision.allowed is False
    assert decision.reason == "invalid_proposed_drops"
