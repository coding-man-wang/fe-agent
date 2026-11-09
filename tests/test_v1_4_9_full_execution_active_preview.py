from dataclasses import replace

from agent_core.benchmark.full_execution_active_preview import (
    ActiveCandidatePreview,
)


def test_preview_result_never_enables_active():
    preview = ActiveCandidatePreview(
        status="preview_valid",
        reason="candidate_snapshot_and_protection_valid",
        proposed_selection_drops=("src/unused.ts",),
        drops_in_final_context=("src/unused.ts",),
    )

    assert preview.applied is False
    assert preview.production_ready is False


def test_preview_defaults_to_no_policy_report():
    preview = ActiveCandidatePreview(
        status="abstained",
        reason="no_change",
    )

    assert preview.p1_report is None
    assert preview.proposed_selection_drops == ()
    assert preview.drops_in_final_context == ()
    assert preview.applied is False
