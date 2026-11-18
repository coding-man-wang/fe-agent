"""D5-F: deterministic Context Optimization end-to-end tests.

Graph Filter -> Shadow Observer -> Snapshot Preview
-> Guarded Trial -> A/B Evaluation

No real LLM, no production Active, no Oracle changes.
"""

from copy import deepcopy

import pytest

from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
)
from agent_core.benchmark.task_rerank_diagnostics import (
    RerankFileScore,
    TaskRerankDiagnosticResult,
)
from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.full_execution_active_preview import (
    preview_full_execution_active_candidate,
)
from agent_core.benchmark.full_execution_guarded_trial import (
    run_guarded_active_trial,
)
from agent_core.benchmark.full_execution_ab import (
    compare_full_execution_ab,
)
from agent_core.benchmark.d5_effectiveness import (
    evaluate_d5_effectiveness,
)

# Reuse existing verified fixtures.
from test_v1_4_9_full_execution_guarded_trial import (
    trial_case,  # noqa: F401
)
from test_v1_4_9_full_execution_ab import (
    arms,  # noqa: F401
)


class FakeRerankDiagnosticRunner:
    """Replace only external rerank inference.

    Preserve the real EvidenceProtectedRerankFilter.
    """

    def __init__(self, *, complete=True):
        self.complete = complete
        self.calls = 0

    async def run(
        self,
        *,
        query,
        repository_root,
        selection,
    ):
        self.calls += 1

        if not self.complete:
            return TaskRerankDiagnosticResult(
                query=query,
                status="incomplete",
                issues=["fake_provider_incomplete"],
            )

        files = selection.selected_files

        return TaskRerankDiagnosticResult(
            query=query,
            status="complete",
            candidate_paths=[item.relative_path for item in files],
            ranked_files=[
                RerankFileScore(
                    relative_path=item.relative_path,
                    rank=index,
                    score=round(1 - index * 0.02, 4),
                    context_role=item.context_role,
                    has_symbol_evidence=bool(item.symbol_candidates),
                    source_truncated=False,
                )
                for index, item in enumerate(files, 1)
            ],
        )


async def build_shadow_and_preview(
    trial_case,
    *,
    rerank_complete=True,
    obligation_files=None,
    current_revision=None,
):
    fake_runner = FakeRerankDiagnosticRunner(complete=rerank_complete)

    runner = ContextOptimizationShadowRunner(
        rerank_runner=fake_runner,
    )

    observer = FullExecutionContextShadowObserver(
        runner=runner,
    )

    selected = trial_case["selection_result"]
    baseline = trial_case["final_baseline_bundle"]
    root = trial_case["repository_root"]
    requirement = trial_case["raw_requirement"]

    before_selection = deepcopy(selected.selection.model_dump(mode="json"))
    before_bundle = deepcopy(baseline.model_dump(mode="json"))

    shadow = await observer.observe(
        repository_root=root,
        repository_revision=selected.repository_revision,
        raw_requirement=requirement,
        selection_result=selected,
        bundle=baseline,
    )

    preview = preview_full_execution_active_candidate(
        repository_root=root,
        raw_requirement=requirement,
        selection_result=selected,
        final_bundle=baseline,
        execution_surface=trial_case["execution_surface"],
        obligation_evidence_files=(
            obligation_files
            if obligation_files is not None
            else trial_case["obligation_evidence_files"]
        ),
        shadow_report=shadow,
        get_current_revision=(
            current_revision
            if current_revision is not None
            else trial_case["get_current_revision"]
        ),
        fresh_in_process_report=True,
    )

    assert fake_runner.calls == 1

    # Shadow must never mutate the real inputs.
    assert selected.selection.model_dump(mode="json") == before_selection
    assert baseline.model_dump(mode="json") == before_bundle

    return shadow, preview


@pytest.mark.asyncio
async def test_real_shadow_to_guarded_trial_valid_candidate(
    trial_case,
):
    """Positive path: f6 is weak Graph-only evidence."""

    shadow, preview = await build_shadow_and_preview(trial_case)

    assert shadow["status"] == "complete", shadow
    assert shadow["applied"] is False

    assert shadow["graph_proposed_drops"] == ["src/f6.ts"]
    assert shadow["rerank_proposed_drops"] == []
    assert shadow["effective_proposed_drops"] == ["src/f6.ts"]

    assert preview.status == "preview_valid", preview
    assert preview.drops_in_final_context == ("src/f6.ts",)
    assert preview.applied is False

    trial = run_guarded_active_trial(
        **{
            **trial_case,
            "preview": preview,
        }
    )

    assert trial.status == "candidate_valid", trial
    assert trial.removed_files == ("src/f6.ts",)
    assert trial.candidate_chars < trial.baseline_chars

    # This is only an in-memory trial.
    assert trial.applied is False
    assert trial.production_ready is False


@pytest.mark.asyncio
async def test_obligation_protection_blocks_candidate(
    trial_case,
):
    """Required evidence must not be removed."""

    shadow, preview = await build_shadow_and_preview(
        trial_case,
        obligation_files=["src/f6.ts"],
    )

    # Shadow is not an authorization boundary.
    assert shadow["effective_proposed_drops"] == ["src/f6.ts"]

    # Final Context protection rejects the drop.
    assert preview.status == "abstained"
    assert preview.reason == "final_context_protected_drop"
    assert preview.applied is False


@pytest.mark.asyncio
async def test_incomplete_rerank_discards_graph_proposal(
    trial_case,
):
    """Incomplete rerank must not release even Graph drops."""

    shadow, preview = await build_shadow_and_preview(
        trial_case,
        rerank_complete=False,
    )

    assert shadow["status"] == "abstained"
    assert shadow["effective_proposed_drops"] == []
    assert preview.status == "abstained"
    assert preview.applied is False


@pytest.mark.asyncio
async def test_stale_revision_blocks_preview(
    trial_case,
):
    shadow, preview = await build_shadow_and_preview(
        trial_case,
        current_revision=lambda: "f" * 40,
    )

    assert shadow["status"] == "complete"
    assert preview.status == "abstained"
    assert preview.applied is False


def test_synthetic_ab_passes_but_never_enables_production(
    arms,
):
    """Existing frozen synthetic A/B records, no execution."""

    baseline, active = deepcopy(arms)

    ab = compare_full_execution_ab(
        baseline=baseline,
        active=active,
    )

    effectiveness = evaluate_d5_effectiveness(ab)

    assert ab["status"] == "passed"
    assert effectiveness["status"] == "saving_confirmed"
    assert effectiveness["llm_tokens_saved"] == 50
    assert effectiveness["token_savings_confirmed"] is True

    assert effectiveness["production_ready"] is False
    assert ab["production_active_enabled"] is False


def test_incomplete_usage_cannot_claim_token_savings(
    arms,
):
    baseline, active = deepcopy(arms)

    active["model_usage"].update(
        usage_complete=False,
        failed_calls=1,
        usage_missing_calls=1,
        total_tokens=None,
    )

    ab = compare_full_execution_ab(
        baseline=baseline,
        active=active,
    )

    effectiveness = evaluate_d5_effectiveness(ab)

    assert effectiveness["token_savings_confirmed"] is False
    assert effectiveness["llm_tokens_saved"] is None
    assert effectiveness["model_usage_comparable"] is False


def test_active_fallback_is_not_optimization_success(
    arms,
):
    baseline, active = deepcopy(arms)

    active["tasks"][0]["context_guarded_active"]["applied"] = False

    ab = compare_full_execution_ab(
        baseline=baseline,
        active=active,
    )

    effectiveness = evaluate_d5_effectiveness(ab)

    assert ab["status"] == "failed"
    assert ab["gates"]["actual_active_consumed"] is False
    assert effectiveness["token_savings_confirmed"] is False
    assert effectiveness["production_ready"] is False
