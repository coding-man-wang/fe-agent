from agent_core.codebase.reranker import (
    RerankHit,
)

from agent_core.codebase.rerank_selection import (
    AdaptiveRerankSelectionPolicy,
)


def make_hit(
    rank: int,
    score: float,
) -> RerankHit:

    return RerankHit(
        document_id=(f"file-{rank}"),
        rank=rank,
        score=score,
    )


def test_large_score_gap_cuts_noise():

    policy = AdaptiveRerankSelectionPolicy(
        min_files=2,
        max_files=8,
        min_score_gap=0.20,
    )

    hits = [
        make_hit(1, 0.93),
        make_hit(2, 0.92),
        make_hit(3, 0.44),
        make_hit(4, 0.39),
        make_hit(5, 0.35),
    ]

    selected = policy.select(hits)

    assert [item.rank for item in selected] == [1, 2]


def test_no_large_gap_keeps_candidates():

    policy = AdaptiveRerankSelectionPolicy(
        min_files=2,
        max_files=5,
        min_score_gap=0.20,
    )

    hits = [
        make_hit(1, 0.80),
        make_hit(2, 0.72),
        make_hit(3, 0.65),
        make_hit(4, 0.60),
    ]

    selected = policy.select(hits)

    assert len(selected) == 4


def test_never_cuts_before_min_files():

    policy = AdaptiveRerankSelectionPolicy(
        min_files=2,
        max_files=8,
        min_score_gap=0.20,
    )

    hits = [
        make_hit(1, 0.95),
        make_hit(2, 0.50),
        make_hit(3, 0.20),
    ]

    selected = policy.select(hits)

    assert len(selected) == 2
