import pytest

from agent_core.benchmark.symbol_retrieval_evaluator import (
    SymbolRetrievalBenchmarkCase,
    SymbolRetrievalBenchmarkEvaluator,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)


def candidate(
    symbol_id: str,
    *,
    source: str = "name",
) -> SymbolCandidate:

    return SymbolCandidate(
        symbol_id=symbol_id,
        name=symbol_id,
        kind="function",
        relative_path="src/test.ts",
        start_line=1,
        end_line=1,
        score=1.0,
        source=source,
    )


def test_legacy_expected_symbols_become_required():

    case = SymbolRetrievalBenchmarkCase(
        case_id="LEGACY",
        queries=["orders"],
        expected_symbol_ids=[
            "A",
            "B",
        ],
    )

    result = SymbolRetrievalBenchmarkEvaluator().evaluate(
        case=case,
        direct_candidates=[
            candidate("A"),
        ],
        expanded_candidates=[
            candidate("A"),
            candidate(
                "B",
                source="graph",
            ),
        ],
    )

    assert result.required_symbol_ids == ["A", "B"]

    assert result.symbol_recall == 1.0

    assert result.status == "passed"


def test_strict_and_useful_precision_are_different():

    case = SymbolRetrievalBenchmarkCase(
        case_id="TIERED",
        queries=["orders"],
        required_symbol_ids=[
            "A",
            "B",
        ],
        supporting_symbol_ids=[
            "C",
        ],
    )

    result = SymbolRetrievalBenchmarkEvaluator().evaluate(
        case=case,
        direct_candidates=[
            candidate("A"),
        ],
        expanded_candidates=[
            candidate("A"),
            candidate(
                "B",
                source="graph",
            ),
            candidate(
                "C",
                source="graph",
            ),
            candidate(
                "D",
                source="graph",
            ),
        ],
    )

    assert result.symbol_recall == 1.0

    assert result.strict_precision == 0.5

    assert result.useful_precision == 0.75

    # Graph 新增：
    # B required
    # C supporting
    # D harmful
    #
    # Strict Noise：
    # C + D = 2/3
    #
    # Harmful Noise：
    # D = 1/3

    assert result.strict_noise_expansion_rate == pytest.approx(2 / 3)

    assert result.harmful_noise_expansion_rate == pytest.approx(1 / 3)

    assert result.harmful_graph_noise_symbol_ids == ["D"]


def test_forbidden_symbol_still_fails_case():

    case = SymbolRetrievalBenchmarkCase(
        case_id="FORBIDDEN",
        queries=["orders"],
        required_symbol_ids=[
            "A",
        ],
        forbidden_symbol_ids=[
            "USER",
        ],
    )

    result = SymbolRetrievalBenchmarkEvaluator().evaluate(
        case=case,
        direct_candidates=[
            candidate("A"),
        ],
        expanded_candidates=[
            candidate("A"),
            candidate("USER"),
        ],
    )

    assert result.status == "failed"

    assert result.forbidden_hits == ["USER"]


def test_ground_truth_tiers_cannot_overlap():

    with pytest.raises(ValueError):
        SymbolRetrievalBenchmarkCase(
            case_id="INVALID",
            queries=["orders"],
            required_symbol_ids=[
                "A",
            ],
            supporting_symbol_ids=[
                "A",
            ],
        )
