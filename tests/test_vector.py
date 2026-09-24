import pytest

from agent_core.codebase.vector import (
    cosine_similarity,
)


def test_same_direction():
    score = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors():
    score = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert score == pytest.approx(0.0)


def test_vector_dimension_mismatch():
    with pytest.raises(
            ValueError,
            match="向量维度必须一致",
    ):
        cosine_similarity(
            [1.0],
            [1.0, 2.0],
        )
