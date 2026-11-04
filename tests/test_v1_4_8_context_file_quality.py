import pytest

from agent_core.benchmark.context_file_quality import (
    ContextFileOracle,
    ContextFileQualityEvaluator,
)

from tests.test_v1_4_7_context_comparison import (
    create_ab_environment,
)


def test_file_quality_all_required_found(tmp_path):
    _, enhanced, _ = create_ab_environment(tmp_path)

    result = ContextFileQualityEvaluator().evaluate(
        bundle=enhanced,
        oracle=ContextFileOracle(
            required_files=["src/orders.ts"],
        ),
    )

    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.missing_required == []
    assert result.irrelevant_chars == 0


def test_file_quality_detects_missing_required(tmp_path):
    _, enhanced, _ = create_ab_environment(tmp_path)

    result = ContextFileQualityEvaluator().evaluate(
        bundle=enhanced,
        oracle=ContextFileOracle(
            required_files=[
                "src/orders.ts",
                "src/missing.ts",
            ],
        ),
    )

    assert result.recall == pytest.approx(0.5)
    assert result.missing_required == ["src/missing.ts"]


def test_file_oracle_rejects_overlapping_roles(tmp_path):
    _, enhanced, _ = create_ab_environment(tmp_path)

    with pytest.raises(
        ValueError,
        match="角色存在交集",
    ):
        ContextFileQualityEvaluator().evaluate(
            bundle=enhanced,
            oracle=ContextFileOracle(
                required_files=["src/orders.ts"],
                irrelevant_files=["src/orders.ts"],
            ),
        )
