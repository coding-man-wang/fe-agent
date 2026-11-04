from pathlib import Path

import pytest

from agent_core.benchmark.context_comparison_evaluator import (
    ContextComparisonEvaluator,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.model import CodeSymbol
from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.context.builder import ProjectContextBuilder
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)

SYMBOL_ID = "src/orders.ts::function::listOrders"


def create_ab_environment(root: Path):
    path = root / "src" / "orders.ts"
    path.parent.mkdir(parents=True)

    lines = ["const unrelated = 1;"] * 100

    lines[59:62] = [
        "export function listOrders() {",
        "  return []",
        "}",
    ]

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")

    candidate = SymbolCandidate(
        symbol_id=SYMBOL_ID,
        name="listOrders",
        kind="function",
        relative_path="src/orders.ts",
        start_line=60,
        end_line=62,
        score=1.0,
        source="evidence",
    )

    index = SymbolIndex()
    index.add_symbol(
        CodeSymbol(
            symbol_id=SYMBOL_ID,
            name="listOrders",
            kind="function",
            relative_path="src/orders.ts",
            start_line=60,
            end_line=62,
        )
    )

    relevant_file = HybridRelevantFile(
        relative_path="src/orders.ts",
        category="source",
        language="TypeScript",
        size_bytes=len(content.encode("utf-8")),
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
        context_role="reference",
        symbol_candidates=[candidate],
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=relevant_file.size_bytes,
        selected_files=[relevant_file],
    )

    common = dict(
        repository_root=root,
        requirement=RequirementSpec(
            title="订单查询",
            summary="检查订单列表函数",
        ),
        technical_spec=TechnicalSpec(
            summary="分析现有订单查询",
        ),
        task=TaskDefinition(
            task_id="TASK-001",
            task_type="code_change",
            title="检查订单查询",
            objective="分析 listOrders",
            source_step_order=1,
        ),
        repository=RepositoryProfile(
            project_name="benchmark",
            root_path=str(root),
        ),
        selection=selection,
        max_files=1,
        max_total_chars=150,
        max_chars_per_file=150,
    )

    baseline = ProjectContextBuilder(
        enable_symbol_windows=False,
    ).build(**common)

    enhanced = ProjectContextBuilder(
        enable_symbol_windows=True,
    ).build(**common)

    return baseline, enhanced, index


def test_symbol_window_ab_comparison(tmp_path):
    baseline, enhanced, index = create_ab_environment(tmp_path)

    result = ContextComparisonEvaluator().evaluate(
        case_id="AB-ORDERS-001",
        repository_root=tmp_path,
        symbol_index=index,
        required_symbol_ids=[SYMBOL_ID],
        baseline_bundle=baseline,
        enhanced_bundle=enhanced,
    )

    assert baseline.code_context[0].content_mode == "truncated"
    assert enhanced.code_context[0].content_mode == "symbol_window"

    assert result.status == "passed"
    assert result.baseline.symbol_coverage == 0.0
    assert result.enhanced.symbol_coverage == 1.0

    assert result.coverage_delta == 1.0
    assert result.no_coverage_regression is True

    assert result.enhanced.source_integrity_preserved
    assert result.enhanced.metadata_integrity_preserved
    assert result.enhanced.budget_preserved

    assert result.enhanced.context_chars <= 150
    assert result.saved_chars > 0


def test_comparison_detects_invalid_metadata(tmp_path):
    baseline, enhanced, index = create_ab_environment(tmp_path)

    # 模拟 Builder 或下游模块错误声明：
    # 实际已经包含符号，却声称没有覆盖。
    tampered = enhanced.model_copy(deep=True)
    tampered.code_context[0].covered_symbol_ids = []

    result = ContextComparisonEvaluator().evaluate(
        case_id="AB-INTEGRITY-001",
        repository_root=tmp_path,
        symbol_index=index,
        required_symbol_ids=[SYMBOL_ID],
        baseline_bundle=baseline,
        enhanced_bundle=tampered,
    )

    assert result.status == "failed"
    assert result.enhanced.metadata_integrity_preserved is False
    assert "enhanced_integrity_failed" in result.issues


def test_noise_oracle_reports_context_occupation(
    tmp_path,
):
    baseline, enhanced, index = create_ab_environment(tmp_path)

    # 这个测试故意把唯一文件标记为噪声，
    # 仅验证评估器统计公式，不代表真实订单任务。
    result = ContextComparisonEvaluator().evaluate(
        case_id="AB-NOISE-001",
        repository_root=tmp_path,
        symbol_index=index,
        required_symbol_ids=[SYMBOL_ID],
        baseline_bundle=baseline,
        enhanced_bundle=enhanced,
        irrelevant_paths={"src/orders.ts"},
    )

    assert result.baseline.noise_ratio == 1.0
    assert result.enhanced.noise_ratio == 1.0

    assert result.enhanced.noise_chars == result.enhanced.context_chars


def test_rejects_noise_oracle_outside_selection(
    tmp_path,
):
    baseline, enhanced, index = create_ab_environment(tmp_path)

    with pytest.raises(
        ValueError,
        match="Noise Oracle",
    ):
        ContextComparisonEvaluator().evaluate(
            case_id="AB-NOISE-INVALID",
            repository_root=tmp_path,
            symbol_index=index,
            required_symbol_ids=[SYMBOL_ID],
            baseline_bundle=baseline,
            enhanced_bundle=enhanced,
            irrelevant_paths={"src/missing.ts"},
        )
