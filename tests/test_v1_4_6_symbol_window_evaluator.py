from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.benchmark.symbol_window_evaluator import (
    SymbolWindowBenchmarkCase,
    SymbolWindowBenchmarkEvaluator,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.codebase.symbol_retrieval_model import SymbolCandidate
from agent_core.code_intelligence.model import CodeSymbol
from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.context.model import (
    CodeContextItem,
    CodeContextSegment,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)

SYMBOL_ID = "src/demo.ts::function::target"

SOURCE = (
    "import { request } from './request'\n"
    "const unused = 1\n"
    "const other = 2\n"
    "export function target() {\n"
    "  return request()\n"
    "} // end target\n"
    "const tail = 3\n"
)

WINDOW_CONTENT = (
    "import { request } from './request'\n\n"
    "export function target() {\n"
    "  return request()\n"
    "} // end target"
)


def make_candidate(
    symbol_id: str = SYMBOL_ID,
    path: str = "src/demo.ts",
    kind: str = "function",
    start: int = 4,
    end: int = 6,
    name: str = "target",
) -> SymbolCandidate:
    return SymbolCandidate(
        symbol_id=symbol_id,
        name=name,
        kind=kind,
        relative_path=path,
        start_line=start,
        end_line=end,
        score=1.0,
        source="evidence",
    )


def make_index() -> SymbolIndex:
    index = SymbolIndex()
    index.add_symbol(
        CodeSymbol(
            symbol_id=SYMBOL_ID,
            name="target",
            kind="function",
            relative_path="src/demo.ts",
            start_line=4,
            end_line=6,
        )
    )
    return index


def make_bundle(
    tmp_path: Path,
    item: CodeContextItem,
) -> TaskContextBundle:
    return TaskContextBundle(
        requirement=RequirementSpec(
            title="test",
            summary="test",
        ),
        technical_spec=TechnicalSpec(
            summary="test",
        ),
        task=TaskDefinition(
            task_id="TASK-001",
            task_type="code_change",
            title="test",
            objective="test",
            source_step_order=1,
        ),
        repository=RepositoryProfile(
            project_name="test",
            root_path=str(tmp_path),
        ),
        code_context=[item],
        budget=ContextBudgetUsage(
            max_files=1,
            max_total_chars=10_000,
            max_chars_per_file=10_000,
            used_files=1,
            used_chars=len(item.content),
            symbol_window_files=(
                ["src/demo.ts"] if item.content_mode == "symbol_window" else []
            ),
            symbol_window_saved_chars=max(
                0,
                item.source_file_chars - len(item.content),
            ),
        ),
    )


@pytest.fixture
def valid_case(tmp_path: Path):
    path = tmp_path / "src" / "demo.ts"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE, encoding="utf-8")

    item = CodeContextItem(
        relative_path="src/demo.ts",
        category="source",
        language="TypeScript",
        content_mode="symbol_window",
        content=WINDOW_CONTENT,
        start_line=None,
        end_line=None,
        segments=[
            CodeContextSegment(
                start_line=1,
                end_line=1,
                sources=["import_coverage"],
            ),
            CodeContextSegment(
                start_line=4,
                end_line=6,
                sources=["symbol"],
                symbol_ids=[SYMBOL_ID],
            ),
        ],
        symbol_candidates=[make_candidate()],
        covered_symbol_ids=[SYMBOL_ID],
        localized_symbol_ids=[],
        missing_symbol_ids=[],
        content_chars=len(WINDOW_CONTENT),
        source_file_chars=len(SOURCE),
    )

    bundle = make_bundle(tmp_path, item)

    case = SymbolWindowBenchmarkCase(
        case_id="INTEGRITY",
        required_symbol_ids=[SYMBOL_ID],
        require_independent_symbol_index=True,
    )

    return tmp_path, bundle, case, make_index()


def evaluate(valid_case):
    root, bundle, case, index = valid_case

    return SymbolWindowBenchmarkEvaluator().evaluate(
        case=case,
        bundle=bundle,
        repository_root=root,
        symbol_index=index,
    )


def test_valid_window_passes_independent_verification(valid_case):
    result = evaluate(valid_case)

    assert result.status == "passed"
    assert result.symbol_coverage == 1.0
    assert result.source_integrity_preserved is True
    assert result.metadata_integrity_preserved is True
    assert result.independent_symbol_index_used is True
    assert result.missing_symbol_ids == []


def test_injected_text_is_rejected(valid_case):
    _, bundle, _, _ = valid_case

    # 保持原始源码片段完整，但额外插入不属于源码的文本。
    bundle.code_context[0].content += "\nINJECTED"

    # 即使攻击者同步修正字符统计，也必须失败。
    bundle.code_context[0].content_chars = len(bundle.code_context[0].content)
    bundle.budget.used_chars = len(bundle.code_context[0].content)

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.source_integrity_preserved is False
    assert any("window_content_mismatch" in x for x in result.issues)


def test_reversed_segments_are_rejected(valid_case):
    _, bundle, _, _ = valid_case

    bundle.code_context[0].segments.reverse()

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.source_integrity_preserved is False
    assert any("segment_order_or_overlap" in issue for issue in result.issues)


def test_overlapping_segments_are_rejected(valid_case):
    _, bundle, _, _ = valid_case

    bundle.code_context[0].segments = [
        CodeContextSegment(
            start_line=1,
            end_line=4,
            sources=["import_coverage"],
        ),
        CodeContextSegment(
            start_line=4,
            end_line=6,
            sources=["symbol"],
            symbol_ids=[SYMBOL_ID],
        ),
    ]

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert any("segment_order_or_overlap" in issue for issue in result.issues)


def test_fake_covered_symbol_metadata_is_rejected(valid_case):
    _, bundle, _, _ = valid_case

    bundle.code_context[0].covered_symbol_ids = ["src/demo.ts::function::notReal"]

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.metadata_integrity_preserved is False
    assert any("coverage_metadata_mismatch" in issue for issue in result.issues)


def test_candidate_range_conflicting_with_index_is_rejected(valid_case):
    _, bundle, _, _ = valid_case

    # Builder 对外声称 target 函数仅有 4-5 行；
    # 独立 SymbolIndex 明确它是 4-6 行。
    bundle.code_context[0].symbol_candidates[0].end_line = 5

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.metadata_integrity_preserved is False
    assert any("symbol_candidate_index_mismatch" in issue for issue in result.issues)


def test_semantic_chunk_inflated_end_line_is_rejected(valid_case):
    _, bundle, _, _ = valid_case

    item = bundle.code_context[0]
    lines = SOURCE.split("\n")

    item.content_mode = "semantic_chunk"
    item.segments = []
    item.content = "\n".join(lines[3:6])
    item.start_line = 4
    item.end_line = 7
    item.content_chars = len(item.content)

    bundle.budget.used_chars = len(item.content)

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.source_integrity_preserved is False
    assert any("chunk_end_line_mismatch" in issue for issue in result.issues)


def test_partially_provided_last_line_is_not_full_coverage(valid_case):
    _, bundle, _, _ = valid_case

    item = bundle.code_context[0]
    lines = SOURCE.split("\n")

    # 第 6 行只保留前半段。
    # end_line=6 可以表示最后触及的行号，
    # 但不能表示这一行的代码已经完整提供。
    item.content_mode = "semantic_chunk"
    item.segments = []
    item.content = "\n".join(
        [
            lines[3],
            lines[4],
            "} //",
        ]
    )
    item.start_line = 4
    item.end_line = 6
    item.content_chars = len(item.content)

    bundle.budget.used_chars = len(item.content)

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.symbol_coverage == 0.0
    assert SYMBOL_ID in result.missing_symbol_ids
    assert result.metadata_integrity_preserved is False


def test_stale_repository_source_is_rejected(valid_case):
    root, _, _, _ = valid_case

    path = root / "src" / "demo.ts"
    path.write_text(
        SOURCE.replace("return request()", "return changed()"),
        encoding="utf-8",
    )

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.source_integrity_preserved is False


def test_independent_index_can_be_required(valid_case):
    root, bundle, case, _ = valid_case

    result = SymbolWindowBenchmarkEvaluator().evaluate(
        case=case,
        bundle=bundle,
        repository_root=root,
        symbol_index=None,
    )

    assert result.status == "failed"
    assert result.independent_symbol_index_used is False
    assert "independent_symbol_index_required" in result.issues


def test_budget_misreport_is_rejected(valid_case):
    _, bundle, _, _ = valid_case

    bundle.budget.used_chars = 1

    result = evaluate(valid_case)

    assert result.status == "failed"
    assert result.budget_preserved is False


def test_complete_full_file_is_valid_without_window(
    tmp_path: Path,
):
    path = tmp_path / "src" / "demo.ts"
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE, encoding="utf-8")

    item = CodeContextItem(
        relative_path="src/demo.ts",
        category="source",
        language="TypeScript",
        content_mode="full_file",
        content=SOURCE,
        start_line=1,
        end_line=len(SOURCE.split("\n")),
        symbol_candidates=[make_candidate()],
        covered_symbol_ids=[SYMBOL_ID],
        content_chars=len(SOURCE),
        source_file_chars=len(SOURCE),
    )

    result = SymbolWindowBenchmarkEvaluator().evaluate(
        case=SymbolWindowBenchmarkCase(
            case_id="FULL-FILE",
            required_symbol_ids=[SYMBOL_ID],
            require_symbol_window=False,
            require_independent_symbol_index=True,
        ),
        bundle=make_bundle(tmp_path, item),
        repository_root=tmp_path,
        symbol_index=make_index(),
    )

    assert result.status == "passed"
    assert result.symbol_coverage == 1.0


def test_vue_container_localization_is_not_full_coverage(
    tmp_path: Path,
):
    relative_path = "src/views/OrdersView.vue"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)

    source = (
        '<script setup lang="ts">\n'
        "const orders = []\n"
        "</script>\n"
        "<template>\n"
        "  <div>Orders</div>\n"
        "</template>"
    )
    path.write_text(source, encoding="utf-8")

    symbol_id = "src/views/OrdersView.vue::component::OrdersView"

    index = SymbolIndex()
    index.add_symbol(
        CodeSymbol(
            symbol_id=symbol_id,
            name="OrdersView",
            kind="component",
            relative_path=relative_path,
            start_line=1,
            end_line=6,
        )
    )

    # 只提供脚本内容的一行。
    # Component 被定位，但不能声称整个 SFC 已完整提供。
    item = CodeContextItem(
        relative_path=relative_path,
        category="source",
        language="Vue",
        content_mode="symbol_window",
        content="const orders = []",
        start_line=None,
        end_line=None,
        segments=[
            CodeContextSegment(
                start_line=2,
                end_line=2,
                sources=[
                    "container_evidence",
                    "semantic_evidence",
                ],
                symbol_ids=[symbol_id],
            )
        ],
        symbol_candidates=[
            make_candidate(
                symbol_id=symbol_id,
                path=relative_path,
                kind="component",
                start=1,
                end=6,
                name="OrdersView",
            )
        ],
        covered_symbol_ids=[],
        localized_symbol_ids=[symbol_id],
        missing_symbol_ids=[],
        content_chars=len("const orders = []"),
        source_file_chars=len(source),
    )

    bundle = make_bundle(tmp_path, item)

    result = SymbolWindowBenchmarkEvaluator().evaluate(
        case=SymbolWindowBenchmarkCase(
            case_id="VUE-LOCALIZATION",
            required_symbol_ids=[symbol_id],
            require_independent_symbol_index=True,
        ),
        bundle=bundle,
        repository_root=tmp_path,
        symbol_index=index,
    )

    assert result.status == "passed"
    assert result.covered_symbol_ids == []
    assert result.localized_symbol_ids == [symbol_id]
    assert result.symbol_coverage == 1.0
