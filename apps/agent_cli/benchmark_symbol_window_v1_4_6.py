from __future__ import annotations

import asyncio
import json
import os

from pathlib import Path

from tempfile import (
    TemporaryDirectory,
)

from dotenv import (
    load_dotenv,
)

from agent_core.artifacts.query_builder import (
    ArtifactAwareRetrievalQueryBuilder,
)

from agent_core.benchmark.symbol_window_evaluator import (
    SymbolWindowBenchmarkCase,
    SymbolWindowBenchmarkEvaluator,
)

from agent_core.codebase.chunker import (
    CodeChunker,
)

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)

from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)

from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)

from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)

from agent_core.codebase.openai_embedding import (
    OpenAIEmbeddingProvider,
)

from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)

from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)

from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)

from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)

from agent_core.context.builder import (
    ProjectContextBuilder,
)

from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)

from agent_core.code_intelligence.model import CodeSymbol

from agent_core.code_intelligence.symbol_index import SymbolIndex

from agent_core.benchmark.context_comparison_evaluator import (
    ContextComparisonEvaluator,
)

RAW_REQUIREMENT = "给订单列表增加状态筛选，" "不要修改用户列表"


def resolve_repository_root() -> Path:

    configured = os.getenv("FE_AGENT_BENCHMARK_WEB_ROOT")

    if configured:
        root = Path(configured).resolve()

    else:
        agent_root = Path(__file__).resolve().parents[2]

        root = (agent_root.parent / "fe-agent-benchmark-web").resolve()

    if not root.is_dir():
        raise RuntimeError("找不到 " "fe-agent-benchmark-web")

    return root


def build_requirement():
    return RequirementSpec(
        title="订单列表状态筛选",
        summary=("为订单列表增加按状态筛选能力，" "保持现有订单加载行为"),
        goals=[
            "支持订单列表按状态过滤",
            "保持订单数据加载能力",
        ],
        non_goals=[
            "不修改用户列表功能",
        ],
        acceptance_criteria=[
            ("选择订单状态后，" "只展示匹配状态的订单"),
            "订单列表仍能正常加载",
        ],
        constraints=[
            "不要修改用户列表功能",
        ],
    )


def build_technical_spec():
    return TechnicalSpec(
        summary=("在现有订单列表加载链路" "增加状态筛选"),
        solution_approach=[
            "复用现有订单加载入口",
            ("保持 request / mock " "运行路径兼容"),
        ],
        affected_modules=[
            "src/views",
            "src/api",
            "src/mocks",
            "src/types",
        ],
        reuse_plan=["复用现有 listOrders " "和 Order 类型"],
        testing_strategy=["验证订单加载与" "状态筛选行为"],
    )


def build_task():
    return TaskDefinition(
        task_id=("TASK-ORDER-FILTER"),
        task_type="code_change",
        title=("实现订单列表状态筛选"),
        objective=("让订单列表支持按状态过滤，" "保持现有加载行为"),
        source_step_order=1,
        responsibilities=[
            ("定位订单列表的" "数据加载责任"),
            ("实现订单状态过滤行为"),
        ],
        investigation_questions=[
            ("订单列表的数据" "从哪里加载？"),
            ("订单状态在哪里" "定义和展示？"),
        ],
        expected_outputs=["订单状态筛选可用"],
        verification=["验证订单筛选行为"],
    )


def build_retrieval_service(
    *,
    semantic_index,
):
    return TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(semantic_index),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=6,
        lexical_max_results=20,
        semantic_top_k=8,
        max_files=8,
        max_total_bytes=200_000,
        query_builder=(ArtifactAwareRetrievalQueryBuilder(max_queries=6)),
    )


async def run_real_benchmark(
    *,
    include_ab: bool = False,
):

    repository_root = resolve_repository_root()

    embedding_model = os.getenv("EMBEDDING_MODEL_ID")

    embedding_api_key = os.getenv("EMBEDDING_API_KEY")

    if not embedding_model:
        raise RuntimeError("当前 Benchmark 必须配置 " "EMBEDDING_MODEL_ID")

    if not embedding_api_key:
        raise RuntimeError("当前 Benchmark 必须配置 " "EMBEDDING_API_KEY")

    embedding_provider = OpenAIEmbeddingProvider(
        model=(embedding_model),
        api_key=(embedding_api_key),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        batch_size=20,
        request_timeout_seconds=60,
    )

    semantic_index = InMemorySemanticIndex(embedding_provider)

    file_indexer = RepositoryFileIndexer()

    retrieval_service = build_retrieval_service(semantic_index=(semantic_index))

    # ==================================
    # Production v1.4.5 Pipeline
    #
    # Natural Language
    # → Retrieval
    # → Symbol Discovery
    # → Admission
    # → Seed
    # → Graph
    # → Fusion
    # ==================================

    context_selection_service = TaskContextSelectionService(
        file_indexer=(file_indexer),
        retrieval_service=(retrieval_service),
        code_chunker=(
            CodeChunker(
                max_lines=40,
                overlap_lines=5,
            )
        ),
        semantic_index=(semantic_index),
        enable_symbol_retrieval=True,
    )

    requirement = build_requirement()

    technical_spec = build_technical_spec()

    task = build_task()

    context_selection = await context_selection_service.select(
        repository_root=(repository_root),
        repository_revision=("v1.4.6-real-benchmark"),
        task=task,
        raw_requirement=(RAW_REQUIREMENT),
        requirement_spec=(requirement),
    )

    independent_snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=context_selection.index,
        repository_revision=("v1.4.6-independent-audit"),
    )

    repository = RepositoryProfile(
        project_name=("fe-agent-benchmark-web"),
        root_path=str(repository_root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    # ==================================
    # v1.4.6
    #
    # Final Selection
    # → ProjectContextBuilder
    # → Symbol Windows
    # ==================================

    bundle = ProjectContextBuilder().build(
        repository_root=(repository_root),
        requirement=(requirement),
        technical_spec=(technical_spec),
        task=task,
        repository=(repository),
        selection=(context_selection.selection),
        engineering_rules=[],
        max_files=8,
        max_total_chars=40_000,
        max_chars_per_file=10_000,
    )

    comparison = None

    if include_ab:
        baseline_bundle = ProjectContextBuilder(
            enable_symbol_windows=False,
        ).build(
            repository_root=repository_root,
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            selection=context_selection.selection,
            engineering_rules=[],
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )

        # 当前订单场景的候选无关文件。
        # 需要通过人工审查确认后，才作为固定 Oracle。
        irrelevant_paths = {
            "src/components/UserFilters.vue",
            "src/components/UserTable.vue",
        }

        selected_paths = {
            item.relative_path for item in context_selection.selection.selected_files
        }

        irrelevant_paths &= selected_paths

        selected_symbol_ids_for_ab = sorted(
            {
                candidate.symbol_id
                for file in context_selection.selection.selected_files
                for candidate in file.symbol_candidates
            }
        )

        comparison = ContextComparisonEvaluator().evaluate(
            case_id="REAL-ORDER-CONTEXT-AB",
            repository_root=repository_root,
            symbol_index=independent_snapshot.symbol_index,
            required_symbol_ids=selected_symbol_ids_for_ab,
            baseline_bundle=baseline_bundle,
            enhanced_bundle=bundle,
            irrelevant_paths=irrelevant_paths,
        )
    pipeline_symbol_ids = {
        candidate.symbol_id for candidate in context_selection.symbol_candidates
    }

    selected_symbol_ids = {
        candidate.symbol_id
        for file in context_selection.selection.selected_files
        for candidate in file.symbol_candidates
    }

    if not selected_symbol_ids:
        raise RuntimeError(
            "v1.4.6 Benchmark "
            "没有任何进入 Final File Selection "
            "的 Symbol Candidate"
        )

    # ==================================
    # 注意：
    #
    # v1.4.6 的责任不是重新证明：
    #
    #     request 是否应该被 Retrieval 找到
    #
    # 而是证明：
    #
    #     已经真正进入 Final Context Selection
    #     的 Symbol Evidence，
    #     经过压缩后有没有丢失。
    #
    # Symbol-only file 如果因为
    # v1.4.5 File Baseline Budget
    # 没能进入 Selection，
    # 单独作为 diagnostic。
    # ==================================

    evaluation = SymbolWindowBenchmarkEvaluator().evaluate(
        case=(
            SymbolWindowBenchmarkCase(
                case_id=("REAL-ORDER-" "SYMBOL-WINDOW"),
                required_symbol_ids=(sorted(selected_symbol_ids)),
                min_reduction_ratio=0.0,
                require_symbol_window=True,
                require_independent_symbol_index=True,
            )
        ),
        bundle=bundle,
        repository_root=(repository_root),
        symbol_index=independent_snapshot.symbol_index,
    )

    result = {
        "repository_root": str(repository_root),
        "symbol_pipeline": {
            "pipeline_symbol_ids": sorted(pipeline_symbol_ids),
            "selected_symbol_ids": sorted(selected_symbol_ids),
            "symbol_candidates_without_selected_file": (
                sorted(pipeline_symbol_ids - selected_symbol_ids)
            ),
            "seed_symbol_ids": list(context_selection.symbol_seed_ids),
            "admitted_symbol_ids": list(context_selection.symbol_admitted_ids),
            "rejected_symbol_ids": list(context_selection.symbol_rejected_ids),
            "symbol_added_files": list(context_selection.symbol_added_files),
            "symbol_error": (context_selection.symbol_error),
        },
        "context": [
            {
                "relative_path": (item.relative_path),
                "content_mode": (item.content_mode),
                "content_chars": (item.content_chars),
                "source_file_chars": (item.source_file_chars),
                "covered_symbol_ids": (item.covered_symbol_ids),
                "localized_symbol_ids": (item.localized_symbol_ids),
                "missing_symbol_ids": (item.missing_symbol_ids),
                "segments": [
                    segment.model_dump(mode="json") for segment in item.segments
                ],
                "window_issues": (item.window_issues),
            }
            for item in bundle.code_context
        ],
        "budget": (bundle.budget.model_dump(mode="json")),
        "evaluation": (evaluation.model_dump(mode="json")),
    }

    if comparison is not None:
        result["context_ab"] = comparison.model_dump(mode="json")

    return result


def run_synthetic_large_file_benchmark():

    with TemporaryDirectory() as directory:

        root = Path(directory)

        source_path = root / "src" / "large.ts"

        source_path.parent.mkdir(parents=True)

        # ==================================
        # 420-line file
        #
        # target = 301-304
        # ==================================

        lines = [
            ("import { request } " "from './requestClient'"),
            ("import type { Order } " "from './domain'"),
            *[
                f"const filler_{line} = {line}"
                for line in range(
                    3,
                    301,
                )
            ],
            ("export async function " "listOrders(): " "Promise<Order[]> {"),
            ("  const response = await " "request<Order[]>('/orders')"),
            "  return response",
            "}",
            "",
            *[
                f"const filler_{line} = {line}"
                for line in range(
                    306,
                    421,
                )
            ],
        ]

        content = "\n".join(lines)

        source_path.write_text(
            content,
            encoding="utf-8",
        )

        symbol_id = "src/large.ts" "::function::listOrders"

        symbol = SymbolCandidate(
            symbol_id=(symbol_id),
            name="listOrders",
            kind="function",
            relative_path=("src/large.ts"),
            start_line=301,
            end_line=304,
            score=1.0,
            source="evidence",
        )

        independent_index = SymbolIndex()

        independent_index.add_symbol(
            CodeSymbol(
                symbol_id=symbol_id,
                name="listOrders",
                kind="function",
                relative_path="src/large.ts",
                start_line=301,
                end_line=304,
            )
        )

        semantic_chunk = CodeChunk(
            chunk_id=("src/large.ts:" "295-310"),
            relative_path=("src/large.ts"),
            start_line=295,
            end_line=310,
            content="\n".join(lines[294:310]),
            language="TypeScript",
            category="source",
        )

        file = HybridRelevantFile(
            relative_path=("src/large.ts"),
            category="source",
            language="TypeScript",
            size_bytes=len(content.encode("utf-8")),
            retrieval_score=1.0,
            retrieval_rank=1,
            policy_score=0,
            selection_score=1.0,
            context_role="reference",
            symbol_candidates=[symbol],
            best_semantic_chunk=(semantic_chunk),
            best_semantic_score=0.95,
        )

        selection = HybridRelevantFileSelection(
            candidate_count=1,
            eligible_count=1,
            selected_count=1,
            total_selected_bytes=(file.size_bytes),
            selected_files=[file],
        )

        requirement = RequirementSpec(
            title="synthetic",
            summary="synthetic",
        )

        technical_spec = TechnicalSpec(summary="synthetic")

        task = TaskDefinition(
            task_id=("TASK-SYNTHETIC"),
            task_type="code_change",
            title="synthetic",
            objective="synthetic",
            source_step_order=1,
        )

        repository = RepositoryProfile(
            project_name=("synthetic"),
            root_path=str(root),
            language="TypeScript",
        )

        bundle = ProjectContextBuilder().build(
            repository_root=root,
            requirement=(requirement),
            technical_spec=(technical_spec),
            task=task,
            repository=(repository),
            selection=(selection),
            max_files=1,
            max_total_chars=5_000,
            max_chars_per_file=3_000,
        )

        evaluation = SymbolWindowBenchmarkEvaluator().evaluate(
            case=(
                SymbolWindowBenchmarkCase(
                    case_id=("SYNTHETIC-" "LARGE-FILE-" "SYMBOL-WINDOW"),
                    required_symbol_ids=[symbol_id],
                    min_reduction_ratio=0.70,
                    require_symbol_window=True,
                    require_independent_symbol_index=True,
                )
            ),
            bundle=bundle,
            repository_root=root,
            symbol_index=independent_index,
        )

        return {
            "context": [
                {
                    "relative_path": (item.relative_path),
                    "content_mode": (item.content_mode),
                    "content_chars": (item.content_chars),
                    "source_file_chars": (item.source_file_chars),
                    "covered_symbol_ids": (item.covered_symbol_ids),
                    "segments": [
                        segment.model_dump(mode="json") for segment in item.segments
                    ],
                }
                for item in bundle.code_context
            ],
            "budget": (bundle.budget.model_dump(mode="json")),
            "evaluation": (evaluation.model_dump(mode="json")),
        }


async def main():

    load_dotenv()

    real = await run_real_benchmark()

    synthetic = run_synthetic_large_file_benchmark()

    overall_status = (
        "passed"
        if (
            real["evaluation"]["status"] == "passed"
            and synthetic["evaluation"]["status"] == "passed"
        )
        else "failed"
    )

    output = {
        "version": "v1.4.6",
        "overall_status": (overall_status),
        "real_repository": (real),
        "synthetic_large_file": (synthetic),
    }

    print("=" * 80)

    print("v1.4.6 " "Symbol Window Context Benchmark")

    print("=" * 80)

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
