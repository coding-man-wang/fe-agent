from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from agent_core.benchmark.context_p3_cases import (
    build_cases,
)
from agent_core.benchmark.context_file_quality import (
    ContextFileQualityEvaluator,
)
from agent_core.benchmark.symbol_window_evaluator import (
    SymbolWindowBenchmarkCase,
    SymbolWindowBenchmarkEvaluator,
)
from agent_core.benchmark.task_rerank_diagnostics import (
    TaskRerankDiagnosticRunner,
)
from agent_core.benchmark.evidence_rerank_filter import (
    EvidenceProtectedRerankFilter,
)
from agent_core.codebase.chunker import CodeChunker
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph_evidence_filter import (
    GraphEvidenceFilter,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.codebase.openai_embedding import (
    OpenAIEmbeddingProvider,
)
from agent_core.codebase.openai_reranker import (
    OpenAICompatibleReranker,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.context.builder import ProjectContextBuilder

from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    build_retrieval_service,
    resolve_repository_root,
)
from agent_core.codebase.graph import (
    RepositoryGraphContextExpander,
)
from agent_core.codebase.graph.anchor_dependency_protection import (
    AnchorDependencyProtection,
)


def git_command(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def make_services():
    for name in (
        "EMBEDDING_MODEL_ID",
        "EMBEDDING_API_KEY",
        "RERANK_MODEL_ID",
        "RERANK_API_KEY",
        "RERANK_BASE_URL",
    ):
        if not os.getenv(name):
            raise RuntimeError(f"缺少环境变量: {name}")

    embedding = OpenAIEmbeddingProvider(
        model=os.environ["EMBEDDING_MODEL_ID"],
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        batch_size=20,
        request_timeout_seconds=60,
    )

    semantic_index = InMemorySemanticIndex(embedding)

    selection_service = TaskContextSelectionService(
        file_indexer=RepositoryFileIndexer(),
        retrieval_service=build_retrieval_service(
            semantic_index=semantic_index,
        ),
        graph_expander=RepositoryGraphContextExpander(
            max_files=8,
            max_graph_files=3,
            min_retrieval_files=5,
            dependency_depth=2,
            dependent_depth=1,
            allow_retrieval_replacement=True,
            protect_anchor_dependencies=True,
        ),
        code_chunker=CodeChunker(
            max_lines=40,
            overlap_lines=5,
        ),
        semantic_index=semantic_index,
        enable_symbol_retrieval=True,
    )

    reranker = OpenAICompatibleReranker(
        model=os.environ["RERANK_MODEL_ID"],
        api_key=os.environ["RERANK_API_KEY"],
        base_url=os.environ["RERANK_BASE_URL"],
        request_timeout_seconds=60,
    )

    rerank_runner = TaskRerankDiagnosticRunner(
        reranker=reranker,
        max_document_chars=6000,
    )

    return selection_service, rerank_runner


def evaluate_stage(
    *,
    case,
    stage_name,
    root,
    repository,
    selection,
    symbol_index,
):
    # Oracle 在这里才参与评估，不进入过滤器。
    bundle = ProjectContextBuilder().build(
        repository_root=root,
        requirement=case.requirement,
        technical_spec=case.technical_spec,
        task=case.task,
        repository=repository,
        selection=selection,
        engineering_rules=[],
        max_files=8,
        max_total_chars=40_000,
        max_chars_per_file=10_000,
    )

    file_quality = ContextFileQualityEvaluator().evaluate(
        bundle=bundle,
        oracle=case.oracle,
    )

    symbol_quality = SymbolWindowBenchmarkEvaluator().evaluate(
        case=SymbolWindowBenchmarkCase(
            case_id=f"{case.case_id}-{stage_name}",
            required_symbol_ids=list(case.required_symbol_ids),
            require_symbol_window=False,
            require_independent_symbol_index=True,
        ),
        bundle=bundle,
        repository_root=root,
        symbol_index=symbol_index,
    )

    return {
        "selected_files": [item.relative_path for item in bundle.code_context],
        "context_chars": bundle.budget.used_chars,
        "required_file_recall": file_quality.recall,
        "file_precision": file_quality.precision,
        "missing_required_files": (file_quality.missing_required),
        "irrelevant_chars": (file_quality.irrelevant_chars),
        "missing_required_symbols": (symbol_quality.missing_symbol_ids),
        "required_symbol_coverage": (symbol_quality.symbol_coverage),
        "symbol_evaluation_status": (symbol_quality.status),
        "source_integrity": (symbol_quality.source_integrity_preserved),
        "metadata_integrity": (symbol_quality.metadata_integrity_preserved),
        "budget_integrity": (symbol_quality.budget_preserved),
        "issues": symbol_quality.issues,
    }


async def run_case(
    *,
    case,
    root,
    revision,
    selection_service,
    rerank_runner,
):
    # 每个任务重新生成 Retrieval Query、文件选择、
    # Graph 与 Symbol 候选；只复用相同版本的语义索引。
    result = await selection_service.select(
        repository_root=root,
        repository_revision=revision,
        task=case.task,
        raw_requirement=case.raw_requirement,
        requirement_spec=case.requirement,
    )

    # 记录正式 Selection，用于检测 Shadow 是否有副作用。
    original_dump = result.selection.model_dump(mode="json")

    retrieval_paths = {
        item.relative_path for item in result.retrieval_selection.selected_files
    }

    final_paths = {item.relative_path for item in result.selection.selected_files}

    lost_retrieval_files = sorted(retrieval_paths - final_paths)

    # 独立 SymbolIndex，不使用 Builder 自报的结果。
    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=result.index,
        repository_revision=revision,
    )

    # ======================================
    # P3-D1 Required Evidence Diagnostics
    #
    # Oracle 只用于 Benchmark 诊断。
    # 不参与 Retrieval / Graph / Fusion。
    # ======================================

    retrieval_paths = {
        item.relative_path for item in result.retrieval_selection.selected_files
    }

    final_paths = {item.relative_path for item in result.selection.selected_files}

    expected_paths = set(case.oracle.required_files)

    for symbol_id in case.required_symbol_ids:
        symbol = snapshot.symbol_index.get_symbol(symbol_id)
        if symbol is not None:
            expected_paths.add(symbol.relative_path)

    required_file_diagnostics = []

    for path in sorted(expected_paths):
        in_retrieval = path in retrieval_paths
        in_final = path in final_paths

        if in_retrieval and not in_final:
            state = "lost_after_retrieval"
        elif not in_retrieval and not in_final:
            state = "not_selected_by_pipeline"
        else:
            state = "selected"

        direct_imports = [
            {
                "source": edge.source_path,
                "target": edge.target_path,
                "import_kind": edge.import_kind,
            }
            for edge in result.graph.edges
            if edge.target_path == path
        ]

        required_file_diagnostics.append(
            {
                "path": path,
                "state": state,
                "retrieval_selected": in_retrieval,
                "final_selected": in_final,
                "retrieval_skipped_by_budget": (
                    path in result.retrieval_selection.skipped_by_budget
                ),
                "final_skipped_by_budget": (path in result.selection.skipped_by_budget),
                "direct_imports": direct_imports,
            }
        )

    final_symbol_ids = {
        symbol.symbol_id
        for item in result.selection.selected_files
        for symbol in item.symbol_candidates
    }

    pipeline_symbol_ids = {symbol.symbol_id for symbol in result.symbol_candidates}

    required_symbol_diagnostics = []

    for symbol_id in case.required_symbol_ids:
        indexed = snapshot.symbol_index.get_symbol(symbol_id)

        required_symbol_diagnostics.append(
            {
                "symbol_id": symbol_id,
                "index_resolved": indexed is not None,
                "source_path": (indexed.relative_path if indexed is not None else None),
                "pipeline_candidate": (symbol_id in pipeline_symbol_ids),
                "final_candidate": (symbol_id in final_symbol_ids),
            }
        )

    # 预先定义的 Oracle Symbol 是否存在于 Index。
    unresolved_oracle_symbols = [
        symbol_id
        for symbol_id in case.required_symbol_ids
        if snapshot.symbol_index.get_symbol(symbol_id) is None
    ]

    # P1: Graph Shadow
    graph_proposal = GraphEvidenceFilter().propose(
        selection=result.selection,
    )

    # P2-A: Rerank Shadow，只观察完整结果。
    diagnostic = await rerank_runner.run(
        query=case.raw_requirement,
        repository_root=root,
        selection=graph_proposal.selection,
    )

    dependency_protected = AnchorDependencyProtection(
        anchor_count=5,
        max_depth=2,
    ).protected_paths(
        selection=result.retrieval_selection,
        graph=result.graph,
    )

    # P2-B: Evidence-protected Filter Shadow
    rerank_proposal = EvidenceProtectedRerankFilter().propose(
        selection=graph_proposal.selection,
        diagnostic=diagnostic,
        protected_paths=dependency_protected,
    )

    repository = RepositoryProfile(
        project_name="fe-agent-benchmark-web",
        root_path=str(root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    stages = {
        "original": evaluate_stage(
            case=case,
            stage_name="ORIGINAL",
            root=root,
            repository=repository,
            selection=result.selection,
            symbol_index=snapshot.symbol_index,
        ),
        "graph_shadow": evaluate_stage(
            case=case,
            stage_name="GRAPH",
            root=root,
            repository=repository,
            selection=graph_proposal.selection,
            symbol_index=snapshot.symbol_index,
        ),
        "rerank_shadow": evaluate_stage(
            case=case,
            stage_name="RERANK",
            root=root,
            repository=repository,
            selection=rerank_proposal.selection,
            symbol_index=snapshot.symbol_index,
        ),
    }

    original = stages["original"]
    enhanced = stages["rerank_shadow"]

    unchanged = result.selection.model_dump(mode="json") == original_dump

    lost_protected_retrieval = sorted(set(lost_retrieval_files) & dependency_protected)
    gates = {
        "rerank_diagnostic_complete": (diagnostic.status == "complete"),
        "oracle_symbols_resolved": (not unresolved_oracle_symbols),
        "required_file_recall": (enhanced["required_file_recall"] == 1.0),
        "required_symbol_coverage": (enhanced["required_symbol_coverage"] == 1.0),
        "no_file_recall_regression": (
            enhanced["required_file_recall"] >= original["required_file_recall"]
        ),
        "no_symbol_coverage_regression": (
            enhanced["required_symbol_coverage"] >= original["required_symbol_coverage"]
        ),
        "independent_symbol_evaluation": (
            enhanced["symbol_evaluation_status"] == "passed"
        ),
        "noise_not_increased": (
            enhanced["irrelevant_chars"] <= original["irrelevant_chars"]
        ),
        "context_not_increased": (
            enhanced["context_chars"] <= original["context_chars"]
        ),
        "protected_retrieval_preserved": (not lost_protected_retrieval),
        "shadow_selection_unchanged": unchanged,
        "source_integrity": (enhanced["source_integrity"]),
        "metadata_integrity": (enhanced["metadata_integrity"]),
        "budget_integrity": (enhanced["budget_integrity"]),
    }

    return {
        "case_id": case.case_id,
        "status": ("passed" if all(gates.values()) else "failed"),
        "retrieval_mode": result.retrieval_mode,
        "semantic_error": result.semantic_error,
        "symbol_error": result.symbol_error,
        "unresolved_oracle_symbols": (unresolved_oracle_symbols),
        "graph_proposed_drops": (graph_proposal.dropped_paths),
        "required_evidence_diagnostics": {
            "files": required_file_diagnostics,
            "symbols": required_symbol_diagnostics,
            "symbol_admitted_ids": result.symbol_admitted_ids,
            "symbol_rejected_ids": result.symbol_rejected_ids,
        },
        "rerank_diagnostic_status": diagnostic.status,
        "rerank_issues": diagnostic.issues,
        "rerank_ranking": [
            {
                "path": item.relative_path,
                "rank": item.rank,
                "score": item.score,
            }
            for item in diagnostic.ranked_files
        ],
        "rerank_filter_status": (rerank_proposal.status),
        "rerank_proposed_drops": (rerank_proposal.proposed_drops),
        "retrieval_preservation": {
            "retrieval_files": sorted(retrieval_paths),
            "final_files": sorted(final_paths),
            "lost_files": lost_retrieval_files,
            "lost_protected_files": (lost_protected_retrieval),
        },
        "graph_policy": "anchor_dependency_protected_replacement",
        "stages": stages,
        "completion_gates": gates,
        "dependency_protected_paths": sorted(dependency_protected),
    }


async def main():
    load_dotenv()

    root = resolve_repository_root()
    revision = git_command(root, "rev-parse", "HEAD")

    # 保证当前测试源文件与 Commit 对应。
    dirty = bool(
        git_command(
            root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    )

    if dirty:
        raise RuntimeError("Benchmark 仓库存在未提交修改；" "请先固定测试仓库版本")

    selection_service, rerank_runner = make_services()

    reports = []

    for case in build_cases():
        try:
            report = await run_case(
                case=case,
                root=root,
                revision=revision,
                selection_service=selection_service,
                rerank_runner=rerank_runner,
            )
        except Exception as exc:
            # 某个 Case 失败不能导致其他 Case 被跳过。
            report = {
                "case_id": case.case_id,
                "status": "error",
                "error": (f"{type(exc).__name__}: {exc}"),
            }

        reports.append(report)

    revision_after = git_command(
        root,
        "rev-parse",
        "HEAD",
    )

    overall_passed = (
        all(item["status"] == "passed" for item in reports)
        and revision_after == revision
    )

    output = {
        "version": "v1.4.8-P3",
        "mode": "multi_task_shadow",
        "overall_status": ("passed" if overall_passed else "failed"),
        "repository_revision": revision,
        "repository_revision_unchanged": (revision_after == revision),
        "embedding_model": os.getenv("EMBEDDING_MODEL_ID"),
        "rerank_model": os.getenv("RERANK_MODEL_ID"),
        "total_cases": len(reports),
        "passed_cases": sum(item["status"] == "passed" for item in reports),
        "cases": reports,
        # 尚未通过 Graph-only 必需依赖安全反例
        # 及更大样本集验证，不能升级为生产默认策略。
        "production_ready": False,
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
