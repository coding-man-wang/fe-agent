from __future__ import annotations

from time import perf_counter
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.model import (
    RepositoryFileIndex,
)
from agent_core.codebase.symbol_retrieval import (
    SymbolGraphExpander,
    SymbolRanker,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)

SymbolRetrievalBenchmarkStatus = Literal[
    "passed",
    "partial",
    "failed",
]


class SymbolRetrievalBenchmarkCase(BaseModel):
    """
    Symbol Retrieval Benchmark 的
    Ground Truth。

    expected_symbol_ids:
        当前任务真正应该找到的 Symbol。

    forbidden_symbol_ids:
        当前任务明确不应该进入的 Symbol。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    queries: list[str] = Field(
        min_length=1,
    )

    expected_symbol_ids: list[str] = Field(
        min_length=1,
    )

    forbidden_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Benchmark 可以控制检索预算，
    # 但不能修改 Ranker 算法。
    rank_limit: int = Field(
        default=20,
        ge=1,
    )

    max_graph_neighbors: int = Field(
        default=20,
        ge=0,
    )


class SymbolRetrievalBenchmarkEvaluation(BaseModel):
    """
    Symbol Retrieval Benchmark 的
    定量结果。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str

    status: SymbolRetrievalBenchmarkStatus

    expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    direct_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    expanded_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    matched_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    missed_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    unexpected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    forbidden_hits: list[str] = Field(
        default_factory=list,
    )

    graph_added_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    graph_recovered_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    graph_noise_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # ==================================
    # Quality Metrics
    # ==================================

    direct_recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    symbol_recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    symbol_precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    graph_expansion_gain: float = Field(
        ge=0.0,
        le=1.0,
    )

    noise_expansion_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ==================================
    # Cost Metrics
    # ==================================

    symbol_snapshot_build_ms: float = Field(
        ge=0.0,
    )

    symbol_count: int = Field(
        ge=0,
    )

    graph_edge_count: int = Field(
        ge=0,
    )

    # ==================================
    # File Baseline Safety
    # ==================================

    baseline_file_paths: list[str] = Field(
        default_factory=list,
    )

    final_file_paths: list[str] = Field(
        default_factory=list,
    )

    baseline_preserved: bool = True


class SymbolRetrievalBenchmarkEvaluator:
    """
    Benchmark Evaluator。

    只负责评分。

    不参与：
        Ranking
        Graph Expansion
        Result Fusion

    防止 Benchmark 反向改变算法。
    """

    def evaluate(
        self,
        *,
        case: SymbolRetrievalBenchmarkCase,
        direct_candidates: list[SymbolCandidate],
        expanded_candidates: list[SymbolCandidate],
        symbol_snapshot_build_ms: float = 0.0,
        symbol_count: int = 0,
        graph_edge_count: int = 0,
        baseline_file_paths: list[str] | None = None,
        final_file_paths: list[str] | None = None,
    ) -> SymbolRetrievalBenchmarkEvaluation:

        direct_ids = self._unique_ids(direct_candidates)

        expanded_ids = self._unique_ids(expanded_candidates)

        expected = set(case.expected_symbol_ids)

        forbidden = set(case.forbidden_symbol_ids)

        direct_set = set(direct_ids)

        expanded_set = set(expanded_ids)

        matched = expected & expanded_set

        missed = expected - expanded_set

        unexpected = expanded_set - expected

        forbidden_hits = expanded_set & forbidden

        graph_added_ids = [
            symbol_id for symbol_id in expanded_ids if symbol_id not in direct_set
        ]

        graph_added_set = set(graph_added_ids)

        graph_recovered = graph_added_set & expected

        graph_noise = graph_added_set - expected

        # ==================================
        # Recall
        # ==================================

        direct_recall = len(direct_set & expected) / len(expected)

        symbol_recall = len(matched) / len(expected)

        # ==================================
        # Precision
        # ==================================

        if expanded_set:

            symbol_precision = len(matched) / len(expanded_set)

        else:

            symbol_precision = 0.0

        # ==================================
        # Graph Expansion Gain
        #
        # 图扩展前后的 Recall 增量。
        # ==================================

        graph_expansion_gain = max(
            0.0,
            symbol_recall - direct_recall,
        )

        # ==================================
        # Noise Expansion Rate
        #
        # Graph 新增结果里，
        # 有多少不属于 Ground Truth。
        # ==================================

        if graph_added_set:

            noise_expansion_rate = len(graph_noise) / len(graph_added_set)

        else:

            noise_expansion_rate = 0.0

        baseline = list(baseline_file_paths or [])

        final_files = list(final_file_paths or [])

        if baseline:

            # 当前 SymbolResultFusion 的
            # Invariant：
            #
            # 原文件集合与顺序必须保持为
            # Final Selection 的前缀。
            baseline_preserved = final_files[: len(baseline)] == baseline

        else:

            baseline_preserved = True

        # ==================================
        # Status
        #
        # Precision 暂时不设硬阈值。
        # 第一轮先测量真实数据。
        # ==================================

        if forbidden_hits or not baseline_preserved or symbol_recall == 0.0:

            status = "failed"

        elif symbol_recall == 1.0:

            status = "passed"

        else:

            status = "partial"

        return SymbolRetrievalBenchmarkEvaluation(
            case_id=case.case_id,
            status=status,
            expected_symbol_ids=sorted(expected),
            direct_symbol_ids=(direct_ids),
            expanded_symbol_ids=(expanded_ids),
            matched_expected_symbol_ids=sorted(matched),
            missed_expected_symbol_ids=sorted(missed),
            unexpected_symbol_ids=sorted(unexpected),
            forbidden_hits=sorted(forbidden_hits),
            graph_added_symbol_ids=(graph_added_ids),
            graph_recovered_expected_symbol_ids=sorted(graph_recovered),
            graph_noise_symbol_ids=sorted(graph_noise),
            direct_recall=(direct_recall),
            symbol_recall=(symbol_recall),
            symbol_precision=(symbol_precision),
            graph_expansion_gain=(graph_expansion_gain),
            noise_expansion_rate=(noise_expansion_rate),
            symbol_snapshot_build_ms=(symbol_snapshot_build_ms),
            symbol_count=(symbol_count),
            graph_edge_count=(graph_edge_count),
            baseline_file_paths=(baseline),
            final_file_paths=(final_files),
            baseline_preserved=(baseline_preserved),
        )

    @staticmethod
    def _unique_ids(
        candidates: list[SymbolCandidate],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for candidate in candidates:

            if candidate.symbol_id in seen:
                continue

            seen.add(candidate.symbol_id)

            result.append(candidate.symbol_id)

        return result


class SymbolRetrievalBenchmarkRunner:
    """
    执行真实 Symbol Retrieval Pipeline：

        Repository
            ↓
        Symbol Snapshot
            ↓
        Symbol Ranker
            ↓
        Graph Expansion
            ↓
        Benchmark Evaluator
    """

    def __init__(
        self,
        *,
        snapshot_builder: RepositorySymbolSnapshotBuilder | None = None,
        ranker: SymbolRanker | None = None,
        graph_expander: SymbolGraphExpander | None = None,
        evaluator: SymbolRetrievalBenchmarkEvaluator | None = None,
    ) -> None:

        self.snapshot_builder = snapshot_builder or RepositorySymbolSnapshotBuilder()

        self.ranker = ranker or SymbolRanker()

        self.graph_expander = graph_expander or SymbolGraphExpander()

        self.evaluator = evaluator or SymbolRetrievalBenchmarkEvaluator()

    def run(
        self,
        *,
        case: SymbolRetrievalBenchmarkCase,
        file_index: RepositoryFileIndex,
        repository_revision: str,
        aliases: dict[str, str] | None = None,
        baseline_file_paths: list[str] | None = None,
        final_file_paths: list[str] | None = None,
    ) -> SymbolRetrievalBenchmarkEvaluation:

        started = perf_counter()

        snapshot = self.snapshot_builder.build(
            file_index=file_index,
            repository_revision=(repository_revision),
            aliases=aliases,
        )

        build_ms = (perf_counter() - started) * 1000.0

        direct_candidates = self.ranker.rank(
            queries=case.queries,
            symbol_index=(snapshot.symbol_index),
            limit=(case.rank_limit),
        )

        expanded_candidates = self.graph_expander.expand(
            candidates=(direct_candidates),
            symbol_index=(snapshot.symbol_index),
            symbol_graph=(snapshot.symbol_graph),
            max_neighbors=(case.max_graph_neighbors),
        )

        return self.evaluator.evaluate(
            case=case,
            direct_candidates=(direct_candidates),
            expanded_candidates=(expanded_candidates),
            symbol_snapshot_build_ms=(build_ms),
            symbol_count=len(snapshot.symbol_index),
            graph_edge_count=len(snapshot.symbol_graph),
            baseline_file_paths=(baseline_file_paths),
            final_file_paths=(final_file_paths),
        )
