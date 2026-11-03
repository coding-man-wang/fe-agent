from __future__ import annotations

from time import perf_counter
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
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
from agent_core.code_intelligence.symbol_graph import (
    SymbolGraph,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)

SymbolRetrievalBenchmarkStatus = Literal[
    "passed",
    "partial",
    "failed",
]


class SymbolGraphProvenance(BaseModel):
    """
    Benchmark-only Graph Evidence。

    一个 Symbol 即使已经被 Semantic Evidence 找到，
    仍然可能同时被 SymbolGraph 支持。

    例如：

        listOrders
            --CALLS-->
        mockListOrders

    即使 mockListOrders 已经是 Evidence Candidate，
    这条 Graph Evidence 仍然应该被 Benchmark 看见。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    seed_symbol_id: str

    related_symbol_id: str

    relation: Literal[
        "calls",
        "references",
    ]

    direction: Literal[
        "outgoing",
        "incoming",
    ]

    line_number: int = Field(
        ge=1,
    )


class SymbolRetrievalBenchmarkCase(BaseModel):
    """
    Symbol Retrieval Ground Truth。

    新模型：

        Required
        Supporting
        Forbidden

    expected_symbol_ids：
        v1.4.5D-3 之前的兼容字段。

        如果没有 required_symbol_ids，
        expected_symbol_ids 自动视为 Required。
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

    # ==================================
    # Legacy Ground Truth
    # ==================================

    expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # ==================================
    # Tiered Ground Truth
    # ==================================

    required_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    supporting_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    forbidden_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    rank_limit: int = Field(
        default=20,
        ge=1,
    )

    max_graph_neighbors: int = Field(
        default=20,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_ground_truth(
        self,
    ) -> "SymbolRetrievalBenchmarkCase":

        required = set(self.effective_required_symbol_ids())

        supporting = set(self.supporting_symbol_ids)

        forbidden = set(self.forbidden_symbol_ids)

        if not required:
            raise ValueError("Benchmark 至少需要一个 " "Required Symbol")

        # 同时提供 legacy expected
        # 和 required 时必须表达同一集合，
        # 防止两个 Ground Truth 冲突。
        if (
            self.expected_symbol_ids
            and self.required_symbol_ids
            and (set(self.expected_symbol_ids) != set(self.required_symbol_ids))
        ):
            raise ValueError("expected_symbol_ids 与 " "required_symbol_ids 冲突")

        if required & supporting:
            raise ValueError("Required 与 Supporting " "Symbol 不能重叠")

        if required & forbidden:
            raise ValueError("Required 与 Forbidden " "Symbol 不能重叠")

        if supporting & forbidden:
            raise ValueError("Supporting 与 Forbidden " "Symbol 不能重叠")

        return self

    def effective_required_symbol_ids(
        self,
    ) -> list[str]:

        if self.required_symbol_ids:
            return list(self.required_symbol_ids)

        return list(self.expected_symbol_ids)


class SymbolRetrievalBenchmarkEvaluation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    case_id: str

    status: SymbolRetrievalBenchmarkStatus

    # ==================================
    # Ground Truth
    # ==================================

    # Legacy compatibility：
    # 始终等价于 Required。
    expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    required_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    supporting_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    forbidden_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # ==================================
    # Retrieval Results
    # ==================================

    direct_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    expanded_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    matched_required_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    missed_required_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    matched_supporting_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Legacy：
    # matched_expected == matched_required
    matched_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    missed_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Strict：
    # 所有非 Required。
    unexpected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Harmful / non-useful：
    # 既不是 Required，也不是 Supporting。
    harmful_unexpected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    forbidden_hits: list[str] = Field(
        default_factory=list,
    )

    # ==================================
    # Graph
    # ==================================

    graph_added_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    graph_recovered_required_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Legacy compatibility
    graph_recovered_expected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Strict Noise：
    # Graph 新增但非 Required。
    graph_noise_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Harmful Noise：
    # Graph 新增且既不是 Required
    # 也不是 Supporting。
    harmful_graph_noise_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # 即使 Symbol 已经是 direct/evidence candidate，
    # 仍记录 Graph 是否提供独立支持。
    graph_provenance: list[SymbolGraphProvenance] = Field(
        default_factory=list,
    )

    # ==================================
    # Recall
    # ==================================

    direct_recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    symbol_recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    graph_expansion_gain: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ==================================
    # Precision
    # ==================================

    # Legacy：
    # 等价于 strict_precision。
    symbol_precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    strict_precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    useful_precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ==================================
    # Graph Noise
    # ==================================

    # Legacy：
    # 等价于 strict_noise_expansion_rate。
    noise_expansion_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    strict_noise_expansion_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    harmful_noise_expansion_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    # ==================================
    # Cost
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


class SymbolGraphProvenanceCollector:
    """
    Benchmark-only Graph inspection。

    不改变 Graph Expansion。

    目的：

    即使某个 Symbol 在 Graph Expansion 前
    已经存在于 Candidate Pool，

    我们仍然知道：

        Graph 有没有独立支持这个 Symbol？
    """

    def collect(
        self,
        *,
        seed_symbol_ids: set[str],
        symbol_index: SymbolIndex,
        symbol_graph: SymbolGraph,
    ) -> list[SymbolGraphProvenance]:

        result: dict[
            tuple,
            SymbolGraphProvenance,
        ] = {}

        for seed_id in sorted(seed_symbol_ids):

            if symbol_index.get_symbol(seed_id) is None:
                continue

            for (
                direction,
                edges,
            ) in (
                (
                    "outgoing",
                    symbol_graph.outgoing(seed_id),
                ),
                (
                    "incoming",
                    symbol_graph.incoming(seed_id),
                ),
            ):

                for edge in edges:

                    if edge.relation not in {
                        "calls",
                        "references",
                    }:
                        continue

                    if (
                        symbol_index.get_symbol(edge.source_symbol_id) is None
                        or symbol_index.get_symbol(edge.target_symbol_id) is None
                    ):
                        continue

                    related_id = (
                        edge.target_symbol_id
                        if direction == "outgoing"
                        else edge.source_symbol_id
                    )

                    provenance = SymbolGraphProvenance(
                        seed_symbol_id=(seed_id),
                        related_symbol_id=(related_id),
                        relation=(edge.relation),
                        direction=(direction),
                        line_number=(edge.line_number),
                    )

                    key = (
                        provenance.seed_symbol_id,
                        provenance.related_symbol_id,
                        provenance.relation,
                        provenance.direction,
                        provenance.line_number,
                    )

                    result[key] = provenance

        return sorted(
            result.values(),
            key=lambda item: (
                item.seed_symbol_id,
                item.direction,
                item.relation,
                item.related_symbol_id,
                item.line_number,
            ),
        )


class SymbolRetrievalBenchmarkEvaluator:
    """
    只评分，不参与 Retrieval 决策。
    """

    def evaluate(
        self,
        *,
        case: SymbolRetrievalBenchmarkCase,
        direct_candidates: list[SymbolCandidate],
        expanded_candidates: list[SymbolCandidate],
        graph_provenance: list[SymbolGraphProvenance] | None = None,
        symbol_snapshot_build_ms: float = 0.0,
        symbol_count: int = 0,
        graph_edge_count: int = 0,
        baseline_file_paths: list[str] | None = None,
        final_file_paths: list[str] | None = None,
    ) -> SymbolRetrievalBenchmarkEvaluation:

        direct_ids = self._unique_ids(direct_candidates)

        expanded_ids = self._unique_ids(expanded_candidates)

        required = set(case.effective_required_symbol_ids())

        supporting = set(case.supporting_symbol_ids)

        useful = required | supporting

        forbidden = set(case.forbidden_symbol_ids)

        direct_set = set(direct_ids)

        expanded_set = set(expanded_ids)

        # ==================================
        # Ground Truth Matching
        # ==================================

        matched_required = required & expanded_set

        missed_required = required - expanded_set

        matched_supporting = supporting & expanded_set

        strict_unexpected = expanded_set - required

        harmful_unexpected = expanded_set - useful

        forbidden_hits = expanded_set & forbidden

        # ==================================
        # Graph Delta
        # ==================================

        graph_added_ids = [
            symbol_id for symbol_id in expanded_ids if symbol_id not in direct_set
        ]

        graph_added_set = set(graph_added_ids)

        graph_recovered_required = graph_added_set & required

        strict_graph_noise = graph_added_set - required

        harmful_graph_noise = graph_added_set - useful

        # ==================================
        # Recall
        # ==================================

        direct_recall = len(direct_set & required) / len(required)

        symbol_recall = len(matched_required) / len(required)

        graph_expansion_gain = max(
            0.0,
            symbol_recall - direct_recall,
        )

        # ==================================
        # Precision
        # ==================================

        if expanded_set:

            strict_precision = len(expanded_set & required) / len(expanded_set)

            useful_precision = len(expanded_set & useful) / len(expanded_set)

        else:

            strict_precision = 0.0

            useful_precision = 0.0

        # ==================================
        # Graph Noise
        # ==================================

        if graph_added_set:

            strict_noise_rate = len(strict_graph_noise) / len(graph_added_set)

            harmful_noise_rate = len(harmful_graph_noise) / len(graph_added_set)

        else:

            strict_noise_rate = 0.0

            harmful_noise_rate = 0.0

        # ==================================
        # File Baseline
        # ==================================

        baseline = list(baseline_file_paths or [])

        final_files = list(final_file_paths or [])

        if baseline:

            baseline_preserved = final_files[: len(baseline)] == baseline

        else:

            baseline_preserved = True

        # ==================================
        # Status
        #
        # Supporting 不作为硬 Gate。
        #
        # Precision 暂时继续测量，
        # 不直接设阈值。
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
            # Ground Truth
            expected_symbol_ids=sorted(required),
            required_symbol_ids=sorted(required),
            supporting_symbol_ids=sorted(supporting),
            forbidden_symbol_ids=sorted(forbidden),
            # Result
            direct_symbol_ids=(direct_ids),
            expanded_symbol_ids=(expanded_ids),
            matched_required_symbol_ids=sorted(matched_required),
            missed_required_symbol_ids=sorted(missed_required),
            matched_supporting_symbol_ids=sorted(matched_supporting),
            # Legacy
            matched_expected_symbol_ids=sorted(matched_required),
            missed_expected_symbol_ids=sorted(missed_required),
            unexpected_symbol_ids=sorted(strict_unexpected),
            harmful_unexpected_symbol_ids=sorted(harmful_unexpected),
            forbidden_hits=sorted(forbidden_hits),
            # Graph
            graph_added_symbol_ids=(graph_added_ids),
            graph_recovered_required_symbol_ids=sorted(graph_recovered_required),
            graph_recovered_expected_symbol_ids=sorted(graph_recovered_required),
            graph_noise_symbol_ids=sorted(strict_graph_noise),
            harmful_graph_noise_symbol_ids=sorted(harmful_graph_noise),
            graph_provenance=list(graph_provenance or []),
            # Recall
            direct_recall=(direct_recall),
            symbol_recall=(symbol_recall),
            graph_expansion_gain=(graph_expansion_gain),
            # Precision
            symbol_precision=(strict_precision),
            strict_precision=(strict_precision),
            useful_precision=(useful_precision),
            # Noise
            noise_expansion_rate=(strict_noise_rate),
            strict_noise_expansion_rate=(strict_noise_rate),
            harmful_noise_expansion_rate=(harmful_noise_rate),
            # Cost
            symbol_snapshot_build_ms=(symbol_snapshot_build_ms),
            symbol_count=(symbol_count),
            graph_edge_count=(graph_edge_count),
            # File safety
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
    保持旧 Benchmark Runner 行为。

    D-3 只升级 Evaluation Contract，
    不在这里引入 Seed Gate。
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
            file_index=(file_index),
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
