from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)

SeedConfidence = Literal[
    "high",
    "medium",
    "low",
]


class SymbolSeedDecision(BaseModel):
    """
    一个 Symbol 是否具备
    Graph Expansion Permission。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    symbol_id: str

    eligible: bool

    confidence: SeedConfidence

    reasons: list[str] = Field(
        default_factory=list,
    )


class SymbolSeedEligibilityResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    seed_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    decisions: list[SymbolSeedDecision] = Field(
        default_factory=list,
    )


class SymbolSeedEligibilityGate:
    """
    Candidate != Graph Seed。

    Candidate:
        允许作为阅读证据存在。

    Seed:
        允许从这个 Symbol
        启动 SymbolGraph Expansion。

    当前 MVP Policy：

    1. Exact Name Match
       -> High Confidence Seed

    2. Weak Token Match
       -> Candidate only

    3. Evidence Candidate
       -> 必须同时满足：
          - Direct Retrieval File 排名足够高
          - Symbol 属于行为/结构型 Symbol
          - Semantic Chunk 中只有一个结构型 Symbol

    4. Graph Candidate
       -> 永远不能再次成为 Seed
    """

    STRUCTURAL_SEED_KINDS = {
        "function",
        "method",
        "class",
        "component",
    }

    def __init__(
        self,
        *,
        max_evidence_file_rank: int = 3,
    ) -> None:

        if max_evidence_file_rank < 1:
            raise ValueError("max_evidence_file_rank 必须 >= 1")

        self.max_evidence_file_rank = max_evidence_file_rank

    def decide(
        self,
        *,
        candidates: list[SymbolCandidate],
        selection: HybridRelevantFileSelection,
        symbol_index: SymbolIndex,
    ) -> SymbolSeedEligibilityResult:

        # ----------------------------------
        # 一个 Symbol 可能同时具有：
        #
        # name evidence
        # evidence retrieval
        #
        # 所以先按 Symbol 聚合。
        # ----------------------------------

        grouped: dict[
            str,
            list[SymbolCandidate],
        ] = {}

        for candidate in candidates:

            grouped.setdefault(
                candidate.symbol_id,
                [],
            ).append(candidate)

        file_map = {item.relative_path: item for item in selection.selected_files}

        decisions: list[SymbolSeedDecision] = []

        seed_ids: list[str] = []

        for symbol_id in sorted(grouped):

            decision = self._decide_symbol(
                candidates=(grouped[symbol_id]),
                file_map=file_map,
                symbol_index=(symbol_index),
            )

            decisions.append(decision)

            if decision.eligible:
                seed_ids.append(symbol_id)

        return SymbolSeedEligibilityResult(
            seed_symbol_ids=seed_ids,
            decisions=decisions,
        )

    def _decide_symbol(
        self,
        *,
        candidates: list[SymbolCandidate],
        file_map: dict,
        symbol_index: SymbolIndex,
    ) -> SymbolSeedDecision:

        symbol_id = candidates[0].symbol_id

        # ==================================
        # 1. Exact Name Match
        # ==================================

        exact_name = next(
            (
                candidate
                for candidate in candidates
                if (candidate.source == "name" and self._is_exact_name_match(candidate))
            ),
            None,
        )

        if exact_name is not None:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=True,
                confidence="high",
                reasons=["Exact Symbol Name Match；" "允许启动 Graph Expansion"],
            )

        # ==================================
        # 2. Repository Evidence
        #
        # 注意：
        #
        # 即使这个 Symbol 同时有一个
        # weak name candidate，
        # Evidence 仍然可以独立证明它。
        # ==================================

        evidence_candidates = [
            candidate for candidate in candidates if candidate.source == "evidence"
        ]

        if evidence_candidates:

            evidence_candidates.sort(
                key=lambda item: (
                    -item.score,
                    item.relative_path,
                    item.start_line,
                    item.symbol_id,
                )
            )

            return self._decide_evidence(
                candidate=(evidence_candidates[0]),
                file_map=file_map,
                symbol_index=(symbol_index),
            )

        # ==================================
        # 3. Weak Name Match
        # ==================================

        if any(candidate.source == "name" for candidate in candidates):

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=[
                    (
                        "只有 Token Overlap "
                        "等弱 Name Evidence；"
                        "保留 Candidate，"
                        "但不授予 Graph "
                        "Expansion Permission"
                    )
                ],
            )

        # ==================================
        # 4. Graph Candidate
        # ==================================

        return SymbolSeedDecision(
            symbol_id=symbol_id,
            eligible=False,
            confidence="low",
            reasons=[
                "Graph Candidate " "不能再次成为 Seed；" "避免递归 Noise Amplification"
            ],
        )

    def _decide_evidence(
        self,
        *,
        candidate: SymbolCandidate,
        file_map: dict,
        symbol_index: SymbolIndex,
    ) -> SymbolSeedDecision:

        symbol_id = candidate.symbol_id

        # ==================================
        # Symbol Kind Gate
        #
        # variable / type / interface
        # 当前只做 Supporting Evidence。
        # ==================================

        if candidate.kind not in self.STRUCTURAL_SEED_KINDS:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=[
                    (
                        "当前 Symbol Kind "
                        "只作为 Supporting Evidence；"
                        "不作为 MVP Graph Seed"
                    )
                ],
            )

        file = file_map.get(candidate.relative_path)

        if file is None:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=["不存在对应的 Direct " "Retrieval Evidence"],
            )

        chunk = file.best_semantic_chunk

        if chunk is None:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=["不存在 Semantic " "Chunk Evidence"],
            )

        # ==================================
        # Retrieval Confidence
        #
        # 当前 v1.4.5 MVP 使用 Top-3。
        #
        # 这是 Benchmark Policy，
        # 暂时不是长期固定阈值。
        # ==================================

        if file.retrieval_rank > self.max_evidence_file_rank:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=[
                    (
                        "Direct Retrieval "
                        f"rank={file.retrieval_rank}；"
                        "未进入高置信 Evidence "
                        "File 范围"
                    )
                ],
            )

        # ==================================
        # Evidence Ambiguity Gate
        #
        # 关键：
        #
        # 不能根据 Candidate List
        # 判断当前 Chunk 里到底有几个
        # Symbol。
        #
        # Candidate List 可能已经经过
        # limit / top-k。
        #
        # 必须回到 SymbolIndex。
        # ==================================

        structural_symbols = [
            symbol
            for symbol in symbol_index.find_by_file(candidate.relative_path)
            if (
                symbol.kind in self.STRUCTURAL_SEED_KINDS
                and self._overlaps(
                    symbol_start=(symbol.start_line),
                    symbol_end=(symbol.end_line),
                    chunk_start=(chunk.start_line),
                    chunk_end=(chunk.end_line),
                )
            )
        ]

        structural_ids = sorted({symbol.symbol_id for symbol in structural_symbols})

        if symbol_id not in structural_ids:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=[
                    "Candidate 并不属于当前 " "Semantic Chunk 中的" "结构型 Symbol"
                ],
            )

        # ==================================
        # Fail Close
        #
        # 一个 Chunk 同时存在多个
        # function/component/class/method，
        # 当前 Evidence 不足以确定入口。
        # ==================================

        if len(structural_ids) != 1:

            return SymbolSeedDecision(
                symbol_id=symbol_id,
                eligible=False,
                confidence="low",
                reasons=[
                    (
                        "Semantic Chunk "
                        "同时覆盖多个结构型 Symbol；"
                        "存在歧义，Fail Close。"
                        f" symbols={structural_ids}"
                    )
                ],
            )

        return SymbolSeedDecision(
            symbol_id=symbol_id,
            eligible=True,
            confidence="medium",
            reasons=[
                (
                    "高排名 Direct Retrieval "
                    "Evidence + Semantic Chunk "
                    "内唯一结构型 Symbol"
                )
            ],
        )

    @staticmethod
    def _is_exact_name_match(
        candidate: SymbolCandidate,
    ) -> bool:

        return any(
            reason.startswith("符号名称精确匹配:") for reason in candidate.reasons
        )

    @staticmethod
    def _overlaps(
        *,
        symbol_start: int,
        symbol_end: int,
        chunk_start: int,
        chunk_end: int,
    ) -> bool:

        return not (symbol_end < chunk_start or symbol_start > chunk_end)
