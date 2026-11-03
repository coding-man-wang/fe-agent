from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)

SeedConfidence = Literal[
    "high",
    "medium",
    "low",
]


class SymbolSeedDecision(BaseModel):
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
    Candidate ≠ Graph Seed。

    这里只决定：

        是否允许一个 Candidate
        启动 SymbolGraph Expansion。

    不负责：
        - 删除 Candidate
        - 修改 Retrieval Score
        - 修改 Write Permission
        - Impact Analysis
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
        max_evidence_file_rank: int = 2,
    ) -> None:

        if max_evidence_file_rank < 1:
            raise ValueError("max_evidence_file_rank " "必须 >= 1")

        self.max_evidence_file_rank = max_evidence_file_rank

    def decide(
        self,
        *,
        candidates: list[SymbolCandidate],
        selection: HybridRelevantFileSelection,
    ) -> SymbolSeedEligibilityResult:

        decisions: dict[
            str,
            SymbolSeedDecision,
        ] = {}

        seeds: set[str] = set()

        files = {item.relative_path: item for item in selection.selected_files}

        evidence_by_file: dict[
            str,
            list[SymbolCandidate],
        ] = defaultdict(list)

        # ==================================
        # 1. Name Evidence
        # ==================================

        for candidate in candidates:

            if candidate.source == "graph":
                self._record_if_absent(
                    decisions,
                    candidate=candidate,
                    eligible=False,
                    confidence="low",
                    reason=("Graph Candidate " "不能再次成为 Seed"),
                )
                continue

            if candidate.source == "name":

                exact = any(
                    reason.startswith("符号名称精确匹配:")
                    for reason in candidate.reasons
                )

                if exact:
                    seeds.add(candidate.symbol_id)

                    decisions[candidate.symbol_id] = SymbolSeedDecision(
                        symbol_id=(candidate.symbol_id),
                        eligible=True,
                        confidence="high",
                        reasons=["Exact Symbol " "Name Match"],
                    )

                else:
                    self._record_if_absent(
                        decisions,
                        candidate=candidate,
                        eligible=False,
                        confidence="low",
                        reason=(
                            "仅有弱 Token Match，"
                            "保留 Candidate，"
                            "不授予 Graph "
                            "Expansion Permission"
                        ),
                    )

                continue

            # ==================================
            # 2. Evidence Candidate
            # ==================================

            if candidate.source != "evidence":
                continue

            file = files.get(candidate.relative_path)

            if file is None:

                self._record_if_absent(
                    decisions,
                    candidate=candidate,
                    eligible=False,
                    confidence="low",
                    reason=("没有对应的 Direct " "Retrieval Evidence"),
                )

                continue

            if file.best_semantic_chunk is None:

                self._record_if_absent(
                    decisions,
                    candidate=candidate,
                    eligible=False,
                    confidence="low",
                    reason=("没有 Semantic " "Chunk Evidence"),
                )

                continue

            if file.retrieval_rank > self.max_evidence_file_rank:

                self._record_if_absent(
                    decisions,
                    candidate=candidate,
                    eligible=False,
                    confidence="low",
                    reason=("Direct Retrieval " "Evidence 排名不足"),
                )

                continue

            if candidate.kind not in self.STRUCTURAL_SEED_KINDS:

                self._record_if_absent(
                    decisions,
                    candidate=candidate,
                    eligible=False,
                    confidence="low",
                    reason=(
                        "Symbol Kind 仅作为"
                        "Supporting Evidence，"
                        "不作为 MVP Graph Seed"
                    ),
                )

                continue

            evidence_by_file[candidate.relative_path].append(candidate)

        # ==================================
        # 3. Evidence Ambiguity Gate
        #
        # 同一个高置信 Chunk 如果同时
        # 覆盖多个主要结构 Symbol，
        # 当前没有足够 Evidence 判断
        # 哪一个才是任务入口。
        #
        # Fail Close。
        # ==================================

        for path, items in evidence_by_file.items():

            unique = {item.symbol_id: item for item in items}

            values = list(unique.values())

            if len(values) != 1:

                for candidate in values:

                    self._record_if_absent(
                        decisions,
                        candidate=candidate,
                        eligible=False,
                        confidence="low",
                        reason=(
                            "同一 Semantic "
                            f"Evidence 在 {path} "
                            "覆盖多个主要 Symbol，"
                            "当前存在歧义"
                        ),
                    )

                continue

            candidate = values[0]

            # Exact Name 的 high 决策
            # 优先级更高，不覆盖。
            existing = decisions.get(candidate.symbol_id)

            if (
                existing is not None
                and existing.eligible
                and existing.confidence == "high"
            ):
                continue

            seeds.add(candidate.symbol_id)

            decisions[candidate.symbol_id] = SymbolSeedDecision(
                symbol_id=(candidate.symbol_id),
                eligible=True,
                confidence="medium",
                reasons=[
                    "高排名 Direct Retrieval " "Evidence + " "唯一主要结构 Symbol"
                ],
            )

        return SymbolSeedEligibilityResult(
            seed_symbol_ids=sorted(seeds),
            decisions=sorted(
                decisions.values(),
                key=lambda item: (item.symbol_id),
            ),
        )

    @staticmethod
    def _record_if_absent(
        decisions: dict[
            str,
            SymbolSeedDecision,
        ],
        *,
        candidate: SymbolCandidate,
        eligible: bool,
        confidence: SeedConfidence,
        reason: str,
    ) -> None:

        if candidate.symbol_id in decisions:
            return

        decisions[candidate.symbol_id] = SymbolSeedDecision(
            symbol_id=(candidate.symbol_id),
            eligible=eligible,
            confidence=confidence,
            reasons=[reason],
        )
