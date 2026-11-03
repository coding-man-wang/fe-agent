from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.symbol_graph import (
    SymbolGraph,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)

AdmissionReasonKind = Literal[
    "anchor",
    "graph_corroborated",
    "insufficient_evidence",
]


class SymbolCandidateCorroboration(BaseModel):
    """
    一个 Candidate 被哪个 Anchor 的
    哪条 Graph Edge 支持。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    anchor_symbol_id: str

    candidate_symbol_id: str

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


class SymbolCandidateAdmissionDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    symbol_id: str

    admitted: bool

    reason_kind: AdmissionReasonKind

    reasons: list[str] = Field(
        default_factory=list,
    )

    corroborations: list[SymbolCandidateCorroboration] = Field(
        default_factory=list,
    )


class SymbolCandidateAdmissionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    admitted_candidates: list[SymbolCandidate] = Field(
        default_factory=list,
    )

    rejected_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    decisions: list[SymbolCandidateAdmissionDecision] = Field(
        default_factory=list,
    )


class SymbolCandidateAdmissionGate:
    """
    Candidate Admission / Pruning。

    输入：
        Candidate Generation 的高 Recall Pool

    输出：
        真正允许进入后续 Symbol Retrieval
        Pipeline 的 Candidate。

    MVP Admission Rule：

    1. Anchor 本身 -> Admit

    2. 已生成 Candidate 与任一 Anchor
       存在一跳 calls / references
       -> Admit

    3. 只有 Semantic Chunk Overlap，
       没有第二份结构证据
       -> Reject

    注意：

    Admission 不会：

        - 创建新 Candidate
        - 扩大 Graph
        - 授予 Write Permission
        - 修改 Seed Permission
    """

    def admit(
        self,
        *,
        candidates: list[SymbolCandidate],
        anchor_symbol_ids: set[str],
        symbol_index: SymbolIndex,
        symbol_graph: SymbolGraph,
    ) -> SymbolCandidateAdmissionResult:

        grouped = self._group_candidates(candidates)

        corroborations = self._collect_corroborations(
            anchor_symbol_ids=(anchor_symbol_ids),
            candidate_symbol_ids=set(grouped),
            symbol_index=(symbol_index),
            symbol_graph=(symbol_graph),
        )

        admitted: list[SymbolCandidate] = []

        rejected: list[str] = []

        decisions: list[SymbolCandidateAdmissionDecision] = []

        for symbol_id in sorted(grouped):

            representative = self._representative(grouped[symbol_id])

            if symbol_id in anchor_symbol_ids:

                admitted.append(representative)

                decisions.append(
                    SymbolCandidateAdmissionDecision(
                        symbol_id=(symbol_id),
                        admitted=True,
                        reason_kind="anchor",
                        reasons=[
                            (
                                "Candidate 已通过 "
                                "Bootstrap Seed Gate；"
                                "作为强 Anchor 保留"
                            )
                        ],
                    )
                )

                continue

            symbol_corroborations = corroborations.get(
                symbol_id,
                [],
            )

            if symbol_corroborations:

                admitted.append(representative)

                decisions.append(
                    SymbolCandidateAdmissionDecision(
                        symbol_id=(symbol_id),
                        admitted=True,
                        reason_kind=("graph_corroborated"),
                        reasons=[
                            (
                                "Candidate 已有 Retrieval "
                                "Evidence，且被 Anchor 的"
                                "一跳 SymbolGraph "
                                "Evidence 交叉验证"
                            )
                        ],
                        corroborations=(symbol_corroborations),
                    )
                )

                continue

            rejected.append(symbol_id)

            decisions.append(
                SymbolCandidateAdmissionDecision(
                    symbol_id=(symbol_id),
                    admitted=False,
                    reason_kind=("insufficient_evidence"),
                    reasons=[
                        (
                            "只有 Candidate Generation "
                            "Evidence；没有 Anchor 或 "
                            "Anchor Graph Corroboration，"
                            "Fail Close"
                        )
                    ],
                )
            )

        admitted.sort(key=self._candidate_sort_key)

        return SymbolCandidateAdmissionResult(
            admitted_candidates=(admitted),
            rejected_symbol_ids=(sorted(rejected)),
            decisions=(decisions),
        )

    def _collect_corroborations(
        self,
        *,
        anchor_symbol_ids: set[str],
        candidate_symbol_ids: set[str],
        symbol_index: SymbolIndex,
        symbol_graph: SymbolGraph,
    ) -> dict[
        str,
        list[SymbolCandidateCorroboration],
    ]:

        result: dict[
            str,
            dict[
                tuple,
                SymbolCandidateCorroboration,
            ],
        ] = {}

        for anchor_id in sorted(anchor_symbol_ids):

            if symbol_index.get_symbol(anchor_id) is None:
                continue

            for (
                direction,
                edges,
            ) in (
                (
                    "outgoing",
                    symbol_graph.outgoing(anchor_id),
                ),
                (
                    "incoming",
                    symbol_graph.incoming(anchor_id),
                ),
            ):

                for edge in edges:

                    if edge.relation not in {
                        "calls",
                        "references",
                    }:
                        continue

                    # 两端必须都是当前 Snapshot
                    # 的真实 Symbol。
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

                    # ==================================
                    # Admission 不负责 Expansion。
                    #
                    # Graph 中即使存在另一个 Symbol，
                    # 只要它没有先被 Candidate
                    # Generation 找到，
                    # 就不能在这里凭空创建。
                    # ==================================

                    if related_id not in candidate_symbol_ids:
                        continue

                    if related_id == anchor_id:
                        continue

                    corroboration = SymbolCandidateCorroboration(
                        anchor_symbol_id=(anchor_id),
                        candidate_symbol_id=(related_id),
                        relation=(edge.relation),
                        direction=(direction),
                        line_number=(edge.line_number),
                    )

                    key = (
                        corroboration.anchor_symbol_id,
                        corroboration.candidate_symbol_id,
                        corroboration.relation,
                        corroboration.direction,
                        corroboration.line_number,
                    )

                    result.setdefault(
                        related_id,
                        {},
                    )[key] = corroboration

        return {
            symbol_id: sorted(
                values.values(),
                key=lambda item: (
                    item.anchor_symbol_id,
                    item.direction,
                    item.relation,
                    item.line_number,
                ),
            )
            for (
                symbol_id,
                values,
            ) in result.items()
        }

    @staticmethod
    def _group_candidates(
        candidates: list[SymbolCandidate],
    ) -> dict[
        str,
        list[SymbolCandidate],
    ]:

        result: dict[
            str,
            list[SymbolCandidate],
        ] = {}

        for candidate in candidates:

            result.setdefault(
                candidate.symbol_id,
                [],
            ).append(candidate)

        return result

    @classmethod
    def _representative(
        cls,
        candidates: list[SymbolCandidate],
    ) -> SymbolCandidate:

        return sorted(
            candidates,
            key=(cls._candidate_sort_key),
        )[0]

    @staticmethod
    def _candidate_sort_key(
        candidate: SymbolCandidate,
    ) -> tuple:

        source_priority = {
            "name": 0,
            "evidence": 1,
            "graph": 2,
        }

        return (
            -candidate.score,
            source_priority.get(
                candidate.source,
                99,
            ),
            candidate.relative_path,
            candidate.start_line,
            candidate.symbol_id,
        )
