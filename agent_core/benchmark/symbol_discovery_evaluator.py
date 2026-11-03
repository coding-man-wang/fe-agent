from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.retrieval_query import (
    RetrievalQuery,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)


class SymbolDiscoveryEvaluation(BaseModel):
    """
    Natural-language → Symbol Discovery
    的诊断结果。

    注意：

    这里只做诊断。

    不负责：
        - 修改 Query
        - 修改 Ranker
        - 猜测 Symbol 名
        - 修改 Graph
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    target_symbol_id: str

    target_symbol_name: str

    generated_queries: list[str] = Field(
        default_factory=list,
    )

    natural_language_direct_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    natural_language_expanded_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    control_direct_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    control_expanded_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # QueryBuilder 是否自己产生了目标代码标识符。
    query_contains_target_identifier: bool

    # 自然语言路径是否真正发现目标。
    natural_language_direct_hit: bool

    natural_language_expanded_hit: bool

    # 已知 identifier 控制组是否正常。
    control_direct_hit: bool

    control_expanded_hit: bool

    # ==================================
    # Discovery Gap
    #
    # 控制组正常，
    # 但自然语言无法找到目标。
    #
    # 说明缺口出现在：
    #
    # Natural Language
    #       ↓
    # Symbol Candidate
    #
    # 而不是：
    #
    # SymbolIndex / Graph
    # ==================================

    discovery_gap_detected: bool


class SymbolDiscoveryEvaluator:
    """
    只比较 Natural-language 路径
    与 Known-Identifier Control。

    不参与生产 Retrieval 决策。
    """

    def evaluate(
        self,
        *,
        target_symbol_id: str,
        target_symbol_name: str,
        query_specs: list[RetrievalQuery],
        natural_direct: list[SymbolCandidate],
        natural_expanded: list[SymbolCandidate],
        control_direct: list[SymbolCandidate],
        control_expanded: list[SymbolCandidate],
    ) -> SymbolDiscoveryEvaluation:

        generated_queries = [query.text for query in query_specs]

        natural_direct_ids = self._ids(natural_direct)

        natural_expanded_ids = self._ids(natural_expanded)

        control_direct_ids = self._ids(control_direct)

        control_expanded_ids = self._ids(control_expanded)

        query_contains_target = any(
            target_symbol_name.casefold() in query.casefold()
            for query in generated_queries
        )

        natural_direct_hit = target_symbol_id in natural_direct_ids

        natural_expanded_hit = target_symbol_id in natural_expanded_ids

        control_direct_hit = target_symbol_id in control_direct_ids

        control_expanded_hit = target_symbol_id in control_expanded_ids

        discovery_gap = (
            control_direct_hit and control_expanded_hit and not natural_expanded_hit
        )

        return SymbolDiscoveryEvaluation(
            target_symbol_id=(target_symbol_id),
            target_symbol_name=(target_symbol_name),
            generated_queries=(generated_queries),
            natural_language_direct_symbol_ids=(natural_direct_ids),
            natural_language_expanded_symbol_ids=(natural_expanded_ids),
            control_direct_symbol_ids=(control_direct_ids),
            control_expanded_symbol_ids=(control_expanded_ids),
            query_contains_target_identifier=(query_contains_target),
            natural_language_direct_hit=(natural_direct_hit),
            natural_language_expanded_hit=(natural_expanded_hit),
            control_direct_hit=(control_direct_hit),
            control_expanded_hit=(control_expanded_hit),
            discovery_gap_detected=(discovery_gap),
        )

    @staticmethod
    def _ids(
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
