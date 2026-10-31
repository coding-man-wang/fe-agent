from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from agent_core.code_intelligence.model import (
    ResolvedSymbolReference,
    SymbolGraphEdge,
)


class SymbolGraph:
    """
    Symbol-level Dependency Graph。

    当前只有：
        CALLS

    后续增加：
        REFERENCES
    """

    def __init__(
        self,
    ) -> None:

        self._edges: dict[
            tuple[
                str,
                str,
                str,
                int,
            ],
            SymbolGraphEdge,
        ] = {}

        self._outgoing: dict[
            str,
            list[SymbolGraphEdge],
        ] = defaultdict(list)

        self._incoming: dict[
            str,
            list[SymbolGraphEdge],
        ] = defaultdict(list)

    def add_edge(
        self,
        edge: SymbolGraphEdge,
    ) -> None:

        key = (
            edge.source_symbol_id,
            edge.target_symbol_id,
            edge.relation,
            edge.line_number,
        )

        if key in self._edges:
            return

        self._edges[key] = edge

        self._outgoing[edge.source_symbol_id].append(edge)

        self._incoming[edge.target_symbol_id].append(edge)

    def outgoing(
        self,
        symbol_id: str,
    ) -> list[SymbolGraphEdge]:

        return list(
            self._outgoing.get(
                symbol_id,
                [],
            )
        )

    def incoming(
        self,
        symbol_id: str,
    ) -> list[SymbolGraphEdge]:

        return list(
            self._incoming.get(
                symbol_id,
                [],
            )
        )

    def all_edges(
        self,
    ) -> list[SymbolGraphEdge]:

        return list(self._edges.values())

    def __len__(
        self,
    ) -> int:

        return len(self._edges)


class SymbolCallGraphBuilder:
    """
    Resolved CALL Reference
            ↓
         CALLS Edge
    """

    def build(
        self,
        resolutions: Iterable[ResolvedSymbolReference],
    ) -> SymbolGraph:

        graph = SymbolGraph()

        for resolution in resolutions:

            if resolution.status != "resolved":
                continue

            if resolution.reference_kind != "call":
                continue

            if resolution.source_symbol_id is None:
                # 当前没有 Module Symbol。
                #
                # module-level call
                # 暂不转 Graph Edge。
                continue

            for target_id in resolution.target_symbol_ids:

                graph.add_edge(
                    SymbolGraphEdge(
                        source_symbol_id=(resolution.source_symbol_id),
                        target_symbol_id=(target_id),
                        relation="calls",
                        source_path=(resolution.source_path),
                        line_number=(resolution.line_number),
                    )
                )

        return graph


class SymbolReferenceGraphBuilder:
    """
    所有 resolved Reference
            ↓
    Symbol Graph Edge

    call
        → calls

    read / type_reference
        → references
    """

    def build(
        self,
        resolutions: Iterable[ResolvedSymbolReference],
    ) -> SymbolGraph:

        graph = SymbolGraph()

        for resolution in resolutions:

            if resolution.status != "resolved":
                continue

            if resolution.source_symbol_id is None:
                continue

            relation = (
                "calls" if (resolution.reference_kind == "call") else "references"
            )

            for target_id in resolution.target_symbol_ids:

                graph.add_edge(
                    SymbolGraphEdge(
                        source_symbol_id=(resolution.source_symbol_id),
                        target_symbol_id=(target_id),
                        relation=relation,
                        source_path=(resolution.source_path),
                        line_number=(resolution.line_number),
                    )
                )

        return graph
