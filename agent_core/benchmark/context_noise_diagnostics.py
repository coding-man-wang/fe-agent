from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)

FileOrigin = Literal[
    "retrieval",
    "graph",
    "symbol",
    "unknown",
]


class FileSelectionDiagnostic(BaseModel):
    relative_path: str
    origin: FileOrigin

    context_role: str
    target_match_type: str

    retrieval_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None

    lexical_hit_count: int = 0
    semantic_hit_count: int = 0

    matched_queries: list[str] = Field(default_factory=list)
    matched_intents: list[str] = Field(default_factory=list)
    symbol_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ContextNoiseDiagnostics(BaseModel):
    selected_count: int
    origin_counts: dict[str, int]
    files: list[FileSelectionDiagnostic]


class ContextNoiseDiagnosticBuilder:

    def build(
        self,
        selection_result: TaskContextSelectionResult,
    ) -> ContextNoiseDiagnostics:

        retrieval_paths = {
            item.relative_path
            for item in (selection_result.retrieval_selection.selected_files)
        }

        symbol_added = set(selection_result.symbol_added_files)

        diagnostics = []
        origin_counts = {
            "retrieval": 0,
            "graph": 0,
            "symbol": 0,
            "unknown": 0,
        }

        for item in selection_result.selection.selected_files:
            path = item.relative_path

            if path in retrieval_paths:
                origin: FileOrigin = "retrieval"
            elif path in symbol_added:
                origin = "symbol"
            elif any(
                reason.startswith("Graph dependency:")
                or reason.startswith("Graph dependent:")
                for reason in item.reasons
            ):
                origin = "graph"
            else:
                origin = "unknown"

            origin_counts[origin] += 1

            diagnostics.append(
                FileSelectionDiagnostic(
                    relative_path=path,
                    origin=origin,
                    context_role=item.context_role,
                    target_match_type=item.target_match_type,
                    retrieval_rank=item.retrieval_rank,
                    rerank_rank=item.rerank_rank,
                    rerank_score=item.rerank_score,
                    lexical_hit_count=item.lexical_hit_count,
                    semantic_hit_count=item.semantic_hit_count,
                    matched_queries=list(item.matched_queries),
                    matched_intents=list(item.matched_intents),
                    symbol_ids=[symbol.symbol_id for symbol in item.symbol_candidates],
                    reasons=list(item.reasons),
                )
            )

        return ContextNoiseDiagnostics(
            selected_count=len(diagnostics),
            origin_counts=origin_counts,
            files=diagnostics,
        )
