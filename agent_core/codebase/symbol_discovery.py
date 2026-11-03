from __future__ import annotations

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)


class RetrievalEvidenceSymbolMapper:
    """
    把已有 File / Chunk Retrieval Evidence
    映射回 Repository Symbol。

    核心链路：

        Retrieval
            ↓
        CodeChunk
            ↓
        Source Location
            ↓
        SymbolIndex
            ↓
        SymbolCandidate

    注意：

    这里不负责：

    - Graph Expansion
    - Seed Eligibility
    - Write Permission
    - Impact Analysis
    """

    def discover(
        self,
        *,
        selection: HybridRelevantFileSelection,
        symbol_index: SymbolIndex,
        limit: int = 20,
    ) -> list[SymbolCandidate]:

        if limit < 0:
            raise ValueError("limit 不能小于 0")

        candidates: dict[
            str,
            SymbolCandidate,
        ] = {}

        for file in selection.selected_files:

            chunk = file.best_semantic_chunk

            # ==================================
            # MVP:
            #
            # 当前只接受有明确 Source Range
            # 的 Semantic Evidence。
            #
            # 不根据“文件被选中”就把整个文件
            # 所有 Symbol 都放进来。
            # ==================================

            if chunk is None:
                continue

            if chunk.relative_path != file.relative_path:
                # Evidence Path 不一致时
                # Fail Close。
                continue

            symbols = symbol_index.find_by_file(file.relative_path)

            for symbol in symbols:

                if not self._overlaps(
                    symbol_start=(symbol.start_line),
                    symbol_end=(symbol.end_line),
                    chunk_start=(chunk.start_line),
                    chunk_end=(chunk.end_line),
                ):
                    continue

                overlap_ratio = self._overlap_ratio(
                    symbol_start=(symbol.start_line),
                    symbol_end=(symbol.end_line),
                    chunk_start=(chunk.start_line),
                    chunk_end=(chunk.end_line),
                )

                semantic_score = file.best_semantic_score

                if semantic_score is None:
                    semantic_score = 0.0

                normalized_semantic_score = self._clamp_score(semantic_score)

                # ==================================
                # Evidence Score
                #
                # Semantic Score:
                #   这个 Chunk 和 Query 多相关
                #
                # Overlap Ratio:
                #   这个 Symbol 有多少实际落在
                #   该 Chunk 中
                #
                # 当前只是确定性的 Evidence Rank，
                # 不作为 Seed Permission。
                # ==================================

                score = normalized_semantic_score * (0.5 + 0.5 * overlap_ratio)

                candidate = SymbolCandidate(
                    symbol_id=(symbol.symbol_id),
                    name=(symbol.name),
                    kind=(symbol.kind),
                    relative_path=(symbol.relative_path),
                    start_line=(symbol.start_line),
                    end_line=(symbol.end_line),
                    score=(self._clamp_score(score)),
                    matched_queries=list(file.matched_queries),
                    reasons=[
                        (
                            "Semantic Retrieval "
                            "Evidence 映射到 Symbol: "
                            f"chunk={chunk.chunk_id}, "
                            f"chunk_lines="
                            f"{chunk.start_line}-"
                            f"{chunk.end_line}, "
                            f"symbol_lines="
                            f"{symbol.start_line}-"
                            f"{symbol.end_line}, "
                            f"overlap_ratio="
                            f"{overlap_ratio:.4f}"
                        )
                    ],
                    source="evidence",
                    depth=0,
                )

                previous = candidates.get(symbol.symbol_id)

                if previous is None or candidate.score > previous.score:
                    candidates[symbol.symbol_id] = candidate

        result = list(candidates.values())

        result.sort(
            key=lambda candidate: (
                -candidate.score,
                (candidate.end_line - candidate.start_line),
                candidate.relative_path,
                candidate.start_line,
                candidate.symbol_id,
            )
        )

        return result[:limit]

    @staticmethod
    def _overlaps(
        *,
        symbol_start: int,
        symbol_end: int,
        chunk_start: int,
        chunk_end: int,
    ) -> bool:

        return not (symbol_end < chunk_start or symbol_start > chunk_end)

    @staticmethod
    def _overlap_ratio(
        *,
        symbol_start: int,
        symbol_end: int,
        chunk_start: int,
        chunk_end: int,
    ) -> float:

        overlap_start = max(
            symbol_start,
            chunk_start,
        )

        overlap_end = min(
            symbol_end,
            chunk_end,
        )

        if overlap_start > overlap_end:
            return 0.0

        overlap_lines = overlap_end - overlap_start + 1

        symbol_lines = symbol_end - symbol_start + 1

        if symbol_lines <= 0:
            return 0.0

        return overlap_lines / symbol_lines

    @staticmethod
    def _clamp_score(
        score: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )
