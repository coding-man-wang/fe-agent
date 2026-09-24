from agent_core.codebase.model import (
    ExactSearchResponse,
    HybridFileCandidate,
    HybridSearchResponse,
    SemanticSearchResponse,
)


class HybridRetriever:

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k 必须 >= 1")

        if lexical_weight < 0:
            raise ValueError("lexical_weight 不能小于 0")

        if semantic_weight < 0:
            raise ValueError("semantic_weight 不能小于 0")
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def fuse(
        self,
        *,
        query: str,
        lexical: ExactSearchResponse,
        semantic: SemanticSearchResponse,
    ) -> HybridSearchResponse:
        candidates: dict[
            str,
            HybridFileCandidate,
        ] = {}

        self._add_lexical_candidates(
            candidates=candidates,
            response=lexical,
        )

        self._add_semantic_candidates(
            candidates=candidates,
            response=semantic,
        )

        for candidate in candidates.values():

            score = 0.0

            if candidate.lexical_rank is not None:
                score += self.lexical_weight / (self.rrf_k + candidate.lexical_rank)

            if candidate.semantic_rank is not None:
                score += self.semantic_weight / (self.rrf_k + candidate.semantic_rank)

            candidate.fusion_score = score

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                -item.fusion_score,
                item.relative_path,
            ),
        )

        return HybridSearchResponse(
            query=query,
            candidates=ranked,
        )

    def _add_lexical_candidates(
        self,
        *,
        candidates: dict[
            str,
            HybridFileCandidate,
        ],
        response: ExactSearchResponse,
    ) -> None:
        for rank, result in enumerate(
            response.results,
            start=1,
        ):

            candidate = candidates.get(result.relative_path)

            if candidate is None:
                candidate = HybridFileCandidate(
                    relative_path=(result.relative_path),
                    fusion_score=0.0,
                )

                candidates[result.relative_path] = candidate

            candidate.lexical_rank = rank
            candidate.lexical_score = result.score

            candidate.reasons.append(f"Lexical Retrieval rank={rank}")

    def _add_semantic_candidates(
        self,
        *,
        candidates: dict[
            str,
            HybridFileCandidate,
        ],
        response: SemanticSearchResponse,
    ) -> None:

        seen_files: set[str] = set()

        file_rank = 0

        for result in response.results:

            path = result.chunk.relative_path

            if path in seen_files:
                continue

            seen_files.add(path)

            file_rank += 1

            candidate = candidates.get(path)

            if candidate is None:
                candidate = HybridFileCandidate(
                    relative_path=path,
                    fusion_score=0.0,
                )

                candidates[path] = candidate

            candidate.semantic_rank = file_rank

            candidate.semantic_score = result.score

            candidate.best_semantic_chunk = result.chunk

            candidate.reasons.append(
                "Semantic Retrieval " f"rank={file_rank}, " f"score={result.score:.4f}"
            )
