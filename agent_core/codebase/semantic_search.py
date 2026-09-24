from dataclasses import dataclass

from agent_core.codebase.embedding import (
    EmbeddingProvider,
)
from agent_core.codebase.model import (
    CodeChunk,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from agent_core.codebase.vector import (
    cosine_similarity,
)


@dataclass
class EmbeddedChunk:
    chunk: CodeChunk
    vector: list[float]


class InMemorySemanticIndex:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.entries: list[EmbeddedChunk] = []

    async def build(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            self.entries = []
            return

        vectors = await self.embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )

        if len(vectors) != len(chunks):
            raise ValueError("Embedding 数量与 Chunk 数量不一致")

        self.entries = [
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
            )
            for chunk, vector in zip(
                chunks,
                vectors,
            )
        ]

    async def search(self, query: str, *, top_k: int = 5) -> SemanticSearchResponse:
        if not query.strip():
            raise ValueError("搜索 Query 不能为空")

        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")

        query_vectors = await self.embedding_provider.embed([query])

        if len(query_vectors) != 1:
            raise ValueError("Query Embedding 返回数量错误")

        query_vector = query_vectors[0]

        results: list[SemanticSearchResult] = []

        for entry in self.entries:
            score = cosine_similarity(
                query_vector,
                entry.vector,
            )

            results.append(
                SemanticSearchResult(
                    chunk=entry.chunk,
                    score=score,
                )
            )
        results.sort(key=lambda result: -result.score)

        return SemanticSearchResponse(
            query=query,
            results=results[:top_k],
        )
