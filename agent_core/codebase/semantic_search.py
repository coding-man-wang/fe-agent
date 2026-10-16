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

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        min_score: float | None = None,
    ) -> None:

        self.embedding_provider = embedding_provider

        self.min_score = min_score

        self.entries: list[EmbeddedChunk] = []

        # -------------------------------
        # Query 向量缓存。
        #
        # 同一个 Requirement
        # 会被多个 Task 重复搜索。
        #
        # 不要每次重新调用 Embedding API。
        # -------------------------------

        self.query_vector_cache: dict[
            str,
            list[float],
        ] = {}

    async def build(
        self,
        chunks: list[CodeChunk],
    ) -> None:

        if not chunks:

            self.entries = []

            return

        embedding_texts = [self._build_embedding_text(chunk) for chunk in chunks]

        vectors = await self.embedding_provider.embed(embedding_texts)

        if len(vectors) != len(chunks):

            raise ValueError("Embedding 数量" "与 Chunk 数量不一致")

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

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> SemanticSearchResponse:

        normalized_query = query.strip()

        if not normalized_query:

            raise ValueError("搜索 Query 不能为空")

        if top_k < 1:

            raise ValueError("top_k 必须 >= 1")

        query_vector = self.query_vector_cache.get(normalized_query)

        if query_vector is None:

            query_vectors = await self.embedding_provider.embed([normalized_query])

            if len(query_vectors) != 1:

                raise ValueError("Query Embedding " "返回数量错误")

            query_vector = query_vectors[0]

            self.query_vector_cache[normalized_query] = query_vector

        results: list[SemanticSearchResult] = []

        for entry in self.entries:

            score = cosine_similarity(
                query_vector,
                entry.vector,
            )

            if self.min_score is not None and score < self.min_score:
                continue

            results.append(
                SemanticSearchResult(
                    chunk=entry.chunk,
                    score=score,
                )
            )

        results.sort(key=lambda result: -result.score)

        return SemanticSearchResponse(
            query=query,
            results=(results[:top_k]),
        )

    @staticmethod
    def _build_embedding_text(
        chunk: CodeChunk,
    ) -> str:
        """
        不只 Embedding 源码。

        同时加入：

        - 文件路径
        - 语言
        - 类型
        - 代码

        让类似：

        用户资料表单

        更容易关联：

        UserProfileForm.vue
        """

        language = chunk.language or "unknown"

        return (
            f"path: "
            f"{chunk.relative_path}\n"
            f"language: "
            f"{language}\n"
            f"category: "
            f"{chunk.category}\n"
            f"content:\n"
            f"{chunk.content}"
        )
