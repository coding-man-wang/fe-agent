from pathlib import Path

import pytest

from agent_core.codebase.chunker import (
    CodeChunker,
)
from agent_core.codebase.embedding import (
    EmbeddingProvider,
)
from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)
from agent_core.codebase.semantic_search import (
    InMemorySemanticIndex,
)


class FakeEmbeddingProvider(EmbeddingProvider):

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []

        for text in texts:
            normalized = text.lower()

            vector = [
                1.0 if ("user" in normalized or "用户" in normalized) else 0.0,
                (
                    1.0
                    if (
                        "auth" in normalized
                        or "login" in normalized
                        or "登录" in normalized
                    )
                    else 0.0
                ),
                1.0 if ("payment" in normalized or "支付" in normalized) else 0.0,
            ]

            vectors.append(vector)

        return vectors


def create_test_repository(
    root: Path,
) -> None:
    src = root / "src"
    src.mkdir()

    (src / "user.ts").write_text(
        """
export async function getUserList() {
    return fetch('/api/users')
}
""".strip(),
        encoding="utf-8",
    )

    (src / "auth.ts").write_text(
        """
export function handleUnauthorized() {
    // login authentication handling
}
""".strip(),
        encoding="utf-8",
    )

    (src / "payment.ts").write_text(
        """
export function createPayment() {
    // payment processing
}
""".strip(),
        encoding="utf-8",
    )


async def build_retrieval_components(
    root: Path,
):
    index = RepositoryFileIndexer().build(root)

    exact_searcher = ExactCodeSearcher()

    chunks = CodeChunker(
        max_lines=20,
        overlap_lines=0,
    ).build_chunks(index)

    semantic_index = InMemorySemanticIndex(FakeEmbeddingProvider())

    await semantic_index.build(chunks)

    return (
        index,
        exact_searcher,
        semantic_index,
    )


@pytest.mark.asyncio
async def test_file_hit_by_both_channels_ranked_high(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    (
        index,
        exact_searcher,
        semantic_index,
    ) = await build_retrieval_components(tmp_path)

    query = "user"

    lexical = exact_searcher.search(
        index,
        query,
    )

    semantic = await semantic_index.search(
        query,
        top_k=10,
    )

    hybrid = HybridRetriever().fuse(
        query=query,
        lexical=lexical,
        semantic=semantic,
    )

    assert hybrid.candidates[0].relative_path == "src/user.ts"

    first = hybrid.candidates[0]

    assert first.lexical_rank is not None

    assert first.semantic_rank is not None


@pytest.mark.asyncio
async def test_semantic_only_candidate_is_kept(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    (
        index,
        exact_searcher,
        semantic_index,
    ) = await build_retrieval_components(tmp_path)

    query = "用户"

    lexical = exact_searcher.search(
        index,
        query,
    )

    semantic = await semantic_index.search(
        query,
        top_k=10,
    )

    hybrid = HybridRetriever().fuse(
        query=query,
        lexical=lexical,
        semantic=semantic,
    )

    user_candidate = next(
        item for item in hybrid.candidates if (item.relative_path == "src/user.ts")
    )

    assert user_candidate.lexical_rank is None

    assert user_candidate.semantic_rank is not None


@pytest.mark.asyncio
async def test_lexical_only_candidate_is_kept(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    (
        index,
        exact_searcher,
        semantic_index,
    ) = await build_retrieval_components(tmp_path)

    query = "/api/users"

    lexical = exact_searcher.search(
        index,
        query,
    )

    semantic = await semantic_index.search(
        query,
        top_k=10,
    )

    hybrid = HybridRetriever().fuse(
        query=query,
        lexical=lexical,
        semantic=semantic,
    )

    user_candidate = next(
        item for item in hybrid.candidates if (item.relative_path == "src/user.ts")
    )

    assert user_candidate.lexical_rank is not None


@pytest.mark.asyncio
async def test_semantic_chunks_are_aggregated_by_file(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    content = "\n".join(
        [
            "// user logic",
            "const a = 1",
            "// user service",
            "const b = 2",
            "// user query",
            "const c = 3",
        ]
    )

    (src / "user.ts").write_text(
        content,
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    chunks = CodeChunker(
        max_lines=2,
        overlap_lines=0,
    ).build_chunks(index)

    semantic_index = InMemorySemanticIndex(FakeEmbeddingProvider())

    await semantic_index.build(chunks)

    semantic = await semantic_index.search(
        "用户",
        top_k=10,
    )

    lexical = ExactCodeSearcher().search(
        index,
        "用户",
    )

    hybrid = HybridRetriever().fuse(
        query="用户",
        lexical=lexical,
        semantic=semantic,
    )

    user_candidates = [
        candidate
        for candidate in hybrid.candidates
        if (candidate.relative_path == "src/user.ts")
    ]

    assert len(user_candidates) == 1
