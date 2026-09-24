from pathlib import Path

import pytest

from agent_core.codebase.chunker import (
    CodeChunker,
)
from agent_core.codebase.embedding import (
    EmbeddingProvider,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
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
                # 用户列表语义
                1.0 if ("user" in normalized or "用户" in normalized) else 0.0,
                # 登录认证语义
                (
                    1.0
                    if (
                            "login" in normalized
                            or "auth" in normalized
                            or "登录" in normalized
                    )
                    else 0.0
                ),
                # 支付语义
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
export function loadUsers() {
    // load user list
}
""".strip(),
        encoding="utf-8",
    )

    (src / "auth.ts").write_text(
        """
export function handleUnauthorized() {
    // login authentication
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


def test_build_code_chunks(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    chunker = CodeChunker(
        max_lines=2,
        overlap_lines=1,
    )

    chunks = chunker.build_chunks(index)

    assert len(chunks) > 0

    assert all(chunk.start_line >= 1 for chunk in chunks)

    assert all(chunk.end_line >= chunk.start_line for chunk in chunks)


@pytest.mark.asyncio
async def test_semantic_search_user(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    chunks = CodeChunker(
        max_lines=20,
        overlap_lines=0,
    ).build_chunks(index)

    semantic_index = InMemorySemanticIndex(embedding_provider=(FakeEmbeddingProvider()))

    await semantic_index.build(chunks)

    response = await semantic_index.search(
        "用户列表相关代码",
        top_k=2,
    )

    assert len(response.results) == 2

    assert response.results[0].chunk.relative_path == "src/user.ts"


@pytest.mark.asyncio
async def test_semantic_search_auth(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    chunks = CodeChunker().build_chunks(index)

    semantic_index = InMemorySemanticIndex(FakeEmbeddingProvider())

    await semantic_index.build(chunks)

    response = await semantic_index.search(
        "登录认证逻辑",
        top_k=1,
    )

    assert response.results[0].chunk.relative_path == "src/auth.ts"
