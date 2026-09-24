from pathlib import Path

import pytest

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    ExactSearchResponse,
)


def create_test_repository(
    root: Path,
) -> None:
    src = root / "src"
    src.mkdir()

    api = src / "api"
    api.mkdir()

    views = src / "views"
    views.mkdir()

    utils = src / "utils"
    utils.mkdir()

    public = root / "public"
    public.mkdir()

    (api / "user.ts").write_text(
        """
export async function getUserList() {
    return fetch('/api/users')
}
""".strip(),
        encoding="utf-8",
    )

    (views / "UserList.vue").write_text(
        """
<script setup lang="ts">
import { getUserList } from '../api/user'

async function loadUsers() {
    await getUserList()
}
</script>
""".strip(),
        encoding="utf-8",
    )

    (utils / "date.ts").write_text(
        """
export function formatDate() {
    return ''
}
""".strip(),
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        """
# Demo

UserList is the user management page.
""".strip(),
        encoding="utf-8",
    )

    (public / "logo.png").write_bytes(b"\x89PNG")


def test_search_by_content(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    indexer = RepositoryFileIndexer()

    index = indexer.build(tmp_path)

    searcher = ExactCodeSearcher()

    response = searcher.search(
        index,
        "getUserList",
    )

    assert isinstance(
        response,
        ExactSearchResponse,
    )

    result_paths = [result.relative_path for result in response.results]

    assert "src/api/user.ts" in result_paths

    assert "src/views/UserList.vue" in result_paths

    assert "src/utils/date.ts" not in result_paths


def test_filename_match_ranked_first(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    response = searcher.search(
        index,
        "UserList",
    )

    assert response.results[0].relative_path == "src/views/UserList.vue"

    assert response.results[0].score > 0


def test_search_by_path(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    response = searcher.search(
        index,
        "src/api",
    )

    paths = [result.relative_path for result in response.results]

    assert "src/api/user.ts" in paths


def test_case_insensitive_search(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    response = ExactCodeSearcher().search(
        index,
        "getuserlist",
    )

    paths = {result.relative_path for result in response.results}

    assert "src/api/user.ts" in paths


def test_content_match_contains_line_number(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    response = ExactCodeSearcher().search(
        index,
        "fetch",
    )

    user_api = next(
        result
        for result in response.results
        if (result.relative_path == "src/api/user.ts")
    )

    content_matches = [
        match for match in user_api.matches if (match.match_type == "content")
    ]

    assert len(content_matches) == 1

    match = content_matches[0]

    assert match.line_number == 2

    assert "fetch" in (match.snippet or "")


def test_binary_file_not_content_searched(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    response = ExactCodeSearcher().search(
        index,
        "PNG",
    )

    assert not any(
        result.relative_path == "public/logo.png"
        and any(match.match_type == "content" for match in result.matches)
        for result in response.results
    )


def test_empty_query_rejected(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    with pytest.raises(
        ValueError,
        match="搜索关键词不能为空",
    ):
        searcher.search(
            index,
            "   ",
        )
