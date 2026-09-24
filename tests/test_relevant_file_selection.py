from pathlib import Path

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.relevance import (
    RelevantFileSelector,
)
from agent_core.sdd.model import (
    TaskDefinition,
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

    tests = root / "tests"
    tests.mkdir()

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
    return getUserList()
}
</script>
""".strip(),
        encoding="utf-8",
    )

    (tests / "user.spec.ts").write_text(
        """
describe('UserList', () => {
    // test getUserList
})
""".strip(),
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        """
# User Management

UserList uses getUserList to load users.
""".strip(),
        encoding="utf-8",
    )


def create_code_change_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户列表请求逻辑",
        objective=("调整用户列表的数据查询逻辑"),
        source_step_order=2,
        depends_on=[],
        inputs=["TechnicalSpec"],
        target_modules=[
            "src/api",
            "src/views",
        ],
        expected_outputs=["更新后的用户列表功能"],
        verification=["用户列表能够正常加载"],
        acceptance_criteria_refs=[],
    )


def test_select_relevant_files(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    responses = [
        searcher.search(
            index,
            "getUserList",
        ),
        searcher.search(
            index,
            "UserList",
        ),
        searcher.search(
            index,
            "/api/users",
        ),
    ]

    selector = RelevantFileSelector()

    selection = selector.select(
        index=index,
        search_responses=responses,
        task=create_code_change_task(),
        max_files=3,
        max_total_bytes=10_000,
    )

    assert selection.selected_count == 3

    selected_paths = [file.relative_path for file in selection.selected_files]

    assert "src/api/user.ts" in selected_paths

    assert "src/views/UserList.vue" in selected_paths


def test_code_change_prefers_source_files(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    responses = [
        searcher.search(
            index,
            "UserList",
        )
    ]

    selector = RelevantFileSelector()

    selection = selector.select(
        index=index,
        search_responses=responses,
        task=create_code_change_task(),
        max_files=10,
        max_total_bytes=100_000,
    )

    paths = [file.relative_path for file in selection.selected_files]

    assert paths.index("src/views/UserList.vue") < paths.index("README.md")


def test_multi_query_match_gets_bonus(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    responses = [
        searcher.search(
            index,
            "getUserList",
        ),
        searcher.search(
            index,
            "/api/users",
        ),
    ]

    selection = RelevantFileSelector().select(
        index=index,
        search_responses=responses,
        task=create_code_change_task(),
        max_files=10,
        max_total_bytes=100_000,
    )

    user_api = next(
        file
        for file in selection.selected_files
        if (file.relative_path == "src/api/user.ts")
    )

    assert len(user_api.matched_queries) == 2

    assert any("2 个不同 Query" in reason for reason in user_api.reasons)


def test_max_files_budget(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    responses = [
        searcher.search(
            index,
            "UserList",
        ),
        searcher.search(
            index,
            "getUserList",
        ),
    ]

    selection = RelevantFileSelector().select(
        index=index,
        search_responses=responses,
        task=create_code_change_task(),
        max_files=2,
        max_total_bytes=100_000,
    )

    assert selection.selected_count == 2

    assert len(selection.skipped_by_budget) >= 1


def test_size_budget(
    tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    searcher = ExactCodeSearcher()

    responses = [
        searcher.search(
            index,
            "getUserList",
        )
    ]

    selector = RelevantFileSelector()

    selection = selector.select(
        index=index,
        search_responses=responses,
        task=create_code_change_task(),
        max_files=10,
        # 故意设得非常小
        max_total_bytes=10,
    )

    assert selection.total_selected_bytes <= 10
