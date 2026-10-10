from pathlib import Path

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)
from agent_core.codebase.model import (
    HybridFileCandidate,
    HybridSearchResponse,
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
export function getUserList() {}
""".strip(),
        encoding="utf-8",
    )

    (views / "UserList.vue").write_text(
        """
<script setup lang="ts">
</script>
""".strip(),
        encoding="utf-8",
    )

    (tests / "user.spec.ts").write_text(
        """
describe('user', () => {})
""".strip(),
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        "# User Management",
        encoding="utf-8",
    )

    (root / "logo.png").write_bytes(b"\x89PNG")


def create_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户查询逻辑",
        objective="修改用户列表 API",
        source_step_order=2,
        depends_on=[],
        inputs=[],
        target_modules=["src/api"],
        expected_outputs=["更新后的用户查询逻辑"],
        verification=["用户列表正常加载"],
        acceptance_criteria_refs=[],
    )


def make_candidate(
        path: str,
        *,
        fusion_score: float = 0.03,
) -> HybridFileCandidate:
    return HybridFileCandidate(
        relative_path=path,
        fusion_score=fusion_score,
        reasons=[],
    )


def test_multi_query_retrieval_evidence(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="getUserList",
            candidates=[
                make_candidate("src/api/user.ts"),
                make_candidate("README.md"),
            ],
        ),
        HybridSearchResponse(
            query="/api/users",
            candidates=[
                make_candidate("src/api/user.ts"),
            ],
        ),
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
        max_files=10,
        max_total_bytes=100_000,
    )

    user_api = next(
        item
        for item in selection.selected_files
        if (item.relative_path == "src/api/user.ts")
    )

    assert len(user_api.matched_queries) == 2

    assert user_api.retrieval_rank == 1


def test_task_policy_prefers_target_module(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="user",
            candidates=[
                make_candidate("src/views/UserList.vue"),
                make_candidate("src/api/user.ts"),
            ],
        )
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
        max_files=10,
        max_total_bytes=100_000,
    )

    user_api = next(
        item
        for item in selection.selected_files
        if (item.relative_path == "src/api/user.ts")
    )

    assert user_api.policy_score > 0

    assert any("target_modules" in reason for reason in user_api.reasons)


def test_non_indexable_file_is_skipped(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="logo",
            candidates=[
                make_candidate("logo.png"),
            ],
        )
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
    )

    assert selection.selected_count == 0

    assert "logo.png" in selection.skipped_non_indexable


def test_max_files_budget(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="user",
            candidates=[
                make_candidate("src/api/user.ts"),
                make_candidate("src/views/UserList.vue"),
                make_candidate("tests/user.spec.ts"),
                make_candidate("README.md"),
            ],
        )
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
        max_files=2,
        max_total_bytes=100_000,
    )

    assert selection.selected_count == 2

    assert len(selection.skipped_by_budget) == 2


def test_byte_budget(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="user",
            candidates=[
                make_candidate("src/api/user.ts"),
                make_candidate("src/views/UserList.vue"),
            ],
        )
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
        max_files=10,
        max_total_bytes=10,
    )

    assert selection.total_selected_bytes <= 10


def test_other_category_does_not_break_policy_scoring(
        tmp_path: Path,
):
    create_test_repository(tmp_path)

    other_file = tmp_path / "data.txt"

    other_file.write_text(
        "user data",
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    responses = [
        HybridSearchResponse(
            query="user",
            candidates=[
                make_candidate("data.txt"),
            ],
        )
    ]

    selection = HybridRelevantFileSelector().select(
        index=index,
        hybrid_responses=responses,
        task=create_task(),
        max_files=10,
        max_total_bytes=100_000,
    )

    assert selection.selected_count == 1

    selected = selection.selected_files[0]

    assert selected.relative_path == "data.txt"

    assert selected.policy_score == 0
