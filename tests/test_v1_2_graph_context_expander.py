from pathlib import Path

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.graph import (
    RepositoryDependencyGraphBuilder,
    RepositoryGraphContextExpander,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)


def _find_record(
    index: RepositoryFileIndex,
    relative_path: str,
):
    for item in index.files:
        if item.relative_path == relative_path:
            return item

    raise AssertionError(f"FileRecord 不存在: {relative_path}")


def _retrieval_file(
    *,
    index: RepositoryFileIndex,
    relative_path: str,
    rank: int,
    target_match_type: str = "none",
) -> HybridRelevantFile:
    """
    构造测试用 Retrieval 文件。

    它模拟 TaskRetrievalService 最终返回的
    HybridRelevantFile。
    """

    record = _find_record(
        index,
        relative_path,
    )

    return HybridRelevantFile(
        relative_path=(record.relative_path),
        category=record.category,
        language=record.language,
        size_bytes=record.size_bytes,
        retrieval_score=1.0,
        retrieval_rank=rank,
        policy_score=0,
        policy_rank=None,
        selection_score=1.0,
        best_semantic_chunk=None,
        best_semantic_score=None,
        query_hit_count=1,
        lexical_hit_count=1,
        semantic_hit_count=0,
        hybrid_hit_count=0,
        matched_queries=[relative_path],
        matched_intents=["test"],
        target_match_type=(target_match_type),
        target_match_value=(relative_path if target_match_type != "none" else None),
        context_role="unclassified",
        role_reason=None,
        rerank_score=None,
        rerank_rank=None,
        reasons=["test retrieval"],
    )


def _selection(
    files: list[HybridRelevantFile],
) -> HybridRelevantFileSelection:
    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(item.size_bytes for item in files),
        selected_files=files,
        skipped_by_budget=[],
        skipped_non_indexable=[],
    )


def test_expander_adds_graph_dependencies(
    tmp_path: Path,
):
    src = tmp_path / "src"

    (src / "views").mkdir(parents=True)
    (src / "stores").mkdir(parents=True)
    (src / "api").mkdir(parents=True)

    (src / "views" / "UsersView.vue").write_text(
        (
            '<script setup lang="ts">\n'
            "import { useUsersStore } "
            "from '@/stores/users'\n"
            "</script>\n"
        ),
        encoding="utf-8",
    )

    (src / "stores" / "users.ts").write_text(
        (
            "import { listUsers } "
            "from '@/api/users'\n"
            "export const "
            "useUsersStore = () => "
            "listUsers()\n"
        ),
        encoding="utf-8",
    )

    (src / "api" / "users.ts").write_text(
        ("export function " "listUsers() { " "return [] }\n"),
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(file_index=index)

    base = _selection(
        [
            _retrieval_file(
                index=index,
                relative_path=("src/views/" "UsersView.vue"),
                rank=1,
            )
        ]
    )

    expander = RepositoryGraphContextExpander(
        max_files=3,
        max_graph_files=2,
        dependency_depth=2,
        dependent_depth=1,
    )

    result = expander.expand(
        selection=base,
        index=index,
        graph=graph,
    )

    paths = [item.relative_path for item in result.selected_files]

    assert paths == [
        "src/views/UsersView.vue",
        "src/stores/users.ts",
        "src/api/users.ts",
    ]

    users_store = next(
        item
        for item in result.selected_files
        if item.relative_path == "src/stores/users.ts"
    )

    assert users_store.context_role == "dependency"

    assert users_store.target_match_type == "none"

    assert users_store.retrieval_score == 0.0

    assert any("Graph dependency" in reason for reason in users_store.reasons)


def test_expander_adds_graph_dependents(
    tmp_path: Path,
):
    src = tmp_path / "src"

    (src / "views").mkdir(parents=True)
    (src / "router").mkdir(parents=True)

    (src / "views" / "UsersView.vue").write_text(
        ("<template>" "<div>users</div>" "</template>\n"),
        encoding="utf-8",
    )

    (src / "router" / "index.ts").write_text(
        (
            "import UsersView "
            "from '@/views/UsersView.vue'\n"
            "export default UsersView\n"
        ),
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(file_index=index)

    base = _selection(
        [
            _retrieval_file(
                index=index,
                relative_path=("src/views/" "UsersView.vue"),
                rank=1,
            )
        ]
    )

    expander = RepositoryGraphContextExpander(
        max_files=2,
        max_graph_files=1,
        dependency_depth=1,
        dependent_depth=1,
    )

    result = expander.expand(
        selection=base,
        index=index,
        graph=graph,
    )

    paths = [item.relative_path for item in result.selected_files]

    assert paths == [
        "src/views/UsersView.vue",
        "src/router/index.ts",
    ]

    router_file = next(
        item
        for item in result.selected_files
        if item.relative_path == "src/router/index.ts"
    )

    assert router_file.context_role == "dependency"

    assert router_file.target_match_type == "none"

    assert any("Graph dependent" in reason for reason in router_file.reasons)


def test_expander_deduplicates_existing_retrieval_files(
    tmp_path: Path,
):
    src = tmp_path / "src"

    (src / "views").mkdir(parents=True)
    (src / "stores").mkdir(parents=True)

    (src / "views" / "UsersView.vue").write_text(
        (
            '<script setup lang="ts">\n'
            "import { useUsersStore } "
            "from '@/stores/users'\n"
            "</script>\n"
        ),
        encoding="utf-8",
    )

    (src / "stores" / "users.ts").write_text(
        ("export const " "useUsersStore = () => ({})\n"),
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(file_index=index)

    base = _selection(
        [
            _retrieval_file(
                index=index,
                relative_path=("src/views/" "UsersView.vue"),
                rank=1,
            ),
            _retrieval_file(
                index=index,
                relative_path=("src/stores/users.ts"),
                rank=2,
            ),
        ]
    )

    expander = RepositoryGraphContextExpander(
        max_files=4,
        max_graph_files=2,
        dependency_depth=1,
        dependent_depth=1,
    )

    result = expander.expand(
        selection=base,
        index=index,
        graph=graph,
    )

    paths = [item.relative_path for item in result.selected_files]

    assert paths == [
        "src/views/UsersView.vue",
        "src/stores/users.ts",
    ]

    assert len(paths) == len(set(paths))

    assert result.selected_count == 2


def test_expander_respects_context_budget_and_keeps_target_evidence(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    (src / "anchor.ts").write_text(
        ("import './graph-a'\n" "import './graph-b'\n" "import './graph-c'\n"),
        encoding="utf-8",
    )

    for name in [
        "graph-a.ts",
        "graph-b.ts",
        "graph-c.ts",
        "retrieval-2.ts",
        "retrieval-3.ts",
        "retrieval-4.ts",
        "retrieval-5.ts",
        "retrieval-6.ts",
        "retrieval-7.ts",
        "target.ts",
    ]:
        (src / name).write_text(
            ("export const value = " f"'{name}'\n"),
            encoding="utf-8",
        )

    index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(file_index=index)

    retrieval_paths = [
        "src/anchor.ts",
        "src/retrieval-2.ts",
        "src/retrieval-3.ts",
        "src/retrieval-4.ts",
        "src/retrieval-5.ts",
        "src/retrieval-6.ts",
        "src/retrieval-7.ts",
        "src/target.ts",
    ]

    retrieval_files = []

    for rank, path in enumerate(
        retrieval_paths,
        start=1,
    ):
        retrieval_files.append(
            _retrieval_file(
                index=index,
                relative_path=path,
                rank=rank,
                target_match_type=("exact_file" if path == "src/target.ts" else "none"),
            )
        )

    base = _selection(retrieval_files)

    expander = RepositoryGraphContextExpander(
        max_files=8,
        max_graph_files=3,
        dependency_depth=1,
        dependent_depth=1,
    )

    result = expander.expand(
        selection=base,
        index=index,
        graph=graph,
    )

    paths = [item.relative_path for item in result.selected_files]

    # ---------------------------------
    # 总 Context Budget 必须严格 <= 8。
    # ---------------------------------

    assert len(paths) == 8

    assert result.selected_count == 8

    # ---------------------------------
    # Planner 明确 Target Evidence
    # 即使原本排在最后，也不能被 Graph 挤掉。
    # ---------------------------------

    assert "src/target.ts" in paths

    # ---------------------------------
    # Graph 最多补 3 个。
    # ---------------------------------

    assert "src/graph-a.ts" in paths

    assert "src/graph-b.ts" in paths

    assert "src/graph-c.ts" in paths

    # ---------------------------------
    # 5 Retrieval + 3 Graph
    # ---------------------------------

    graph_files = [
        item
        for item in result.selected_files
        if any(
            reason.startswith("Graph dependency:")
            or reason.startswith("Graph dependent:")
            for reason in item.reasons
        )
    ]

    assert len(graph_files) == 3

    # ---------------------------------
    # 被 Graph 挤掉的普通 Retrieval
    # 应记录进 skipped_by_budget。
    # ---------------------------------

    assert "src/retrieval-5.ts" in result.skipped_by_budget

    assert "src/retrieval-6.ts" in result.skipped_by_budget

    assert "src/retrieval-7.ts" in result.skipped_by_budget

    # Target 绝不能同时出现在 skipped。
    assert "src/target.ts" not in result.skipped_by_budget

    # ---------------------------------
    # Selection 统计信息重新计算。
    # ---------------------------------

    assert result.total_selected_bytes == sum(
        item.size_bytes for item in result.selected_files
    )
