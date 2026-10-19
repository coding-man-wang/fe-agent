from agent_core.change_surface import RoleAwareContextPolicy, ResolvedChangeSurface
from agent_core.codebase.model import HybridRelevantFile, HybridRelevantFileSelection
from agent_core.sdd.model import TaskDefinition


def f(path: str, rank: int) -> HybridRelevantFile:
    category = "test" if path.startswith("tests/") else "source"
    return HybridRelevantFile(
        relative_path=path,
        category=category,
        language="TypeScript",
        size_bytes=100,
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
        policy_score=1,
        selection_score=1.0 / rank,
        rerank_rank=rank,
        rerank_score=1.0 - rank * 0.05,
    )


def test_role_aware_context_drops_unclassified_noise():
    files = [
        f("src/views/UsersView.vue", 1),
        f("src/stores/users.ts", 2),
        f("src/api/users.ts", 3),
        f("src/types/domain.ts", 4),
        f("src/components/PaginationBar.vue", 5),
        f("tests/unit/usersStore.test.ts", 6),
        f("tests/e2e/users.spec.ts", 7),
        f("src/components/UserTable.vue", 8),
    ]
    selection = HybridRelevantFileSelection(
        candidate_count=8,
        eligible_count=8,
        selected_count=8,
        total_selected_bytes=800,
        selected_files=files,
    )
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/views/UsersView.vue",
            "src/stores/users.ts",
        ],
        dependencies=[
            "src/api/users.ts",
            "src/types/domain.ts",
        ],
        references=["src/components/PaginationBar.vue"],
        tests=["tests/unit/usersStore.test.ts"],
    )
    task = TaskDefinition(
        task_id="TASK-002",
        task_type="code_change",
        title="change",
        objective="change",
        source_step_order=2,
    )
    refined = RoleAwareContextPolicy().apply(
        selection=selection,
        surface=surface,
        task=task,
    )
    paths = {item.relative_path for item in refined.selected_files}
    assert "src/components/UserTable.vue" not in paths
    assert "tests/e2e/users.spec.ts" not in paths
    assert "src/views/UsersView.vue" in paths
    assert "src/api/users.ts" in paths
