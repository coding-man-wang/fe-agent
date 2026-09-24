from pathlib import Path

from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalDecision,
    TechnicalSpec,
)


def create_requirement() -> RequirementSpec:
    return RequirementSpec(
        title="用户列表状态筛选",
        summary=("为用户列表增加状态筛选能力"),
        goals=["支持按用户状态筛选"],
        non_goals=["不修改用户详情页"],
        acceptance_criteria=["用户可以按启用状态筛选"],
        constraints=["复用现有列表 API"],
        assumptions=[],
        open_questions=[],
    )


def create_technical_spec() -> TechnicalSpec:
    return TechnicalSpec(
        summary=("在现有用户列表查询链路中" "增加 status 参数"),
        solution_approach=[
            "扩展查询参数",
            "页面增加状态筛选",
        ],
        affected_modules=[
            "src/api",
            "src/views",
        ],
        reuse_plan=["复用现有用户列表 API"],
        data_flow=["筛选条件 -> 页面 -> API"],
        technical_decisions=[
            TechnicalDecision(
                topic="状态管理",
                decision=("筛选状态保持在页面局部状态"),
                rationale=("当前状态没有跨页面共享需求"),
            )
        ],
        testing_strategy=[
            "验证 API 参数",
            "验证页面筛选",
        ],
        risks=[],
        assumptions=[],
        open_questions=[],
    )


def create_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户列表查询",
        objective=("支持 status 查询参数"),
        source_step_order=2,
        depends_on=[],
        inputs=[
            "RequirementSpec",
            "TechnicalSpec",
        ],
        target_modules=[
            "src/api",
            "src/views",
        ],
        expected_outputs=["用户列表支持状态筛选"],
        verification=["筛选状态正确传递给 API"],
        acceptance_criteria_refs=["用户可以按启用状态筛选"],
    )


def create_repository_profile(
        root: Path,
) -> RepositoryProfile:
    return RepositoryProfile(
        project_name="admin-web",
        root_path=str(root.resolve()),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
        state_management="Pinia",
        test_stack=[
            "Vitest",
            "Playwright",
        ],
        scripts={
            "dev": "vite",
            "test": "vitest",
        },
        dependencies=[
            "vue",
            "pinia",
        ],
        config_files=[
            "package.json",
            "vite.config.ts",
        ],
    )


def create_selection(
        *,
        path: str,
        size_bytes: int,
        semantic_chunk: CodeChunk | None = None,
) -> HybridRelevantFileSelection:
    relevant_file = HybridRelevantFile(
        relative_path=path,
        category="source",
        language="TypeScript",
        size_bytes=size_bytes,
        retrieval_score=0.03,
        retrieval_rank=1,
        policy_score=55,
        policy_rank=1,
        selection_score=0.02,
        matched_queries=["getUserList"],
        best_semantic_chunk=(semantic_chunk),
        reasons=[
            "Hybrid Retrieval rank=1",
            "匹配 target_modules: src/api",
        ],
    )

    return HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=(size_bytes),
        selected_files=[relevant_file],
        skipped_by_budget=[],
        skipped_non_indexable=[],
    )


def test_small_file_uses_full_content(
        tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    api = src / "api"
    api.mkdir()

    file_path = api / "user.ts"

    content = """
export function getUserList() {
    return []
}
""".strip()

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    selection = create_selection(
        path="src/api/user.ts",
        size_bytes=len(content.encode("utf-8")),
    )

    builder = ProjectContextBuilder()

    context = builder.build(
        repository_root=tmp_path,
        requirement=create_requirement(),
        technical_spec=(create_technical_spec()),
        task=create_task(),
        repository=(create_repository_profile(tmp_path)),
        selection=selection,
        engineering_rules=["API 请求必须放在 src/api"],
        max_files=5,
        max_total_chars=10_000,
        max_chars_per_file=1_000,
    )

    assert len(context.code_context) == 1

    item = context.code_context[0]

    assert item.content_mode == "full_file"

    assert item.content == content

    assert context.budget.used_files == 1


def test_large_file_uses_semantic_chunk(
        tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    views = src / "views"
    views.mkdir()

    file_path = views / "UserList.vue"

    large_content = "const unrelated = 1\n" * 500

    file_path.write_text(
        large_content,
        encoding="utf-8",
    )

    semantic_chunk = CodeChunk(
        chunk_id=("src/views/" "UserList.vue:100-110"),
        relative_path=("src/views/UserList.vue"),
        start_line=100,
        end_line=110,
        content="""
async function loadUsers() {
    return getUserList(filters)
}
""".strip(),
        language="Vue",
        category="source",
    )

    selection = create_selection(
        path=("src/views/UserList.vue"),
        size_bytes=len(large_content.encode("utf-8")),
        semantic_chunk=(semantic_chunk),
    )

    context = ProjectContextBuilder().build(
        repository_root=tmp_path,
        requirement=(create_requirement()),
        technical_spec=(create_technical_spec()),
        task=create_task(),
        repository=(create_repository_profile(tmp_path)),
        selection=selection,
        max_files=5,
        max_total_chars=10_000,
        max_chars_per_file=100,
    )

    item = context.code_context[0]

    assert item.content_mode == "semantic_chunk"

    assert "getUserList" in item.content

    assert item.start_line == 100

    assert "src/views/UserList.vue" in context.budget.compressed_files


def test_large_file_without_chunk_is_truncated(
        tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    file_path = src / "large.ts"

    content = "const value = 1;\n" * 500

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    selection = create_selection(
        path="src/large.ts",
        size_bytes=len(content.encode("utf-8")),
    )

    context = ProjectContextBuilder().build(
        repository_root=tmp_path,
        requirement=(create_requirement()),
        technical_spec=(create_technical_spec()),
        task=create_task(),
        repository=(create_repository_profile(tmp_path)),
        selection=selection,
        max_files=5,
        max_total_chars=200,
        max_chars_per_file=100,
    )

    item = context.code_context[0]

    assert item.content_mode == "truncated"

    assert item.content_chars == 100
