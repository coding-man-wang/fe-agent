from pathlib import Path

import pytest

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)
from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)
from agent_core.codebase.model import (
    CodeChunk,
    RepositoryProfile,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.context.budget_provider import (
    StaticContextBudgetProvider,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
)
from agent_core.context.factory import (
    DefaultTaskContextFactory,
)
from agent_core.context.model import (
    ContextBudgetConfig,
)
from agent_core.context.rules_provider import (
    StaticEngineeringRulesProvider,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class FakeSemanticSearcher:

    async def search(
            self,
            query: str,
            *,
            top_k: int = 5,
    ) -> SemanticSearchResponse:
        return SemanticSearchResponse(
            query=query,
            results=[
                SemanticSearchResult(
                    chunk=CodeChunk(
                        chunk_id=("src/api/user.ts:1-1"),
                        relative_path=("src/api/user.ts"),
                        start_line=1,
                        end_line=1,
                        content=("export function " "getUserList() {}"),
                        language=("TypeScript"),
                        category="source",
                    ),
                    score=0.95,
                )
            ][:top_k],
        )


def create_repository(
        root: Path,
) -> None:
    api = root / "src" / "api"

    api.mkdir(parents=True)

    (api / "user.ts").write_text(
        ("export function " "getUserList() {}\n"),
        encoding="utf-8",
    )


def create_requirement():
    return RequirementSpec(
        title="用户状态筛选",
        summary=("用户列表支持 status 筛选"),
        goals=[
            "支持 status 查询",
        ],
        acceptance_criteria=[
            ("用户列表可以按 " "status 过滤"),
        ],
    )


def create_technical_spec():
    return TechnicalSpec(
        summary=("扩展用户列表 API"),
        solution_approach=[
            ("为用户查询增加 " "status 参数"),
        ],
        affected_modules=[
            "src/api",
        ],
    )


def create_task():
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户查询逻辑",
        objective=("给用户列表 API " "增加 status 筛选"),
        source_step_order=1,
        target_modules=[
            "src/api",
        ],
        inputs=[
            "现有用户列表 API",
        ],
        expected_outputs=[
            "用户查询支持 status",
        ],
        verification=[
            "用户列表正常加载",
        ],
    )


@pytest.mark.asyncio
async def test_context_factory_builds_real_context(
        tmp_path: Path,
):
    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    repository = RepositoryProfile(
        project_name="test-web",
        root_path=str(tmp_path),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
    )

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
    )

    factory = DefaultTaskContextFactory(
        repository_root=(tmp_path),
        requirement=(create_requirement()),
        technical_spec=(create_technical_spec()),
        repository=(repository),
        index=index,
        retrieval_service=(retrieval_service),
        context_builder=(ProjectContextBuilder()),
        rules_provider=(StaticEngineeringRulesProvider(["保持现有 API " "行为兼容"])),
        budget_provider=(
            StaticContextBudgetProvider(
                ContextBudgetConfig(
                    max_files=5,
                    max_total_chars=10_000,
                    max_chars_per_file=5_000,
                )
            )
        ),
    )

    bundle = await factory.build(create_task())

    assert bundle.task.task_id == "TASK-003"

    assert bundle.repository.project_name == "test-web"

    assert bundle.code_context

    paths = {item.relative_path for item in bundle.code_context}

    assert "src/api/user.ts" in paths

    user_context = next(
        item
        for item in bundle.code_context
        if (item.relative_path == "src/api/user.ts")
    )

    assert "getUserList" in user_context.content

    assert bundle.engineering_rules == ["保持现有 API " "行为兼容"]

    assert bundle.budget.used_files >= 1

    assert bundle.budget.used_chars > 0


@pytest.mark.asyncio
async def test_context_factory_respects_char_budget(
        tmp_path: Path,
):
    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    repository = RepositoryProfile(
        project_name="test-web",
        root_path=str(tmp_path),
    )

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
    )

    factory = DefaultTaskContextFactory(
        repository_root=tmp_path,
        requirement=(create_requirement()),
        technical_spec=(create_technical_spec()),
        repository=repository,
        index=index,
        retrieval_service=(retrieval_service),
        context_builder=(ProjectContextBuilder()),
        rules_provider=(StaticEngineeringRulesProvider([])),
        budget_provider=(
            StaticContextBudgetProvider(
                ContextBudgetConfig(
                    max_files=1,
                    max_total_chars=10,
                    max_chars_per_file=10,
                )
            )
        ),
    )

    bundle = await factory.build(create_task())

    assert bundle.budget.used_files <= 1

    assert bundle.budget.used_chars <= 10
