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
    SemanticSearchResponse,
    SemanticSearchResult,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def create_repository(
        root: Path,
) -> None:
    api = root / "src" / "api"
    api.mkdir(parents=True)

    views = root / "src" / "views"
    views.mkdir(parents=True)

    tests = root / "tests"
    tests.mkdir()

    (api / "user.ts").write_text(
        ("export function " "getUserList(status?: string) {}\n"),
        encoding="utf-8",
    )

    (views / "UserList.vue").write_text(
        (
            '<script setup lang="ts">\n'
            "import { getUserList } "
            "from '../api/user'\n"
            "</script>\n"
        ),
        encoding="utf-8",
    )

    (tests / "user.spec.ts").write_text(
        ("describe('user list', " "() => {})\n"),
        encoding="utf-8",
    )


def create_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-003",
        task_type="code_change",
        title="修改用户查询逻辑",
        objective=("给用户列表 API " "增加 status 筛选"),
        source_step_order=2,
        depends_on=[],
        inputs=[
            "现有用户列表 API",
        ],
        target_modules=[
            "src/api",
        ],
        expected_outputs=[
            "用户 API 支持 status",
        ],
        verification=[
            "用户列表正常加载",
        ],
        acceptance_criteria_refs=[],
    )


class FakeSemanticSearcher:

    def __init__(self):
        self.queries: list[str] = []

    async def search(
            self,
            query: str,
            *,
            top_k: int = 5,
    ) -> SemanticSearchResponse:

        self.queries.append(query)

        if "status" in query.lower() or "用户" in query:
            results = [
                SemanticSearchResult(
                    chunk=CodeChunk(
                        chunk_id=("src/api/user.ts:1-1"),
                        relative_path=("src/api/user.ts"),
                        start_line=1,
                        end_line=1,
                        content=(
                            "export function " "getUserList(" "status?: string" ") {}"
                        ),
                        language="TypeScript",
                        category="source",
                    ),
                    score=0.95,
                )
            ]

        else:
            results = []

        return SemanticSearchResponse(
            query=query,
            results=results[:top_k],
        )


@pytest.mark.asyncio
async def test_task_retrieval_selects_relevant_file(
        tmp_path: Path,
):
    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    semantic = FakeSemanticSearcher()

    service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=semantic,
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=6,
        max_files=5,
        max_total_bytes=100_000,
    )

    selection = await service.retrieve(
        task=create_task(),
        index=index,
    )

    selected_paths = {item.relative_path for item in selection.selected_files}

    assert "src/api/user.ts" in selected_paths

    assert semantic.queries


def test_build_queries_is_deterministic():
    service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
    )

    task = create_task()

    queries = service.build_queries(task)

    assert queries == [
        "修改用户查询逻辑",
        ("给用户列表 API " "增加 status 筛选"),
        "src/api",
        "现有用户列表 API",
    ]


def test_build_queries_deduplicates():
    service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
    )

    task = create_task().model_copy(
        update={
            "title": "src/api",
            "target_modules": [
                "src/api",
                "SRC/API",
            ],
        }
    )

    queries = service.build_queries(task)

    assert queries.count("src/api") == 1

    assert "SRC/API" not in queries


def test_query_budget_limits_query_count():
    service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
        max_queries=2,
    )

    queries = service.build_queries(create_task())

    assert len(queries) == 2
