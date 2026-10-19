import json

import pytest

from agent_core.change_surface.model import (
    InvestigationArtifact,
    ObservedRepositoryFile,
)
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, ChatResponse
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class StubLLM(LLM):
    def __init__(self, contents):
        if isinstance(contents, str):
            contents = [contents]

        self.contents = list(contents)
        self.calls = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        index = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        return ChatResponse(
            content=self.contents[index],
            model="stub",
        )


def make_bundle():
    code = "<template />"
    return TaskContextBundle(
        requirement=RequirementSpec(
            title="分页",
            summary="切换 pageSize",
            non_goals=["不修改 API"],
        ),
        technical_spec=TechnicalSpec(summary="复用现有能力"),
        task=TaskDefinition(
            task_id="TASK-002",
            task_type="code_change",
            title="分页控件",
            objective="接入 pageSize",
            source_step_order=2,
            depends_on=["TASK-001"],
        ),
        repository=RepositoryProfile(
            project_name="demo",
            root_path=".",
            framework="Vue3",
            language="TypeScript",
        ),
        code_context=[
            CodeContextItem(
                relative_path="src/views/UsersView.vue",
                category="source",
                content_mode="full_file",
                content=code,
                content_chars=len(code),
            )
        ],
        budget=ContextBudgetUsage(
            max_files=6,
            max_total_chars=30000,
            max_chars_per_file=8000,
            used_files=1,
            used_chars=len(code),
        ),
    )


def make_selection():
    items = []

    for rank, path in enumerate(
        [
            "src/views/UsersView.vue",
            "src/stores/users.ts",
            "src/api/users.ts",
        ],
        1,
    ):
        items.append(
            HybridRelevantFile(
                relative_path=path,
                category="source",
                size_bytes=10,
                retrieval_score=1,
                retrieval_rank=rank,
                policy_score=0,
                selection_score=1,
                rerank_rank=rank,
                rerank_score=1 - rank / 10,
            )
        )

    return HybridRelevantFileSelection(
        candidate_count=3,
        eligible_count=3,
        selected_count=3,
        total_selected_bytes=30,
        selected_files=items,
    )


def make_investigation():
    return InvestigationArtifact(
        task_id="TASK-001",
        summary="facts",
        observed_files=[
            ObservedRepositoryFile(
                relative_path="src/api/users.ts",
                observed_role="dependency",
                evidence=["协议已支持"],
                confidence=0.9,
            )
        ],
    )


@pytest.mark.asyncio
async def test_resolver_accepts_markdown_json_fence():
    content = """```json
{
  \"task_id\": \"TASK-002\",
  \"implementation_targets\": [\"src/views/UsersView.vue\"],
  \"dependencies\": [\"src/api/users.ts\"],
  \"references\": [],
  \"tests\": [],
  \"unresolved\": [],
  \"rationale\": []
}
```"""

    resolver = ChangeSurfaceResolver(llm=StubLLM(content))
    surface = await resolver.resolve(
        bundle=make_bundle(),
        selection=make_selection(),
        upstream_investigations=[make_candidate_investigation()],
    )

    assert surface.implementation_targets == ["src/views/UsersView.vue"]


@pytest.mark.asyncio
async def test_resolver_repairs_duplicate_role_assignment():
    invalid = json.dumps(
        {
            "task_id": "TASK-002",
            "implementation_targets": ["src/views/UsersView.vue"],
            "dependencies": [
                "src/views/UsersView.vue",
                "src/api/users.ts",
            ],
            "references": [],
            "tests": [],
            "unresolved": [],
            "rationale": [],
        },
        ensure_ascii=False,
    )

    repaired = json.dumps(
        {
            "task_id": "TASK-002",
            "implementation_targets": ["src/views/UsersView.vue"],
            "dependencies": ["src/api/users.ts"],
            "references": [],
            "tests": [],
            "unresolved": [],
            "rationale": ["UsersView 为修改目标，API 只读"],
        },
        ensure_ascii=False,
    )

    llm = StubLLM([invalid, repaired])
    resolver = ChangeSurfaceResolver(llm=llm)

    surface = await resolver.resolve(
        bundle=make_bundle(),
        selection=make_selection(),
        upstream_investigations=[make_candidate_investigation()],
    )

    assert llm.calls == 2
    assert surface.dependencies == ["src/api/users.ts"]


@pytest.mark.asyncio
async def test_resolver_raises_diagnostic_error_after_failed_repair():
    invalid = "not-json"
    llm = StubLLM([invalid, invalid])
    resolver = ChangeSurfaceResolver(llm=llm)

    with pytest.raises(Exception) as captured:
        await resolver.resolve(
            bundle=make_bundle(),
            selection=make_selection(),
            upstream_investigations=[make_investigation()],
        )

    message = str(captured.value)
    assert "validation_error:" in message
    assert "raw_output_preview:" in message


def make_test_bundle():
    bundle = make_bundle()
    return bundle.model_copy(
        deep=True,
        update={
            "task": bundle.task.model_copy(
                update={
                    "task_id": "TASK-005",
                    "task_type": "test",
                    "title": "补充 usersStore 测试",
                    "objective": "验证 pageSize 切换",
                }
            )
        },
    )


def make_selection_with_test_and_pagination():
    base = make_selection()
    extra = [
        HybridRelevantFile(
            relative_path="src/components/PaginationBar.vue",
            category="source",
            size_bytes=10,
            retrieval_score=0.5,
            retrieval_rank=4,
            policy_score=0,
            selection_score=0.5,
            rerank_rank=4,
            rerank_score=0.7,
        ),
        HybridRelevantFile(
            relative_path="tests/unit/usersStore.test.ts",
            category="test",
            size_bytes=10,
            retrieval_score=0.5,
            retrieval_rank=5,
            policy_score=0,
            selection_score=0.5,
            rerank_rank=5,
            rerank_score=0.6,
        ),
    ]
    files = [*base.selected_files, *extra]
    return base.model_copy(
        deep=True,
        update={
            "candidate_count": len(files),
            "eligible_count": len(files),
            "selected_count": len(files),
            "total_selected_bytes": sum(item.size_bytes for item in files),
            "selected_files": files,
        },
    )


def make_candidate_investigation():
    return InvestigationArtifact(
        task_id="TASK-001",
        summary="facts",
        observed_files=[
            ObservedRepositoryFile(
                relative_path="src/views/UsersView.vue",
                observed_role="candidate_target",
                evidence=["UsersView 当前没有 pageSize 选择控件"],
                confidence=0.95,
            ),
            ObservedRepositoryFile(
                relative_path="src/stores/users.ts",
                observed_role="candidate_target",
                evidence=["store 缺少 pageSize 变更动作"],
                confidence=0.95,
            ),
            ObservedRepositoryFile(
                relative_path="src/api/users.ts",
                observed_role="dependency",
                evidence=["API 已支持 pageSize"],
                confidence=0.9,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_resolver_moves_test_task_changes_into_tests_role():
    content = json.dumps(
        {
            "task_id": "TASK-005",
            "implementation_targets": [
                "tests/unit/usersStore.test.ts",
                "src/stores/users.ts",
            ],
            "dependencies": [],
            "references": [],
            "tests": [],
            "unresolved": [],
            "rationale": [],
            "execution_disposition": None,
        },
        ensure_ascii=False,
    )

    resolver = ChangeSurfaceResolver(llm=StubLLM(content))
    surface = await resolver.resolve(
        bundle=make_test_bundle(),
        selection=make_selection_with_test_and_pagination(),
        upstream_investigations=[make_candidate_investigation()],
    )

    assert surface.implementation_targets == []
    assert surface.tests == ["tests/unit/usersStore.test.ts"]
    assert "src/stores/users.ts" in surface.dependencies
    assert surface.execution_disposition == "change_required"


@pytest.mark.asyncio
async def test_resolver_downgrades_uninvestigated_change_target():
    content = json.dumps(
        {
            "task_id": "TASK-002",
            "implementation_targets": [
                "src/views/UsersView.vue",
                "src/components/PaginationBar.vue",
            ],
            "dependencies": ["src/api/users.ts"],
            "references": [],
            "tests": [],
            "unresolved": [],
            "rationale": [
                "UsersView 缺少控件",
                "PaginationBar 最可能需要适配",
            ],
            "execution_disposition": None,
        },
        ensure_ascii=False,
    )

    resolver = ChangeSurfaceResolver(llm=StubLLM(content))
    surface = await resolver.resolve(
        bundle=make_bundle(),
        selection=make_selection_with_test_and_pagination(),
        upstream_investigations=[make_candidate_investigation()],
    )

    assert surface.implementation_targets == ["src/views/UsersView.vue"]
    assert "src/components/PaginationBar.vue" in surface.dependencies
    assert any(
        "PaginationBar.vue" in item and "缺少足够 Repository Evidence" in item
        for item in surface.unresolved
    )
    assert surface.execution_disposition == "change_required"


@pytest.mark.asyncio
async def test_resolver_allows_code_change_no_op():
    content = json.dumps(
        {
            "task_id": "TASK-002",
            "implementation_targets": [],
            "dependencies": ["src/stores/users.ts"],
            "references": [],
            "tests": [],
            "unresolved": [],
            "rationale": ["现有 totalPages 已基于 total/pageSize 计算，无需修改"],
            "execution_disposition": None,
        },
        ensure_ascii=False,
    )

    resolver = ChangeSurfaceResolver(llm=StubLLM(content))
    surface = await resolver.resolve(
        bundle=make_bundle(),
        selection=make_selection_with_test_and_pagination(),
        upstream_investigations=[make_candidate_investigation()],
    )

    assert surface.implementation_targets == []
    assert surface.execution_disposition == "no_change_required"
    assert surface.requires_patch() is False
