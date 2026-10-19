import json
import pytest

from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.errors import InvestigationArtifactBuildError
from agent_core.codebase.model import RepositoryProfile
from agent_core.context.model import CodeContextItem, ContextBudgetUsage, TaskContextBundle
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, ChatResponse
from agent_core.sdd.model import RequirementSpec, TaskDefinition, TechnicalSpec


class StubLLM(LLM):
    def __init__(self, content: str):
        self.content = content
    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content=self.content, model="stub")


def bundle():
    content = "const pageSize = ref(10)"
    return TaskContextBundle(
        requirement=RequirementSpec(title="分页", summary="支持 pageSize", constraints=[]),
        technical_spec=TechnicalSpec(summary="复用 pageSize"),
        task=TaskDefinition(
            task_id="TASK-001", task_type="investigation", title="排查分页",
            objective="确认 pageSize", source_step_order=1,
        ),
        repository=RepositoryProfile(project_name="demo", root_path=".", framework="Vue3", language="TypeScript"),
        code_context=[CodeContextItem(
            relative_path="src/stores/users.ts", category="source", language="TypeScript",
            content_mode="full_file", content=content, start_line=1, end_line=1,
            content_chars=len(content),
        )],
        budget=ContextBudgetUsage(
            max_files=6, max_total_chars=30000, max_chars_per_file=8000,
            used_files=1, used_chars=len(content),
        ),
    )


@pytest.mark.asyncio
async def test_builds_only_from_provided_files():
    llm = StubLLM(json.dumps({
        "task_id":"TASK-001", "summary":"确认 pageSize",
        "observed_files":[{
            "relative_path":"src/stores/users.ts", "observed_role":"candidate_target",
            "evidence":["定义 pageSize=10"], "confidence":0.95,
        }], "findings":["pageSize 已存在"], "unresolved_questions":[],
    }, ensure_ascii=False))
    artifact = await InvestigationArtifactBuilder(llm=llm).build(bundle=bundle())
    assert artifact.observed_files[0].relative_path == "src/stores/users.ts"


@pytest.mark.asyncio
async def test_rejects_hallucinated_file():
    llm = StubLLM(json.dumps({
        "task_id":"TASK-001", "summary":"x",
        "observed_files":[{
            "relative_path":"src/fake.ts", "observed_role":"dependency",
            "evidence":["fake"], "confidence":0.8,
        }], "findings":[], "unresolved_questions":[],
    }))
    with pytest.raises(InvestigationArtifactBuildError):
        await InvestigationArtifactBuilder(llm=llm).build(bundle=bundle())
