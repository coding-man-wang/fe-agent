import json
import pytest

from agent_core.change_surface.model import InvestigationArtifact, ObservedRepositoryFile
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.codebase.model import HybridRelevantFile, HybridRelevantFileSelection, RepositoryProfile
from agent_core.context.model import CodeContextItem, ContextBudgetUsage, TaskContextBundle
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, ChatResponse
from agent_core.sdd.model import RequirementSpec, TaskDefinition, TechnicalSpec


class StubLLM(LLM):
    def __init__(self, content): self.content=content
    async def chat(self, request: ChatRequest)->ChatResponse:
        return ChatResponse(content=self.content, model="stub")


def make_bundle():
    code="<template />"
    return TaskContextBundle(
        requirement=RequirementSpec(title="分页", summary="切换 pageSize", non_goals=["不修改 API"]),
        technical_spec=TechnicalSpec(summary="复用现有能力"),
        task=TaskDefinition(task_id="TASK-002", task_type="code_change", title="分页控件", objective="接入 pageSize", source_step_order=2, depends_on=["TASK-001"]),
        repository=RepositoryProfile(project_name="demo", root_path=".", framework="Vue3", language="TypeScript"),
        code_context=[CodeContextItem(relative_path="src/views/UsersView.vue", category="source", content_mode="full_file", content=code, content_chars=len(code))],
        budget=ContextBudgetUsage(max_files=6,max_total_chars=30000,max_chars_per_file=8000,used_files=1,used_chars=len(code)),
    )


def make_selection():
    items=[]
    for rank,path in enumerate(["src/views/UsersView.vue","src/stores/users.ts","src/api/users.ts"],1):
        items.append(HybridRelevantFile(relative_path=path,category="source",size_bytes=10,retrieval_score=1,retrieval_rank=rank,policy_score=0,selection_score=1,rerank_rank=rank,rerank_score=1-rank/10))
    return HybridRelevantFileSelection(candidate_count=3,eligible_count=3,selected_count=3,total_selected_bytes=30,selected_files=items)


@pytest.mark.asyncio
async def test_resolver_separates_target_and_dependency():
    llm=StubLLM(json.dumps({
        "task_id":"TASK-002","implementation_targets":["src/views/UsersView.vue","src/stores/users.ts"],
        "dependencies":["src/api/users.ts"],"references":[],"tests":[],"unresolved":[],"rationale":["API 只读"]
    },ensure_ascii=False))
    inv=InvestigationArtifact(task_id="TASK-001",summary="facts",observed_files=[ObservedRepositoryFile(relative_path="src/api/users.ts",observed_role="dependency",evidence=["协议已支持"],confidence=.9)])
    surface=await ChangeSurfaceResolver(llm=llm).resolve(bundle=make_bundle(),selection=make_selection(),upstream_investigations=[inv])
    assert surface.implementation_targets == ["src/views/UsersView.vue","src/stores/users.ts"]
    assert surface.dependencies == ["src/api/users.ts"]
