import asyncio
import json

import pytest

from agent_core.change_surface import (
    InvestigationArtifactBuildError,
    InvestigationArtifactBuilder,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.llm.base import LLM
from agent_core.llm.model import (
    ChatRequest,
    ChatResponse,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class StubLLM(LLM):

    def __init__(
        self,
        *,
        content: str,
    ) -> None:
        self.content = content

        self.last_request: ChatRequest | None = None

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        self.last_request = request

        return ChatResponse(
            content=self.content,
            model="stub",
        )


def make_bundle(
    *,
    task_type: str = "investigation",
) -> TaskContextBundle:

    requirement = RequirementSpec(
        title=("用户列表每页条数切换"),
        summary=("用户列表支持 10/20 " "pageSize 切换"),
        goals=["支持 10 和 20"],
        non_goals=["不修改 API 请求协议"],
        acceptance_criteria=["切换 pageSize 后 " "重新加载列表"],
        constraints=["复用已有 pageSize " "能力"],
    )

    technical_spec = TechnicalSpec(
        summary=("复用现有分页和 " "pageSize 能力"),
        solution_approach=["确认 pageSize " "现有流转方式"],
        affected_modules=[
            "src/views",
            "src/stores",
            "src/api",
        ],
        reuse_plan=["复用现有 users store"],
        data_flow=["UsersView -> " "users store -> API"],
    )

    task = TaskDefinition(
        task_id="TASK-001",
        task_type=task_type,
        title=("梳理现有分页与 " "pageSize 流转链路"),
        objective=("确认 pageSize " "当前实现与依赖"),
        source_step_order=1,
        target_modules=[
            "src/views",
            "src/stores",
            "src/api",
        ],
        expected_outputs=["pageSize 流转链路"],
        verification=["明确现有实现位置"],
    )

    repository = RepositoryProfile(
        project_name=("fe-agent-benchmark-web"),
        root_path="D:/repo",
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
        state_management="Pinia",
    )

    code_context = [
        CodeContextItem(
            relative_path=("src/views/" "UsersView.vue"),
            category="source",
            language="Vue",
            content_mode="full_file",
            content="""
<script setup lang="ts">
import { useUsersStore } from "../stores/users"

const usersStore = useUsersStore()
</script>
""".strip(),
            start_line=1,
            end_line=5,
            relevance_reasons=["用户列表相关"],
            content_chars=(len("""
<script setup lang="ts">
import { useUsersStore } from "../stores/users"

const usersStore = useUsersStore()
</script>
""".strip())),
        ),
        CodeContextItem(
            relative_path=("src/stores/users.ts"),
            category="source",
            language="TypeScript",
            content_mode="full_file",
            content="""
export const useUsersStore = defineStore("users", () => {
  const page = ref(1)
  const pageSize = ref(10)

  async function fetchUsers() {
    return listUsers({
      page: page.value,
      pageSize: pageSize.value,
    })
  }

  return {
    page,
    pageSize,
    fetchUsers,
  }
})
""".strip(),
            start_line=1,
            end_line=17,
            relevance_reasons=["pageSize semantic hit"],
            content_chars=(len("""
export const useUsersStore = defineStore("users", () => {
  const page = ref(1)
  const pageSize = ref(10)

  async function fetchUsers() {
    return listUsers({
      page: page.value,
      pageSize: pageSize.value,
    })
  }

  return {
    page,
    pageSize,
    fetchUsers,
  }
})
""".strip())),
        ),
        CodeContextItem(
            relative_path=("src/api/users.ts"),
            category="source",
            language="TypeScript",
            content_mode="full_file",
            content="""
export function listUsers(
  query: UserQuery
) {
  return request.get("/users", {
    params: query,
  })
}
""".strip(),
            start_line=1,
            end_line=7,
            relevance_reasons=["用户查询 API"],
            content_chars=(len("""
export function listUsers(
  query: UserQuery
) {
  return request.get("/users", {
    params: query,
  })
}
""".strip())),
        ),
    ]

    budget = ContextBudgetUsage(
        max_files=8,
        max_total_chars=30_000,
        max_chars_per_file=8_000,
        used_files=3,
        used_chars=sum(item.content_chars for item in code_context),
        skipped_files=[],
        compressed_files=[],
    )

    return TaskContextBundle(
        requirement=requirement,
        technical_spec=(technical_spec),
        task=task,
        repository=repository,
        engineering_rules=["优先复用已有实现"],
        code_context=code_context,
        budget=budget,
    )


def test_builder_creates_artifact():

    content = json.dumps(
        {
            "task_id": "TASK-001",
            "summary": ("pageSize 当前由 " "users store 管理。"),
            "observed_files": [
                {
                    "relative_path": ("src/stores/users.ts"),
                    "observed_role": ("candidate_target"),
                    "evidence": [
                        ("users store 定义 " "pageSize，默认值为 10"),
                        ("fetchUsers 将 " "page 和 pageSize " "传给 listUsers"),
                    ],
                    "confidence": 0.95,
                },
                {
                    "relative_path": ("src/api/users.ts"),
                    "observed_role": ("dependency"),
                    "evidence": [
                        "listUsers 已接收 " "UserQuery 并直接作为 " "请求参数"
                    ],
                    "confidence": 0.9,
                },
            ],
            "findings": [
                ("pageSize 已存在于 " "users store"),
                ("API 层已经透传 " "UserQuery"),
            ],
            "unresolved_questions": ["选择控件最终应放在 " "UsersView 还是分页组件"],
        },
        ensure_ascii=False,
    )

    llm = StubLLM(
        content=content,
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    artifact = asyncio.run(
        builder.build(
            bundle=make_bundle(),
        )
    )

    assert artifact.task_id == "TASK-001"

    assert len(artifact.observed_files) == 2

    assert artifact.observed_files[0].relative_path == "src/stores/users.ts"

    assert llm.last_request is not None


def test_builder_sends_real_code_to_llm():

    content = json.dumps(
        {
            "task_id": "TASK-001",
            "summary": "完成分析",
            "observed_files": [],
            "findings": [],
            "unresolved_questions": [],
        },
        ensure_ascii=False,
    )

    llm = StubLLM(
        content=content,
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    asyncio.run(
        builder.build(
            bundle=make_bundle(),
        )
    )

    assert llm.last_request is not None

    user_message = llm.last_request.messages[1].content

    assert "src/stores/users.ts" in user_message

    assert "const pageSize = ref(10)" in user_message


def test_builder_rejects_non_investigation_task():

    llm = StubLLM(
        content="{}",
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    with pytest.raises(InvestigationArtifactBuildError):
        asyncio.run(
            builder.build(
                bundle=make_bundle(
                    task_type=("code_change"),
                ),
            )
        )


def test_builder_rejects_hallucinated_file():

    content = json.dumps(
        {
            "task_id": "TASK-001",
            "summary": "完成分析",
            "observed_files": [
                {
                    "relative_path": ("src/fake/" "NotExists.ts"),
                    "observed_role": ("dependency"),
                    "evidence": ["模型声称该文件存在"],
                    "confidence": 0.9,
                }
            ],
            "findings": [],
            "unresolved_questions": [],
        },
        ensure_ascii=False,
    )

    llm = StubLLM(
        content=content,
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    with pytest.raises(InvestigationArtifactBuildError):
        asyncio.run(
            builder.build(
                bundle=make_bundle(),
            )
        )


def test_builder_rejects_wrong_task_id():

    content = json.dumps(
        {
            "task_id": "TASK-999",
            "summary": "错误 Task",
            "observed_files": [],
            "findings": [],
            "unresolved_questions": [],
        },
        ensure_ascii=False,
    )

    llm = StubLLM(
        content=content,
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    with pytest.raises(InvestigationArtifactBuildError):
        asyncio.run(
            builder.build(
                bundle=make_bundle(),
            )
        )


def test_builder_normalizes_windows_path():

    content = json.dumps(
        {
            "task_id": "TASK-001",
            "summary": "完成分析",
            "observed_files": [
                {
                    "relative_path": ("src\\stores\\users.ts"),
                    "observed_role": ("candidate_target"),
                    "evidence": ["users store 定义 " "pageSize"],
                    "confidence": 0.9,
                }
            ],
            "findings": [],
            "unresolved_questions": [],
        },
        ensure_ascii=False,
    )

    llm = StubLLM(
        content=content,
    )

    builder = InvestigationArtifactBuilder(
        llm=llm,
    )

    artifact = asyncio.run(
        builder.build(
            bundle=make_bundle(),
        )
    )

    assert artifact.observed_files[0].relative_path == "src/stores/users.ts"
