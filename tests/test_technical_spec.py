import pytest

from agent_core.llm.base import LLM
from agent_core.llm.model import ChatResponse
from agent_core.sdd.model import (
    ProjectContext, TechnicalSpec,
)
from agent_core.sdd.technical import (
    TechnicalSpecGenerator,
)


def test_project_context():
    context = ProjectContext(
        project_name="admin-web",
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
        state_management="Pinia",
        test_stack=[
            "Vitest",
            "Playwright",
        ],
        available_components=[
            "Table",
            "SearchForm",
        ],
    )

    assert context.framework == "Vue3"

    assert "Table" in (
        context.available_components
    )


def test_technical_spec():
    data = {
        "summary": "实现用户列表页面",

        "solution_approach": [
            "新增用户列表页面"
        ],

        "affected_modules": [
            "用户管理"
        ],

        "reuse_plan": [
            "复用 Table"
        ],

        "data_flow": [
            "搜索条件 -> API -> Table"
        ],

        "technical_decisions": [
            {
                "topic": "状态管理",
                "decision": "使用局部状态",
                "rationale": "不需要跨页面共享"
            }
        ],

        "testing_strategy": [
            "使用 Playwright 验证搜索"
        ],

        "risks": [],

        "assumptions": [],

        "open_questions": []
    }

    spec = TechnicalSpec.model_validate(
        data
    )

    assert spec.summary == (
        "实现用户列表页面"
    )

    assert (
            spec.technical_decisions[0].topic
            == "状态管理"
    )


class FakeTechnicalLLM(LLM):

    async def chat(
            self,
            request,
    ):
        return ChatResponse(
            content="""
            {
              "summary":
                "实现用户列表页面",

              "solution_approach": [
                "增加用户列表页面",
                "调用用户分页 API"
              ],

              "affected_modules": [
                "用户管理"
              ],

              "reuse_plan": [
                "复用 Table"
              ],

              "data_flow": [
                "查询条件 -> API -> 用户列表"
              ],

              "technical_decisions": [
                {
                  "topic": "状态管理",
                  "decision": "使用页面局部状态",
                  "rationale":
                    "查询条件不需要跨页面共享"
                }
              ],

              "testing_strategy": [
                "使用 Playwright 验证搜索和分页"
              ],

              "risks": [],

              "assumptions": [],

              "open_questions": []
            }
            """,
            tool_calls=[],
            model="fake",
        )


@pytest.mark.asyncio
async def test_generate_technical_spec():
    generator = TechnicalSpecGenerator(
        llm=FakeTechnicalLLM()
    )

    technical_spec = await generator.generate(
        requirement=requirement,
        project_context=project_context,
    )

    assert (
            technical_spec.summary
            == "实现用户列表页面"
    )

    assert len(
        technical_spec.technical_decisions
    ) == 1
