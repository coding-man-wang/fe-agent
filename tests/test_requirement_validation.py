import pytest

from agent_core.llm.model import ChatResponse
from agent_core.sdd.model import RequirementSpec
from agent_core.sdd.validation import RequireValidator


class FakeLLM:
    async def chat(self, request):
        return ChatResponse(
            content="""
            {
                "is_ready": false,
                "issues": [
                    {
                        "category": "missing",
                        "severity": "blocking",
                        "message": "缺少导出格式说明",
                        "related_field": "open_questions"
                    }
                ],
                "clarification_questions": [
                    "导出文件需要使用 Excel 还是 CSV？"
                ]
            }
            """,
            tool_calls=[],
            model="fake-model",
        )


@pytest.mark.asyncio
async def test_requirement_not_ready():
    spec = RequirementSpec(
        title="用户导出",
        summary="支持用户列表导出",
        goals=[
            "支持导出用户数据"
        ],
        acceptance_criteria=[
            "用户可以完成数据导出"
        ],
        open_questions=[
            "导出格式是什么？"
        ],
    )

    validator = RequireValidator(
        llm=FakeLLM()
    )

    result = await validator.validate(
        spec
    )

    assert result.is_ready is False

    assert len(
        result.clarification_questions
    ) == 1

    assert (
            result.issues[0].severity
            == "blocking"
    )


def test_patch_does_not_modify_original():
    original = RequirementSpec(
        title="用户列表",
        summary="用户列表页面",
        goals=[
            "展示用户"
        ],
        open_questions=[
            "是否需要分页？"
        ],
    )

    patch = RequirementPatch(
        open_questions=StringListPatch(
            remove=[
                "是否需要分页？"
            ]
        )
    )

    updated = apply_requirement_patch(
        original,
        patch,
    )

    assert original.open_questions == [
        "是否需要分页？"
    ]

    assert updated.open_questions == []
