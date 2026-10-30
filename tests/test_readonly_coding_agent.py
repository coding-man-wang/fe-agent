import json
from pathlib import Path

import pytest

from agent_core.coding.agent import (
    CodingOutputError,
    ReadOnlyCodingAgent,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


class FakeTextModel:

    def __init__(
        self,
        response: dict,
    ) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        self.calls.append(
            {
                "system_message": system_message,
                "user_message": user_message,
            }
        )

        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


def make_valid_response() -> dict:
    return {
        "task_id": "TASK-003",
        "summary": "需要扩展用户列表查询参数",
        "findings": [
            {
                "relative_path": "src/api/user.ts",
                "evidence": "getUserList",
                "explanation": ("当前上下文包含用户列表查询函数"),
            }
        ],
        "proposed_changes": [
            {
                "relative_path": "src/api/user.ts",
                "objective": "增加 status 参数",
                "proposed_approach": ("在现有请求函数中扩展查询参数"),
            }
        ],
        "verification_plan": ["验证 status 是否传递到 API"],
        "missing_information": [],
        "execution_status": "not_executed",
    }


@pytest.mark.asyncio
async def test_readonly_agent_returns_proposal(
    tmp_path: Path,
):
    model = FakeTextModel(make_valid_response())

    agent = ReadOnlyCodingAgent(model=model)

    bundle = create_test_bundle(tmp_path)

    result = await agent.analyze(bundle)

    assert result.task_id == "TASK-003"

    assert result.execution_status == "not_executed"

    assert len(result.proposed_changes) == 1

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_agent_uses_rendered_context(
    tmp_path: Path,
):
    model = FakeTextModel(make_valid_response())

    agent = ReadOnlyCodingAgent(model=model)

    await agent.analyze(create_test_bundle(tmp_path))

    call = model.calls[0]

    payload = json.loads(call["user_message"])

    assert payload["task_id"] == "TASK-003"

    assert payload["code_context"][0]["relative_path"] == "src/api/user.ts"

    assert "只返回一个合法 JSON 对象" in call["system_message"]


@pytest.mark.asyncio
async def test_rejects_unknown_file(
    tmp_path: Path,
):
    response = make_valid_response()

    response["proposed_changes"][0]["relative_path"] = "src/unknown.ts"

    agent = ReadOnlyCodingAgent(model=FakeTextModel(response))

    bundle = create_test_bundle(tmp_path)

    # ==================================
    # 当前测试不是验证：
    #
    #   Change Surface 拒绝未知路径
    #
    # 而是验证：
    #
    #   即使路径已经获得 Write Permission，
    #   action=modify 仍必须存在于
    #   Current Code Context。
    #
    # 所以这里显式模拟：
    # src/unknown.ts 已被 Change Surface
    # 批准为 writable target，
    # 但 Code Context 中不存在该文件。
    # ==================================

    surface = dict(bundle.change_surface or {})

    surface["implementation_targets"] = [
        "src/api/user.ts",
        "src/unknown.ts",
    ]

    bundle = bundle.model_copy(
        deep=True,
        update={
            "change_surface": surface,
        },
    )

    with pytest.raises(
        CodingOutputError,
        match="未提供的已有文件",
    ):
        await agent.analyze(bundle)


@pytest.mark.asyncio
async def test_rejects_fabricated_evidence(
    tmp_path: Path,
):
    response = make_valid_response()

    response["findings"][0]["evidence"] = "deleteAllUsers"

    agent = ReadOnlyCodingAgent(model=FakeTextModel(response))

    with pytest.raises(
        CodingOutputError,
        match="evidence",
    ):
        await agent.analyze(create_test_bundle(tmp_path))


@pytest.mark.asyncio
async def test_rejects_wrong_task_id(
    tmp_path: Path,
):
    response = make_valid_response()

    response["task_id"] = "TASK-999"

    agent = ReadOnlyCodingAgent(model=FakeTextModel(response))

    with pytest.raises(
        CodingOutputError,
        match="task_id",
    ):
        await agent.analyze(create_test_bundle(tmp_path))


@pytest.mark.asyncio
async def test_rejects_executed_status(
    tmp_path: Path,
):
    response = make_valid_response()

    response["execution_status"] = "completed"

    agent = ReadOnlyCodingAgent(model=FakeTextModel(response))

    with pytest.raises(
        CodingOutputError,
        match="契约",
    ):
        await agent.analyze(create_test_bundle(tmp_path))
